from contextlib import contextmanager
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import engine
from .prepare import check_shards, check_token_map


def request(url, payload=None, timeout=3):
    data = json.dumps(payload).encode() if payload is not None else None
    req = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ready(url, model):
    try:
        return any(
            row.get("id") == model for row in request(url + "/v1/models")["data"]
        )
    except (URLError, TimeoutError, OSError, KeyError, ValueError):
        return False


def validate(s):
    check_shards(s)
    check_token_map(s, s.assets / "engram-token-map.bin")
    manifest = json.loads((s.engram / "engram-manifest.json").read_text())
    if sorted(t["layer"] for t in manifest["tables"]) != [1, 14]:
        raise ValueError("Engram manifest must include layers 1 and 14")
    for table in manifest["tables"]:
        path = s.engram / table["file"]
        if path.stat().st_size != table["size"]:
            raise ValueError(f"incorrect packed table size for {path}")
    engine.verify_build(s)
    if not os.access(s.command()[0], os.X_OK):
        raise FileNotFoundError(
            f"missing executable {s.command()[0]}. Run python -m mac build"
        )


@contextmanager
def state_lock(s):
    s.state.mkdir(parents=True, exist_ok=True)
    with (s.state / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def identity(pid):
    if type(pid) is not int or pid <= 1:
        raise ValueError("invalid managed server PID")
    result = subprocess.run(
        ["ps", "-ww", "-p", str(pid), "-o", "lstart=", "-o", "comm="],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def registered(s):
    path = s.state / "process.json"
    if path.exists():
        record = json.loads(path.read_text())
        stamp = identity(record["pid"])
        if stamp and stamp == record["identity"]:
            return record
    return None


def serve(s):
    validate(s)
    args = s.command()
    os.execve(args[0], args, s.server_env())


def up(s, timeout, *, launch_env=None):
    settings = s.launch_settings(launch_env)
    with state_lock(s):
        record = registered(s)
        if record:
            if record["model"] != s.name or record["command"] != s.command():
                raise RuntimeError(
                    f"PID {record['pid']} already serves {record['model']}. Stop it before switching configurations."
                )
            if launch_env is not None and record["settings"] != settings:
                raise RuntimeError(
                    "An existing managed process uses different launch settings"
                )
            print(f"Already started (PID {record['pid']})", flush=True)
        else:
            validate(s)
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((s.host, s.port))
            log_path = s.state / "server.log"
            with log_path.open("ab") as log:
                log_offset = log.tell()
                process = subprocess.Popen(
                    s.command(),
                    env=s.server_env(launch_env),
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            record = {
                "pid": process.pid,
                "identity": identity(process.pid),
                "url": s.url,
                "model": s.name,
                "command": s.command(),
                "settings": settings,
                "log_offset": log_offset,
                "source_repo": s.repo,
                "source_revision": s.revision,
                "source_model": str(s.model),
            }
            if not record["identity"] or process.poll() is not None:
                raise RuntimeError(f"server exited at startup. See {log_path}")
            pending = s.state / "process.tmp"
            try:
                pending.write_text(json.dumps(record) + "\n")
                pending.replace(s.state / "process.json")
            except OSError:
                process.terminate()
                raise
            print(f"Started PID {process.pid}. Log {log_path}", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if identity(record["pid"]) != record["identity"]:
            raise RuntimeError(f"server exited. See {s.state / 'server.log'}")
        if ready(record["url"], record["model"]):
            print(f"Ready at {record['url']}")
            return
        time.sleep(1)
    raise RuntimeError(
        f"readiness timed out. Check python -m mac status and {s.state / 'server.log'}"
    )


def down(s):
    with state_lock(s):
        record = registered(s)
        if not record:
            print("No managed server is running")
            return
        pid = record["pid"]
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 60
        while identity(pid) == record["identity"]:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"PID {pid} is still shutting down. See {s.state / 'server.log'}"
                )
            time.sleep(0.25)
        (s.state / "process.json").unlink(missing_ok=True)
        print(f"Stopped PID {pid}")


def status(s):
    with state_lock(s):
        record = registered(s)
    available = ready(s.url, s.name)
    print(
        json.dumps(
            {
                "pid": record["pid"] if record else None,
                "managed": bool(record),
                "ready": available,
                "url": s.url,
                "model": s.name,
                "running_model": record["model"] if record else None,
            }
        )
    )
    return 0 if available else 1


def chat(s, prompt, tokens):
    result = request(
        s.url + "/v1/chat/completions",
        {
            "model": s.name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": tokens,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=300,
    )
    text = result["choices"][0]["message"].get("content")
    if not text or not text.strip():
        raise ValueError(f"empty chat completion: {result}")
    return result
