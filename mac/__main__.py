"""Build and serve DeepSeek-V4.1-Flash-1017 on a 512 GB Mac Studio."""

import argparse
from collections import deque
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from urllib.error import HTTPError

from . import engine, prepare, repack, server
from .config import ROOT, load, validate_serving

SERVING = {"command", "settings", "validate", "serve", "up"}


def main():
    parser = argparse.ArgumentParser(prog="python -m mac", description=__doc__)
    parser.add_argument(
        "--drafts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable adaptive DSpark drafting or decode serially",
    )
    parser.add_argument(
        "--prompt-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="reuse verified prompt state with a 4 GiB budget; disable for cold controls",
    )
    parser.add_argument(
        "--context", type=int, help="maximum prompt plus reply tokens"
    )
    parser.add_argument(
        "--parallel", type=int, help="number of concurrent client slots (1-16)"
    )
    commands = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in [
        ("build", "build the pinned TensorFold v1.3.3 Metal engine and asset tools"),
        ("fetch", "download the original DeepSeek checkpoint, resuming existing files"),
        ("prepare", "build the token map and pack or verify the SSD Engram tables"),
        ("validate", "check local engine, model and asset paths and checksums"),
        ("settings", "show effective serving options and native command"),
        ("command", "print the environment and command for foreground serving"),
        ("serve", "run the server in the foreground"),
        ("down", "stop the server started by this recipe"),
        ("status", "report process identity and endpoint readiness"),
    ]:
        commands.add_parser(name, help=help_text)
    rp = commands.add_parser(
        "repack",
        help="losslessly repack the checkpoint into 128B/16KiB-aligned per-consumer shards (omitting packed Engram)",
    )
    rp.add_argument(
        "--from",
        dest="source",
        type=Path,
        help="source official checkpoint directory (default: configured model_dir)",
    )
    rp.add_argument(
        "--to",
        dest="dest",
        type=Path,
        help="destination repacked directory (default: .local/checkpoint-repacked)",
    )
    rp.add_argument(
        "--workers",
        type=int,
        default=repack.DEFAULT_WORKERS,
        help="parallel shard writer threads (default: 10)",
    )
    rp.add_argument(
        "--verify",
        action="store_true",
        help="spot-verify repacked tensor bytes against the source checkpoint",
    )
    rp.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing destination directory",
    )
    up = commands.add_parser(
        "up", help="start the server in the background and wait for readiness"
    )
    up.add_argument(
        "--timeout", type=int, default=600, help="readiness deadline in seconds"
    )
    smoke = commands.add_parser("smoke", help="send a real, greedy chat completion")
    smoke.add_argument(
        "--prompt", default="What is 2 + 2? Reply with just the number."
    )
    smoke.add_argument("--max-tokens", type=int, default=16)
    chat = commands.add_parser(
        "chat", help="send one prompt and print the assistant's reply"
    )
    chat.add_argument("prompt")
    chat.add_argument("--max-tokens", type=int, default=128)
    logs = commands.add_parser(
        "logs", help="show the latest lines from the managed server log"
    )
    logs.add_argument("-n", type=int, default=40)
    bench = commands.add_parser(
        "bench",
        help="benchmark streamed completions using TensorFold's bench_openai",
    )
    bench.add_argument("--tokens", type=int, default=64)
    bench.add_argument("--reps", type=int, default=3)
    bench.add_argument("--temperatures", default="0")
    bench.add_argument("--label", help="label for the benchmark rows")
    bench.add_argument("--output", type=Path)
    args = parser.parse_args()
    for key in ("max_tokens", "timeout", "reps", "tokens"):
        if getattr(args, key, 1) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    s = load()
    serving_flags = {
        key: getattr(args, key)
        for key in ("context", "parallel", "drafts")
        if getattr(args, key) is not None
    }
    if args.prompt_cache is not None:
        serving_flags["prompt_cache_gib"] = 4 if args.prompt_cache else 0
    if serving_flags and args.cmd not in SERVING:
        parser.error(
            "--context, --parallel, --[no-]drafts and --[no-]prompt-cache apply to settings, command, validate, serve and up"
        )
    s = replace(s, **serving_flags)
    validate_serving(s.context, s.parallel)
    if args.cmd == "build":
        engine.build(s)
    elif args.cmd == "fetch":
        prepare.fetch(s)
    elif args.cmd == "prepare":
        prepare.prepare(s)
    elif args.cmd == "repack":
        src = args.source.expanduser().resolve() if args.source else s.model
        dst = (
            args.dest.expanduser().resolve()
            if args.dest
            else (ROOT / ".local/checkpoint-repacked").resolve()
        )
        manifest = repack.repack_checkpoint(
            src,
            dst,
            workers=args.workers,
            expected_source_index_sha256=prepare.INDEX_SHA256,
            force=args.force,
        )
        repack.verify_repack(dst, expected_source_index_sha256=prepare.INDEX_SHA256)
        if args.verify:
            repack.verify_against_source(src, dst)
        print(
            f"Repacked {manifest['tensor_count']} tensors into {manifest['shard_count']} shards "
            f"({manifest['total_file_bytes'] / (1 << 30):.2f} GiB) at {dst}"
        )
    elif args.cmd == "settings":
        manifest = engine.load_manifest()
        print(
            json.dumps(
                {
                    "engine": {
                        "tag": s.engine_tag,
                        "commit": s.engine_sha,
                        "source_tree": manifest["source_tree"],
                    },
                    "model": {
                        "repo": s.repo,
                        "revision": s.revision,
                        "path": str(s.model),
                    },
                    "serve": {
                        "context": s.context,
                        "parallel": s.parallel,
                        "drafts": s.drafts,
                        "prompt_cache": s.prompt_cache_gib > 0,
                    },
                    "native_argv": s.command(),
                    "native_env": s.launch_settings(),
                },
                indent=2,
            )
        )
    elif args.cmd == "command":
        settings = [
            f"{key}={shlex.quote(value)}"
            for key, value in s.launch_settings().items()
        ]
        print(" ".join([*settings, shlex.join(s.command())]))
    elif args.cmd == "validate":
        s.launch_settings()
        server.validate(s)
        print("Engine, checkpoint and Engram files verified")
    elif args.cmd == "serve":
        server.serve(s)
    elif args.cmd == "up":
        server.up(s, args.timeout, launch_env=os.environ)
    elif args.cmd == "down":
        server.down(s)
    elif args.cmd == "status":
        return server.status(s)
    elif args.cmd in {"smoke", "chat"}:
        result = server.chat(s, args.prompt, args.max_tokens)
        if args.cmd == "smoke":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["choices"][0]["message"]["content"])
    elif args.cmd == "logs":
        path = s.state / "server.log"
        if path.exists():
            with path.open(errors="replace") as f:
                print("".join(deque(f, maxlen=args.n)), end="")
        else:
            print(f"No server log at {path}")
    elif args.cmd == "bench":
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = args.output or ROOT / "runs" / f"bench-{now}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                s.source / "tools/bench_openai.py",
                s.url,
                s.name,
                "--tokens",
                str(args.tokens),
                "--reps",
                str(args.reps),
                "--temperatures",
                args.temperatures,
                "--label",
                args.label or s.name,
                "--output",
                output,
            ],
            check=True,
        )
        print(f"Benchmark written to {output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except HTTPError as err:
        sys.exit(f"HTTP {err.code}: {err.read().decode(errors='replace')[:1000]}")
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as err:
        sys.exit(str(err))
