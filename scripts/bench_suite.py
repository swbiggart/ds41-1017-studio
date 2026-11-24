"""Run reproducible parity and performance benchmarks for the Mac Studio v1.3.3 recipe."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
from threading import Barrier
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mac import config, engine, server

SEVEN_WORKLOADS = [
    "quicksort-raw",
    "rust-reverse-words",
    "multiply-17-23",
    "js-debounce-raw",
    "linear-equation",
    "sql-left-join",
    "fishing-village",
]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def load_ids(path: Path) -> list[int]:
    raw = path.read_bytes()
    count = len(raw) // 4
    return list(struct.unpack(f"<{count}I", raw))


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def post_json(url: str, body: dict, timeout: int = 900) -> dict:
    req = Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


class ManagedServer:
    def __init__(
        self,
        binary: Path,
        s: config.Settings,
        log_path: Path,
        env_override: dict | None = None,
    ):
        self.binary = binary
        self.s = s
        self.log_path = log_path
        self.env_override = env_override
        self.proc = None

    def __enter__(self):
        argv = list(self.s.command())
        argv[0] = str(self.binary)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("wb")
        env = self.env_override if self.env_override is not None else self.s.server_env()
        self.proc = subprocess.Popen(
            argv,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Server {self.binary} exited early ({self.proc.returncode}); see {self.log_path}"
                )
            if server.ready(self.s.url, self.s.name):
                return self
            time.sleep(0.5)
        raise RuntimeError(f"Server readiness timed out; see {self.log_path}")

    def __exit__(self, *_):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()


def call_prompt(
    base_url: str,
    model: str,
    item: dict,
    max_tokens: int,
    seed: int = 1234,
    temperature: float = 0,
    extra_sampling: dict | None = None,
):
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "return_token_ids": True,
    }
    if extra_sampling:
        body.update(extra_sampling)
    if item["kind"] == "chat":
        url = base_url + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base_url + "/v1/completions"
        body["prompt"] = item["prompt"]
    t0 = time.perf_counter()
    data = post_json(url, body)
    wall_s = time.perf_counter() - t0
    rt = data["tensorfold"]
    token_ids = rt["token_ids"]
    choice = data["choices"][0]
    return {
        "name": item["name"],
        "prompt_tokens": data["usage"]["prompt_tokens"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "cached_tokens": rt.get("cached", 0),
        "drafts": rt.get("drafts"),
        "ttft_s": rt["time_to_first_token"],
        "decode_s": rt["seconds"],
        "decode_tps": rt.get("tokens_per_second"),
        "wall_s": round(wall_s, 4),
        "finish_reason": choice["finish_reason"],
        "token_ids": token_ids,
        "token_sha256": ids_sha256(token_ids),
    }


def warmup(base_url: str, model: str):
    post_json(
        base_url + "/v1/completions",
        {"model": model, "prompt": "Hello", "max_tokens": 4, "temperature": 0},
    )


def run_decode_suite(binary: Path, base: config.Settings, output: Path, max_tokens: int):
    prompts = {
        row["name"]: row
        for row in json.loads((ROOT / "assets/prompts/bench_prompts.json").read_text())
    }
    selected = [prompts[name] for name in SEVEN_WORKLOADS]
    runs = {}
    for label, drafts in (("drafted", True), ("serial", False)):
        s = replace(
            base,
            port=free_port(),
            context=8192,
            parallel=1,
            drafts=drafts,
            prompt_cache_gib=0,
        )
        log = ROOT / "runs" / f"bench-decode-{label}.log"
        with ManagedServer(binary, s, log):
            warmup(s.url, s.name)
            rows = [call_prompt(s.url, s.name, item, max_tokens) for item in selected]
        committed_after_first = sum(max(0, r["completion_tokens"] - 1) for r in rows)
        total_decode_s = sum(r["decode_s"] for r in rows)
        pooled_tps = (
            round(committed_after_first / total_decode_s, 2)
            if total_decode_s > 0
            else 0.0
        )
        runs[label] = {
            "drafts": drafts,
            "pooled_decode_tps": pooled_tps,
            "committed_tokens_total": sum(r["completion_tokens"] for r in rows),
            "committed_after_first": committed_after_first,
            "total_decode_s": round(total_decode_s, 4),
            "requests": rows,
        }
        print(
            f"[decode:{label}] pooled={pooled_tps:.2f} tok/s across {len(rows)} workloads",
            flush=True,
        )

    matched_tokens = 0
    total_tokens = 0
    matched_finishes = 0
    for d_row, s_row in zip(runs["drafted"]["requests"], runs["serial"]["requests"]):
        assert d_row["name"] == s_row["name"]
        total_tokens += len(s_row["token_ids"])
        if d_row["token_ids"] == s_row["token_ids"]:
            matched_tokens += len(s_row["token_ids"])
        else:
            for a, b in zip(d_row["token_ids"], s_row["token_ids"]):
                if a == b:
                    matched_tokens += 1
                else:
                    break
        if d_row["finish_reason"] == s_row["finish_reason"]:
            matched_finishes += 1

    parity = {
        "matched_tokens": matched_tokens,
        "total_tokens": total_tokens,
        "exact_token_match": matched_tokens == total_tokens,
        "matched_finishes": matched_finishes,
        "total_requests": len(selected),
    }
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": display_path(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "max_tokens": max_tokens,
        "parity_drafted_vs_serial": parity,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[decode] parity={parity} -> {output}", flush=True)
    if not parity["exact_token_match"] or matched_finishes != len(selected):
        raise RuntimeError(f"Drafted vs serial parity mismatch: {parity}")


def run_long_suite(binary: Path, base: config.Settings, output: Path, max_tokens: int):
    s = replace(
        base,
        port=free_port(),
        context=139264,
        parallel=1,
        drafts=True,
        prompt_cache_gib=4,
    )
    log = ROOT / "runs" / "bench-long.log"
    cases = [
        ("python-32k", ROOT / "assets/prompts/code_python-32768-1.ids"),
        ("python-131k", ROOT / "assets/prompts/code_python-131072-1.ids"),
    ]
    results = {}
    with ManagedServer(binary, s, log):
        warmup(s.url, s.name)
        for name, path in cases:
            ids = load_ids(path)
            item = {"name": name, "kind": "completion", "prompt": ids}
            cold = call_prompt(s.url, s.name, item, max_tokens)
            warm = call_prompt(s.url, s.name, item, max_tokens)
            exact_cache_match = (
                cold["token_ids"] == warm["token_ids"]
                and cold["finish_reason"] == warm["finish_reason"]
            )
            results[name] = {
                "prompt_file": path.name,
                "prompt_tokens": len(ids),
                "cold": cold,
                "warm_cache_hit": warm,
                "exact_cache_match": exact_cache_match,
            }
            print(
                f"[long:{name}] cold_ttft={cold['ttft_s']:.3f}s decode={cold['decode_tps']:.2f} tok/s "
                f"| warm_ttft={warm['ttft_s']:.4f}s cached={warm['cached_tokens']} exact={exact_cache_match}",
                flush=True,
            )
            if not exact_cache_match:
                raise RuntimeError(f"Cache hit output diverged on {name}")
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": display_path(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "serve": {
            "context": s.context,
            "parallel": s.parallel,
            "drafts": s.drafts,
            "prompt_cache_gib": s.prompt_cache_gib,
        },
        "max_tokens": max_tokens,
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[long] written -> {output}", flush=True)


def run_parity_vs_stock(
    opt_binary: Path,
    stock_binary: Path,
    base: config.Settings,
    output: Path,
    stock_model: Path | None = None,
):
    ids_32k = load_ids(ROOT / "assets/prompts/code_python-32768-1.ids")
    prompts = {
        row["name"]: row
        for row in json.loads((ROOT / "assets/prompts/bench_prompts.json").read_text())
    }
    cases = [
        {"name": "python-32k", "kind": "completion", "prompt": ids_32k, "max_tokens": 32},
        {**prompts["quicksort-raw"], "max_tokens": 64},
        {**prompts["multiply-17-23"], "max_tokens": 64},
    ]
    arms = {}
    for label, binary in (("stock_v133", stock_binary), ("optimized_v133", opt_binary)):
        s = replace(
            base,
            model=stock_model if (label == "stock_v133" and stock_model) else base.model,
            port=free_port(),
            context=40960,
            parallel=1,
            drafts=False,
            prompt_cache_gib=0,
        )
        log = ROOT / "runs" / f"parity-{label}.log"
        env_override = None
        if label == "stock_v133":
            env_override = {
                k: v
                for k, v in s.server_env().items()
                if k
                not in (
                    "TF_DSV41_FP64_BF16",
                    "TF_DSV41_M5_FAST_DECODE",
                    "TF_DSV41_MARKOV_TOPK",
                    "TF_DSV41_ENGRAM_FIRST_RAM",
                )
            }
        with ManagedServer(binary, s, log, env_override=env_override):
            warmup(s.url, s.name)
            rows = {}
            for item in cases:
                rows[item["name"]] = call_prompt(
                    s.url, s.name, item, item["max_tokens"]
                )
        arms[label] = {
            "binary": display_path(binary),
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "cases": rows,
        }
        p32 = rows["python-32k"]
        print(
            f"[parity:{label}] python-32k ttft={p32['ttft_s']:.3f}s sha={p32['token_sha256'][:12]}",
            flush=True,
        )

    comparisons = {}
    all_exact = True
    for item in cases:
        name = item["name"]
        a = arms["stock_v133"]["cases"][name]
        b = arms["optimized_v133"]["cases"][name]
        exact = a["token_ids"] == b["token_ids"] and a["finish_reason"] == b["finish_reason"]
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"])) if x != y),
            None,
        )
        all_exact = all_exact and exact
        comparisons[name] = {
            "exact_match": exact,
            "first_diff_index": first_diff,
            "tokens_compared": len(a["token_ids"]),
            "stock_ttft_s": a["ttft_s"],
            "optimized_ttft_s": b["ttft_s"],
            "ttft_speedup": round(a["ttft_s"] / b["ttft_s"], 4) if b["ttft_s"] else None,
            "ttft_saved_s": round(a["ttft_s"] - b["ttft_s"], 4),
        }
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "all_exact": all_exact,
        "comparisons": comparisons,
        "arms": arms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[parity] all_exact={all_exact} -> {output}", flush=True)
    for name, row in comparisons.items():
        print(
            f"  [parity:{name}] exact={row['exact_match']} first_diff={row['first_diff_index']} "
            f"stock={row['stock_ttft_s']:.3f}s opt={row['optimized_ttft_s']:.3f}s ({row['ttft_speedup']}x)",
            flush=True,
        )


def run_concurrent_suite(
    binary: Path,
    base: config.Settings,
    output: Path,
    max_tokens: int,
    levels: list[int],
):
    bank = json.loads((ROOT / "assets/prompts/bench_prompts.json").read_text())
    results = {}
    for conc in levels:
        s = replace(
            base,
            port=free_port(),
            context=8192,
            parallel=conc,
            drafts=True,
            prompt_cache_gib=0,
        )
        log = ROOT / "runs" / f"bench-concurrent-p{conc}.log"
        items = [bank[i % len(bank)] for i in range(conc)]
        with ManagedServer(binary, s, log):
            warmup(s.url, s.name)
            gate = Barrier(conc, timeout=30)

            def run_one(idx_item):
                idx, item = idx_item
                gate.wait()
                t0 = time.perf_counter()
                row = call_prompt(s.url, s.name, item, max_tokens, seed=1234 + idx)
                t1 = time.perf_counter()
                return {**row, "slot_idx": idx, "start_t": t0, "end_t": t1}

            with ThreadPoolExecutor(max_workers=conc) as pool:
                rows = list(pool.map(run_one, enumerate(items)))

        wall_s = max(r["end_t"] for r in rows) - min(r["start_t"] for r in rows)
        first_tok_t = min(r["start_t"] + r["ttft_s"] for r in rows)
        decode_window_s = max(r["end_t"] for r in rows) - first_tok_t
        total_out = sum(r["completion_tokens"] for r in rows)
        after_first = sum(max(0, r["completion_tokens"] - 1) for r in rows)
        results[str(conc)] = {
            "parallel": conc,
            "wall_s": round(wall_s, 4),
            "wall_tps": round(total_out / wall_s, 2) if wall_s > 0 else 0.0,
            "window_tps": round(after_first / decode_window_s, 2)
            if decode_window_s > 0
            else 0.0,
            "median_client_decode_tps": round(
                statistics.median(r["decode_tps"] for r in rows if r["decode_tps"]), 2
            ),
            "median_client_ttft_s": round(
                statistics.median(r["ttft_s"] for r in rows), 4
            ),
            "max_client_ttft_s": round(max(r["ttft_s"] for r in rows), 4),
            "requests": [
                {k: v for k, v in r.items() if k not in ("start_t", "end_t", "token_ids")}
                for r in rows
            ],
        }
        r = results[str(conc)]
        print(
            f"[concurrent:p{conc}] window={r['window_tps']:.2f} tok/s wall={r['wall_tps']:.2f} tok/s "
            f"per_client={r['median_client_decode_tps']:.2f} tok/s ttft_med={r['median_client_ttft_s']:.3f}s",
            flush=True,
        )
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": display_path(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "max_tokens": max_tokens,
        "levels": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[concurrent] written -> {output}", flush=True)


def run_sampled_suite(binary: Path, base: config.Settings, output: Path, max_tokens: int):
    prompts = {
        row["name"]: row
        for row in json.loads((ROOT / "assets/prompts/bench_prompts.json").read_text())
    }
    cases = [
        ("filtered_code", prompts["quicksort-raw"], 0.8, {"top_k": 20, "top_p": 0.95}),
        ("filtered_chat", prompts["rust-reverse-words"], 0.8, {"top_k": 20, "top_p": 0.95}),
        ("nucleus_code", prompts["quicksort-raw"], 0.8, {"top_k": 0, "top_p": 0.95}),
        ("nucleus_chat", prompts["rust-reverse-words"], 0.8, {"top_k": 0, "top_p": 0.95}),
    ]
    runs = {}
    for label, drafts in (("drafted", True), ("serial", False)):
        s = replace(
            base,
            port=free_port(),
            context=8192,
            parallel=1,
            drafts=drafts,
            prompt_cache_gib=0,
        )
        log = ROOT / "runs" / f"bench-sampled-{label}.log"
        rows = {}
        with ManagedServer(binary, s, log):
            warmup(s.url, s.name)
            for cname, item, temp, extra in cases:
                rows[cname] = call_prompt(
                    s.url,
                    s.name,
                    item,
                    max_tokens,
                    seed=1234,
                    temperature=temp,
                    extra_sampling=extra,
                )
        runs[label] = rows
    comparisons = {}
    all_exact = True
    for cname, _, temp, extra in cases:
        d = runs["drafted"][cname]
        ser = runs["serial"][cname]
        exact = d["token_ids"] == ser["token_ids"] and d["finish_reason"] == ser["finish_reason"]
        all_exact = all_exact and exact
        comparisons[cname] = {
            "temperature": temp,
            "sampling": extra,
            "drafted_tps": d["decode_tps"],
            "serial_tps": ser["decode_tps"],
            "speedup": round(d["decode_tps"] / ser["decode_tps"], 3) if ser["decode_tps"] else None,
            "tokens": len(ser["token_ids"]),
            "exact_match": exact,
            "token_sha256": d["token_sha256"],
        }
        print(
            f"[sampled:{cname}] drafted={d['decode_tps']:.2f} vs serial={ser['decode_tps']:.2f} tok/s "
            f"exact={exact}",
            flush=True,
        )
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "binary": display_path(binary),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "max_tokens": max_tokens,
        "all_exact": all_exact,
        "comparisons": comparisons,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[sampled] all_exact={all_exact} -> {output}", flush=True)
    if not all_exact:
        raise RuntimeError(f"Sampled drafted vs serial mismatch: {comparisons}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("decode", "long", "parity", "concurrent", "sampled"), required=True
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--stock-binary", type=Path)
    parser.add_argument("--stock-model", type=Path)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--levels", default="1,2,4,8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = config.load()
    binary = args.binary or base.serve_binary
    if args.mode == "decode":
        run_decode_suite(binary, base, args.output, args.max_tokens)
    elif args.mode == "long":
        run_long_suite(binary, base, args.output, args.max_tokens)
    elif args.mode == "concurrent":
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
        run_concurrent_suite(binary, base, args.output, args.max_tokens, levels)
    elif args.mode == "sampled":
        run_sampled_suite(binary, base, args.output, args.max_tokens)
    elif args.mode == "parity":
        if not args.stock_binary:
            parser.error("--stock-binary is required for --mode parity")
        run_parity_vs_stock(
            binary, args.stock_binary, base, args.output, stock_model=args.stock_model
        )


if __name__ == "__main__":
    main()
