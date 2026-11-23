# DeepSeek-V4.1-Flash-1017 on a 512 GB Mac Studio (TensorFold `v1.3.3`)

by Scott Biggart · [@woolsey](https://x.com/woolsey)

[![Follow on X](https://img.shields.io/badge/Follow_on_X-@woolsey-000000?logo=x&logoColor=white)](https://x.com/woolsey)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Serve DeepSeek's original `DeepSeek-V4.1-Flash-1017` checkpoint on an 80-core M5 Ultra Mac Studio (512 GB unified memory) with TensorFold `v1.3.3` (`1e4a32c`) and standalone Metal 4.1 kernel optimizations.

## Headline performance & parity (M5 Ultra 512 GB)

| Workload | Measured | Notes |
|---|---:|---|
| Greedy decode (7 workloads, adaptive DSpark) | **93.20 tok/s** | Peaks at **125.4 tok/s** (`multiply-17-23`); **804/804** exact raw token IDs & **7/7** finish reasons vs serial (**1.77×**) |
| Greedy decode (7 workloads, `--no-drafts`) | **52.54 tok/s** | Serial control with identical target arithmetic |
| Filtered sampled (`T=0.8, k=20, p=0.95`) | **107.9 / 92.8 tok/s** | Code / chat; **2.05× / 1.78×** vs serial, **100% exact** token IDs |
| Pure nucleus (`T=0.8, k=0, p=0.95`) | **92.9 / 77.7 tok/s** | Code / chat; **2.68× / 2.26×** vs serial (`34.6 / 34.4 tok/s`), **100% exact** token IDs |
| 32,768-token Python prompt (cold TTFT / decode) | **19.84 s** / **96.7 tok/s** | **1.86 s faster** (**1.094×**) than stock `v1.3.3` (`19.83 s` vs `21.68 s` serial matched pair) |
| 32,768-token prompt-cache hit | **0.009 s** / **98.8 tok/s** | **100% exact** token IDs & finish reason vs cold run (`32,768` cached tokens) |
| 131,072-token Python prompt (cold TTFT / decode) | **79.28 s** / **80.7 tok/s** | Default 139,264-token context |
| 131,072-token prompt-cache hit | **0.015 s** / **82.4 tok/s** | **100% exact** token IDs & finish reason vs cold run (`131,072` cached tokens) |
| Concurrent drafted serving (`p=1 / 2 / 4 / 8`) | **123.5 / 101.2 / 96.7 / 114.9 tok/s** | Active-window throughput across 1–8 simultaneous clients |

See [PROFILES.md](PROFILES.md) for per-request receipts and parity details, and [SETTINGS.md](SETTINGS.md) for serving flags.

- **Pinned checkpoint:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) (`08c8657f02291e11b6f6b63e0e16014f8e748518`)
- **Pinned engine:** TensorFold `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`) + [`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch) + [`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch)

## Quick start

Requires an Apple Silicon Mac with 512 GB unified memory, macOS 27+ (Metal 4), Python 3.12, Git, Xcode command-line tools, and ~715 GB SSD space for the original checkpoint shards and packed Engram tables.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'huggingface_hub==1.31.0'
python -m mac build
python -m mac fetch
python scripts/verify_checkpoint.py .local/checkpoint --sha256
python -m mac prepare
python -m mac validate
python -m mac up
python -m mac chat 'Write a Python generator that merges two sorted iterators.' --max-tokens 128
```

If you already have the pinned checkpoint and prepared Engram assets on disk, copy [`local.example.toml`](local.example.toml) to `local.toml`, point `[model]` paths at them, and skip `fetch` and `prepare`.

The OpenAI-compatible server listens at `http://127.0.0.1:8890/v1` under model ID `deepseek-v41-flash-1017-official`.

```sh
python -m mac status    # Readiness and managed PID
python -m mac smoke     # Greedy JSON chat completion check
python -m mac logs      # Tail managed server log
python -m mac down      # Stop managed server
```

## Configuration tradeoffs

`python -m mac up` defaults to the fastest verified configuration: **1 slot**, **139,264 context**, **adaptive DSpark drafting on**, and **4 GiB prompt cache on**. Only four serving flags are exposed:

- `--drafts` / `--no-drafts`: adaptive DSpark speculative decoding vs one-token-at-a-time serial decoding (identical raw token IDs and finish reasons).
- `--prompt-cache` / `--no-prompt-cache`: 4 GiB unified-memory prompt-state cache vs cold prefill.
- `--context N`: maximum prompt + reply tokens per slot (default `139264`).
- `--parallel N`: concurrent client slots (`1`–`16`, default `1`).

See [SETTINGS.md](SETTINGS.md), [CREDITS.md](CREDITS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [LICENSE](LICENSE) (Apache-2.0).
