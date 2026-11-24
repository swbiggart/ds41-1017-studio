# DeepSeek-V4.1-Flash-1017 on a 512 GB Mac Studio (TensorFold `v1.3.3`)

by Scott Biggart · [@woolsey](https://x.com/woolsey)

[![Follow on X](https://img.shields.io/badge/Follow_on_X-@woolsey-000000?logo=x&logoColor=white)](https://x.com/woolsey)
[![License Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

Serve DeepSeek's `DeepSeek-V4.1-Flash-1017` checkpoint (official or locally losslessly repacked for zero-copy Metal 4.1 residency) on an 80-core M5 Ultra Mac Studio (512 GB unified memory) with TensorFold `v1.3.3` (`1e4a32c`) and standalone Metal 4.1 kernel optimizations.

## Headline performance & parity (M5 Ultra 512 GB)

| Workload | Measured | Notes |
|---|---:|---|
| Greedy decode (7 workloads, adaptive DSpark) | **97.06 tok/s** | Peaks at **133.1 tok/s** (`multiply-17-23`); **804/804** exact raw token IDs & **7/7** finish reasons vs serial (**1.84×**) |
| Greedy decode (7 workloads, `--no-drafts`) | **52.84 tok/s** | Serial control with identical target arithmetic |
| Filtered sampled (`T=0.8, k=20, p=0.95`) | **116.3 / 92.8 tok/s** | Code / chat; **2.19× / 1.76×** vs serial, **100% exact** token IDs |
| Pure nucleus (`T=0.8, k=0, p=0.95`) | **97.0 / 78.2 tok/s** | Code / chat; **2.81× / 2.28×** vs serial (`34.5 / 34.3 tok/s`), **100% exact** token IDs |
| 32,768-token Python prompt (cold TTFT / decode) | **8.78 s** (`3,733 tok/s`) / **119.2 tok/s** | **2.47× faster prefill** (`-12.91 s`) than stock `v1.3.3` (`8.75 s` vs `21.67 s` serial matched pair) |
| 32,768-token prompt-cache hit | **0.0087 s** / **123.2 tok/s** | **100% exact** token IDs & finish reason vs cold run (`32,768` cached tokens) |
| 131,072-token Python prompt (cold TTFT / decode) | **33.44 s** (`3,920 tok/s`) / **97.0 tok/s** | Default 139,264-token context; **0 MB swap** |
| 131,072-token prompt-cache hit | **0.0143 s** / **98.6 tok/s** | **100% exact** token IDs & finish reason vs cold run (`131,072` cached tokens) |
| Concurrent drafted serving (`p=1 / 2 / 4 / 8`) | **120.9 / 106.0 / 121.9 / 127.2 tok/s** | Active-window throughput across 1–8 simultaneous clients |

See [PROFILES.md](PROFILES.md) for per-request receipts and parity details, and [SETTINGS.md](SETTINGS.md) for serving flags.

- **Pinned checkpoint:** [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) (`08c8657f02291e11b6f6b63e0e16014f8e748518`)
- **Pinned engine:** TensorFold `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`) + [`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch) + [`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch) + [`patches/03-m5-nax-prefill-repack-fast.patch`](patches/03-m5-nax-prefill-repack-fast.patch) (0 modified upstream `.metal` files)

## Quick start

Requires an Apple Silicon Mac with 512 GB unified memory, macOS 27+ (Metal 4.1), Python 3.12, Git, Xcode command-line tools, and ~715 GB SSD space during initial fetch/prepare (dropping to ~480 GB total once repacked and the duplicate raw Engram shards are removed).

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'huggingface_hub==1.31.0'
python -m mac build
python -m mac fetch
python scripts/verify_checkpoint.py .local/checkpoint --sha256
python -m mac prepare
python -m mac repack --verify
```

Point `local.toml` at `.local/checkpoint-repacked` (see [`local.example.toml`](local.example.toml)) — or serve the original checkpoint directly if you prefer not to repack — then validate and start:

```sh
python -m mac validate
python -m mac up
python -m mac chat 'Write a Python generator that merges two sorted iterators.' --max-tokens 128
```

**Why repack?** `python -m mac repack --verify` streams the official checkpoint in ~22 s (`12.8 GiB/s`) into 45 lossless per-consumer shards (`model-top`, `model-layer-00..39`, `model-mtp-0..2`, `model-vision`) with 16 KiB header alignment and 128-byte tensor alignment, omitting the 4 raw `.engram.embed.*` tensors already packed into `engram/` (`286.42 GiB` on disk, saving `~203 GB`). This eliminates every unaligned shard copy on Metal (dropping copied weight arenas from `7,789 MiB` to `1,567 MiB`, of which `1,263 MiB` is the reversible packed vocabulary head) and enables zero-copy `mmap` residency for both the 40-layer backbone and DSpark `mtp.0..2`.

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
