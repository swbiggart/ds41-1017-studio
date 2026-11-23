# Performance and parity receipts (M5 Ultra 512 GB)

All measurements below were collected on an 80-core Apple M5 Ultra Mac Studio with 512 GB unified memory, Metal 4.1 (`TF_METAL4=1`), SSD-backed Engram tables (`65,536` cached rows per table), and TensorFold `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`) patched with [`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch) and [`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch).

## What the Metal 4.1 patch stack optimizes

The two patches listed in [`patches/manifest.json`](patches/manifest.json) leave all default `v1.3.3` `.metal` kernels untouched and activate M5 Ultra fast serving via `TF_DSV41_FP64_BF16=1`, `TF_DSV41_M5_FAST_DECODE=1`, and `TF_DSV41_MARKOV_TOPK=64`:

1. **[`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch)** — backport of the full 7-PR M5 Ultra Metal 4.1 + DSpark stack onto `v1.3.3`:
   - **Dense FP8 NAX prefill (`fp8_mma_cert.metal`):** batched `wo_a` dispatch, prepared activation reuse (`prep_x`), and branchless Knuth `TwoSum` compensated accumulation for FP64-certified BF16 outputs.
   - **Routed MXFP4 tensor-unit prefill (`experts_units_cert.metal`):** 32-row native M5 tensor-unit tiling (`XU_RT=32`) with a 256-bit unsafe-group mask and compensated `TwoSum` accumulation.
   - **DSpark drafting stack (`dspark.zig`, `dspark_undo.zig`, `dspark_attn_cert.metal`, `markov_cert_full.metal`, `markov_cert_topk.metal`, `early_cert.metal`):** concurrent batched ingest (`IngestBuf`), GPU blit undo, fused `gate_plan_dspark`, bidirectional DSpark block attention, shared-`w2` full-vocab and top-K single-TG GPU Markov chains, unified GPU filtered/multi-slot behind-the-verify picks, and automatic pure-nucleus speculation.
2. **[`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch)** — adds `TF_DSV41_M5_FAST_DECODE=1` hybrid serving mode:
   - Keeps the M5 Ultra NAX FP8 + tensor-unit MXFP4 prompt prefill and PR 4–7 DSpark stack while keeping `v1.3.3`'s stock FP32 hardware-MMA target decode/verify (`fp8_rows`, `fp8_tile`, `experts`, `engram`) for maximum end-to-end throughput.
   - Automatically routes greedy/raced asks to stock FP32 DSpark projections + full-vocab GPU Markov (`markov_cert_full.metal`), while routing filtered (`top_k`/`top_p`/`min_p`) and pure-nucleus asks to top-K GPU Markov (`markov_cert_topk.metal`) + GPU behind-the-verify filtered picks (`early_cert.metal`).

## Single-session greedy decode & drafted-vs-serial parity

Measured across 7 greedy workloads (`max_tokens=128`, `temperature=0`, `seed=1234`, prompt cache disabled, one warm-up request). Pooled tok/s divides committed tokens after the first by total decode time. Receipt: [`results/decode-draft-vs-serial.json`](results/decode-draft-vs-serial.json).

| Workload | Drafted tok/s | Serial tok/s | Tokens | Raw IDs matched |
|---|---:|---:|---:|---:|
| `quicksort-raw` | 124.49 | 52.90 | 76 | 76/76 |
| `rust-reverse-words` | 88.14 | 52.49 | 128 | 128/128 |
| `multiply-17-23` | 125.43 | 52.38 | 128 | 128/128 |
| `js-debounce-raw` | 112.28 | 52.83 | 88 | 88/88 |
| `linear-equation` | 90.89 | 52.24 | 128 | 128/128 |
| `sql-left-join` | 95.16 | 52.52 | 128 | 128/128 |
| `fishing-village` | 63.85 | 52.69 | 128 | 128/128 |
| **Pooled total** | **93.20** | **52.54** | **804** | **804/804 (7/7 finishes)** |

## Sampled & pure-nucleus decode (drafted vs serial)

Measured with `temperature=0.8`, `seed=1234`, `max_tokens=128`, prompt cache disabled. Both filtered (`top_k=20, top_p=0.95`) and pure-nucleus (`top_k=0, top_p=0.95`) requests verify 100% exact raw token IDs and finish reasons against serial decoding. Receipt: [`results/sampled-draft-vs-serial.json`](results/sampled-draft-vs-serial.json).

| Case | Sampling | Drafted tok/s | Serial tok/s | Speedup | Parity |
|---|---|---:|---:|---:|---|
| `filtered_code` (`quicksort-raw`) | `T=0.8, k=20, p=0.95` | **107.91** | 52.65 | **2.05×** | 74/74 exact |
| `filtered_chat` (`rust-reverse-words`) | `T=0.8, k=20, p=0.95` | **92.82** | 52.18 | **1.78×** | 128/128 exact |
| `nucleus_code` (`quicksort-raw`) | `T=0.8, k=0, p=0.95` | **92.85** | 34.63 | **2.68×** | 74/74 exact |
| `nucleus_chat` (`rust-reverse-words`) | `T=0.8, k=0, p=0.95` | **77.68** | 34.44 | **2.26×** | 128/128 exact |

## Long-context prefill & prompt-cache reuse

Measured with default settings (`context=139264`, `parallel=1`, `drafts=true`, `prompt_cache_gib=4`, `max_tokens=128`, `temperature=0`, `seed=1234`). Each prompt was run cold (`cached=0`) and then immediately retried to verify full-prompt cache reuse. Receipt: [`results/long-context-and-cache.json`](results/long-context-and-cache.json).

| Prompt | Active tokens | Cold TTFT | Cold decode | Warm cache-hit TTFT | Warm decode | Cache-hit parity |
|---|---:|---:|---:|---:|---:|---|
| `code_python-32768-1.ids` | 32,768 | **19.842 s** | **96.71 tok/s** | **0.0086 s** | **98.81 tok/s** | 128/128 exact IDs & finish |
| `code_python-131072-1.ids` | 131,072 | **79.284 s** | **80.65 tok/s** | **0.0145 s** | **82.35 tok/s** | 128/128 exact IDs & finish |

## Concurrent drafted serving

Measured across `p=1, 2, 4, 8` simultaneous clients (`max_tokens=128`, `temperature=0`, `seed=1234`, prompt cache disabled, mixed 76–128 token workloads). Receipt: [`results/concurrent-drafted.json`](results/concurrent-drafted.json).

| Parallel clients | Active-window tok/s | Wall tok/s | Median per-client tok/s | Median TTFT |
|---:|---:|---:|---:|---:|
| `1` | **123.51** | 103.60 | 123.66 | 0.126 s |
| `2` | **101.19** | 96.11 | 66.55 | 0.197 s |
| `4` | **96.74** | 94.99 | 28.07 | 0.314 s |
| `8` | **114.86** | 114.03 | 16.75 | 0.590 s |

## Comparison against stock TensorFold `v1.3.3`

Matched serial runs (`--no-drafts --no-prompt-cache --context 40960`, `temperature=0`, `seed=1234`) comparing unpatched stock `v1.3.3` (`1e4a32c`) against this recipe's patched build. Receipt: [`results/parity-vs-stock-v133.json`](results/parity-vs-stock-v133.json).

| Prompt | Stock `v1.3.3` TTFT | Patched `v1.3.3` TTFT | Speedup | Token agreement vs stock `v1.3.3` |
|---|---:|---:|---:|---|
| `code_python-32768-1.ids` (32K) | 21.683 s | **19.827 s** | **1.094× (-1.86 s)** | Differs at token 2 (`16` vs `1749`) due to FP64-certified BF16 prefill rounding vs stock FP32 prefill rounding |
| `quicksort-raw` (short) | 0.123 s | **0.121 s** | — | **64/64 exact** |
| `multiply-17-23` (short) | 0.123 s | **0.120 s** | — | **64/64 exact** |

## Reproduce locally

With a running server (`python -m mac up`), run TensorFold's built-in streaming benchmark:

```sh
python -m mac bench --tokens 128 --reps 3 --temperatures 0
```

To regenerate the full benchmark receipts (stop any managed server first with `python -m mac down`):

```sh
python scripts/bench_suite.py --mode decode --max-tokens 128 --output runs/decode-draft-vs-serial.json
python scripts/bench_suite.py --mode sampled --max-tokens 128 --output runs/sampled-draft-vs-serial.json
python scripts/bench_suite.py --mode long --max-tokens 128 --output runs/long-context-and-cache.json
python scripts/bench_suite.py --mode concurrent --levels 1,2,4,8 --max-tokens 128 --output runs/concurrent-drafted.json
```
