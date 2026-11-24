# Performance and parity receipts (M5 Ultra 512 GB)

All measurements below were collected on an 80-core Apple M5 Ultra Mac Studio with 512 GB unified memory, Metal 4.1 (`TF_METAL4=1`), locally repacked 128B/16 KiB-aligned per-consumer shards (`286.42 GiB`, zero-copy `292,366 MiB` mapped), Layer-1 Engram in pageable host RAM (`TF_DSV41_ENGRAM_FIRST_RAM=1`) + Layer-14 SSD Engram (`65,536` cached rows), and TensorFold `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`) patched with [`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch), [`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch), and [`patches/03-m5-nax-prefill-repack-fast.patch`](patches/03-m5-nax-prefill-repack-fast.patch).

## What the Metal 4.1 patch stack optimizes

All three patches listed in [`patches/manifest.json`](patches/manifest.json) leave every default `v1.3.3` `.metal` kernel untouched (0 modified upstream `.metal` files) and activate M5 Ultra fast serving via `TF_DSV41_FP64_BF16=1`, `TF_DSV41_M5_FAST_DECODE=1`, `TF_DSV41_MARKOV_TOPK=64`, and `TF_DSV41_ENGRAM_FIRST_RAM=1`:

1. **[`patches/01-m5-metal-dspark-stack.patch`](patches/01-m5-metal-dspark-stack.patch)** — backport of the 7-PR M5 Ultra Metal 4.1 + DSpark stack onto `v1.3.3`:
   - FP64-certified NAX dense FP8 and 32-row tensor-unit MXFP4 prefill kernels (`fp8_mma_cert.metal`, `experts_units_cert.metal`).
   - Full DSpark drafting stack (`dspark.zig`, `dspark_undo.zig`, `dspark_attn_cert.metal`, `markov_cert_full.metal`, `markov_cert_topk.metal`, `early_cert.metal`): concurrent batched ingest (`IngestBuf`), GPU blit undo, fused `gate_plan_dspark`, bidirectional DSpark block attention, shared-`w2` full-vocab and top-K single-TG GPU Markov chains, unified GPU filtered/multi-slot behind-the-verify picks, and automatic pure-nucleus speculation.
2. **[`patches/02-m5-fast-decode-mode.patch`](patches/02-m5-fast-decode-mode.patch)** — adds `TF_DSV41_M5_FAST_DECODE=1` hybrid serving mode combining M5 Ultra NAX/tensor-unit prompt prefill and PR 4–7 DSpark drafting with stock `v1.3.3` FP32 hardware-MMA target decode/verify (`fp8_rows`, `fp8_tile`, `experts`, `engram`).
3. **[`patches/03-m5-nax-prefill-repack-fast.patch`](patches/03-m5-nax-prefill-repack-fast.patch)** — M5 Ultra Metal 4.1 kernel & repack research stack:
   - **Zero-copy repacked checkpoint support:** verifies `repack-manifest.json` in `engram_manifest.zig` (allowing omitted raw `.engram.embed.*` shards once packed into `DSV41EN1`) and zero-copy maps both the 40-layer backbone and DSpark `mtp.0..2` shards into the Metal residency set.
   - **Layer-1 Engram in pageable host RAM + SIMD decode + cross-band dedup (`engram_rows_darwin.zig`, `engram_decode_vector.zig`, `engram_selected.zig`):** loads the 94.4 GiB Layer-1 Engram table into pageable host RAM once at boot and overlaps Layer-14 SSD reads behind encoder bands with zero GPU idle stall.
   - **Layer-band encoder schedule + staged Folded-E4M3 experts (`prompt.zig`, `prompt_band*.zig`, `folded_experts.metal`):** 32-chunk layer-band schedule with 4-chunk (`16,384`-row) cross-chunk routed expert batching and boot-audited staged Folded-E4M3 expert views (`181 MiB` private staging buffer, zero extra disk storage).
   - **Single-pass NAX prefill (`fp8_nax_fast.metal`, `attention_nax.metal`, `glue_fast.metal`):** single-pass 64×64 NAX dense FP8 keeping `wo_a` compact FP8 (saving `2.69 GiB` wired RAM vs BF16 expansion), 16-head NAX prefill attention with fused grouped `wo_a` act_quant+transpose, fused grouped `wo_b` act_quant+FP8 pack, in-place query RoPE tail rotation, and BF16 NAX router/compressor projections.
   - **Decode/verify & concurrent optimizations (`experts_direct.metal`, `fold_expert.metal`, `vocab_head.metal`, `index_units.metal`, `serve_lanes.zig`):** fused `postnorm_expert` + single-dispatch `ds_fp4_word` direct-slot MXFP4 routed experts, reversible coalesced BF16 vocabulary head, tensor-unit FP4 indexer decode (`tf_ix_decode_units`), and `plain_streams_default=5` for optimal concurrent `p=1..8` throughput.

## Single-session greedy decode & drafted-vs-serial parity

Measured across 7 greedy workloads (`max_tokens=128`, `temperature=0`, `seed=1234`, prompt cache disabled, one warm-up request). Pooled tok/s divides committed tokens after the first by total decode time. Receipt: [`results/decode-draft-vs-serial.json`](results/decode-draft-vs-serial.json).

| Workload | Drafted tok/s | Serial tok/s | Tokens | Raw IDs matched |
|---|---:|---:|---:|---:|
| `quicksort-raw` | 123.71 | 53.01 | 76 | 76/76 |
| `rust-reverse-words` | 90.23 | 52.72 | 128 | 128/128 |
| `multiply-17-23` | 133.05 | 52.87 | 128 | 128/128 |
| `js-debounce-raw` | 118.99 | 53.22 | 88 | 88/88 |
| `linear-equation` | 95.74 | 52.62 | 128 | 128/128 |
| `sql-left-join` | 97.59 | 52.76 | 128 | 128/128 |
| `fishing-village` | 67.40 | 52.89 | 128 | 128/128 |
| **Pooled total** | **97.06** | **52.84** | **804** | **804/804 (7/7 finishes)** |

## Sampled & pure-nucleus decode (drafted vs serial)

Measured with `temperature=0.8`, `seed=1234`, `max_tokens=128`, prompt cache disabled. Both filtered (`top_k=20, top_p=0.95`) and pure-nucleus (`top_k=0, top_p=0.95`) requests verify 100% exact raw token IDs and finish reasons against serial decoding. Receipt: [`results/sampled-draft-vs-serial.json`](results/sampled-draft-vs-serial.json).

| Case | Sampling | Drafted tok/s | Serial tok/s | Speedup | Parity |
|---|---|---:|---:|---:|---|
| `filtered_code` (`quicksort-raw`) | `T=0.8, k=20, p=0.95` | **116.31** | 53.17 | **2.19×** | 74/74 exact |
| `filtered_chat` (`rust-reverse-words`) | `T=0.8, k=20, p=0.95` | **92.80** | 52.59 | **1.76×** | 128/128 exact |
| `nucleus_code` (`quicksort-raw`) | `T=0.8, k=0, p=0.95` | **97.04** | 34.53 | **2.81×** | 74/74 exact |
| `nucleus_chat` (`rust-reverse-words`) | `T=0.8, k=0, p=0.95` | **78.23** | 34.26 | **2.28×** | 128/128 exact |

## Long-context prefill & prompt-cache reuse

Measured with default settings (`context=139264`, `parallel=1`, `drafts=true`, `prompt_cache_gib=4`, `max_tokens=128`, `temperature=0`, `seed=1234`, `0 MB` swap growth). Each prompt was run cold (`cached=0`) and then immediately retried to verify full-prompt cache reuse. Receipt: [`results/long-context-and-cache.json`](results/long-context-and-cache.json).

| Prompt | Active tokens | Cold TTFT | Prefill rate | Cold decode | Warm cache-hit TTFT | Warm decode | Cache-hit parity |
|---|---:|---:|---:|---:|---:|---:|---|
| `code_python-32768-1.ids` | 32,768 | **8.777 s** | **3,733 tok/s** | **119.17 tok/s** | **0.0087 s** | **123.23 tok/s** | 128/128 exact IDs & finish |
| `code_python-131072-1.ids` | 131,072 | **33.441 s** | **3,920 tok/s** | **96.96 tok/s** | **0.0143 s** | **98.55 tok/s** | 128/128 exact IDs & finish |

## Concurrent drafted serving

Measured across `p=1, 2, 4, 8` simultaneous clients (`max_tokens=128`, `temperature=0`, `seed=1234`, prompt cache disabled, mixed 76–128 token workloads). Receipt: [`results/concurrent-drafted.json`](results/concurrent-drafted.json).

| Parallel clients | Active-window tok/s | Wall tok/s | Median per-client tok/s | Median TTFT |
|---:|---:|---:|---:|---:|
| `1` | **120.89** | 102.10 | 121.11 | 0.124 s |
| `2` | **105.97** | 100.60 | 71.95 | 0.190 s |
| `4` | **121.86** | 118.97 | 38.78 | 0.282 s |
| `8` | **127.23** | 126.56 | 18.29 | 0.559 s |

## Comparison against stock TensorFold `v1.3.3`

Matched serial runs (`--no-drafts --no-prompt-cache --context 40960`, `temperature=0`, `seed=1234`) comparing unpatched stock `v1.3.3` (`1e4a32c`) on the official checkpoint against this recipe's patched build on the repacked checkpoint. Receipt: [`results/parity-vs-stock-v133.json`](results/parity-vs-stock-v133.json).

| Prompt | Stock `v1.3.3` TTFT | Patched `v1.3.3` TTFT | Speedup | Token agreement vs stock `v1.3.3` |
|---|---:|---:|---:|---|
| `code_python-32768-1.ids` (32K) | 21.668 s | **8.755 s** | **2.475× (-12.91 s)** | Differs at token 2 due to single-pass NAX prefill rounding vs stock FP32 prefill rounding |
| `quicksort-raw` (short) | 0.123 s | **0.119 s** | **1.034×** | 62/64 matched (2-token tie-break at token 35 from NAX prefill rounding) |
| `multiply-17-23` (short) | 0.123 s | **0.101 s** | **1.212×** | **64/64 exact** |

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
