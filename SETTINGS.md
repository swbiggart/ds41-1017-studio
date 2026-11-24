# Serving settings

This recipe serves DeepSeek's Flash-1017 checkpoint (original or losslessly repacked via `python -m mac repack`) on TensorFold `v1.3.3` with optimized native Metal 4.1 kernels. Fresh installs default to one client slot, adaptive DSpark drafting, a 4 GiB prompt cache, and room for 139,264 tokens per request.

| CLI option | Saved setting | Tradeoff |
|---|---|---|
| `--drafts` or `--no-drafts` | `[serve].drafts` | Leave drafting on for fastest interactive decode. Turn it off for one-token-at-a-time serial decoding with identical target arithmetic and sampler. |
| `--prompt-cache` or `--no-prompt-cache` | `[serve].prompt_cache` | Keep on (4 GiB unified-memory budget) to reuse prompt state across repeated prompts and multi-turn histories. Disable for cold benchmarks. |
| `--context 32768` | `[serve].context` | Limit combined prompt + reply tokens per slot (default `139264`, covering 131K prompts + 8K replies). Lower values reduce per-slot KV memory when running multiple slots. |
| `--parallel 2` | `[serve].parallel` | Allocate concurrent client slots (`1`–`16`, default `1`). Each slot reserves its own KV/state memory. |

Place serving flags before the subcommand. To switch settings on a running server, stop it and restart:

```sh
python -m mac down
python -m mac --no-drafts up
```

To persist settings across restarts, copy `local.example.toml` to `local.toml` and edit only the keys you want to override:

```toml
[serve]
parallel = 2
```

CLI flags override `local.toml` for that invocation. Run `python -m mac settings` to inspect the resolved arguments and environment.

**Lossless local repack.** Run `python -m mac repack --verify` after `python -m mac prepare` to stream the official checkpoint into `.local/checkpoint-repacked` (45 per-consumer 128B/16 KiB-aligned shards, `286.42 GiB` on disk, omitting the raw `.engram.embed.*` tensors already packed into `engram/`), then set `model_dir = ".local/checkpoint-repacked"` under `[model]` in `local.toml`. Both `python -m mac validate` and the native server verify `repack-manifest.json` against the pinned Flash-1017 index.

**Pinned M5 Ultra Metal 4.1 environment.** The launcher automatically sets `TF_METAL4=1`, `TF_DSV41_FP64_BF16=1`, `TF_DSV41_M5_FAST_DECODE=1`, `TF_DSV41_MARKOV_TOPK=64`, and `TF_DSV41_ENGRAM_FIRST_RAM=1`. Under `TF_DSV41_M5_FAST_DECODE=1`, the engine automatically enables the verified M5 Ultra kernel defaults (32-chunk layer-band prefill, 4-chunk cross-chunk routed experts, boot-audited staged Folded-E4M3 experts, single-pass NAX dense FP8 with compact FP8 `wo_a`, 16-head NAX prefill attention with fused grouped `wo_a`/`wo_b` and in-place query RoPE, tensor-unit indexer decode, reversible packed vocab head, fused `postnorm_expert` + direct-slot MXFP4 verify experts, and `plain_streams=5` concurrent switching).

**Paths and listener.** Under `[model]`, set `model_dir`, `engram_dir`, and `assets_dir` to reuse an existing checkpoint (official or repacked) and prepared SSD Engram tables. Under `[local]`, set `host`, `port`, and `state_dir` for the listener and managed process files. Relative paths resolve from the recipe directory. See [local.example.toml](local.example.toml).
