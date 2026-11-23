# Serving settings

This recipe serves DeepSeek's original Flash-1017 checkpoint on TensorFold `v1.3.3` with optimized native Metal 4 kernels. Fresh installs default to one client slot, adaptive DSpark drafting, a 4 GiB prompt cache, and room for 139,264 tokens per request.

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

**Pinned M5 Ultra Metal 4.1 environment.** The launcher automatically sets `TF_METAL4=1`, `TF_DSV41_FP64_BF16=1`, `TF_DSV41_M5_FAST_DECODE=1`, and `TF_DSV41_MARKOV_TOPK=64` so fast NAX/MXFP4 prompt prefill, stock FP32 hardware-MMA target decode, full-vocab greedy GPU Markov, and top-K filtered/pure-nucleus GPU drafting all run out of the box with no extra flags.

**Paths and listener.** Under `[model]`, set `model_dir`, `engram_dir`, and `assets_dir` to reuse an existing checkpoint and prepared SSD Engram tables. Under `[local]`, set `host`, `port`, and `state_dir` for the listener and managed process files. Relative paths resolve from the recipe directory. See [local.example.toml](local.example.toml).
