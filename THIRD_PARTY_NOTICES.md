# Third-party notices

This recipe applies standalone Metal kernel and runtime patches to pinned TensorFold `v1.3.3` sources and downloads model weights and build tools separately under their own licenses.

## TensorFold

Copyright 2026 TensorFold contributors. Distributed under the Apache License 2.0 (with pre-0.6.0 code retaining its MIT notice). Patches in `patches/` are distributed as diffs against TensorFold `v1.3.3` (`1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc`). The cloned build checkout retains TensorFold's full `LICENSE`, `NOTICE`, `LICENSES/`, and `THIRD_PARTY_NOTICES.md`.

- [Apache License 2.0](licenses/tensorfold/LICENSE)
- [Original NOTICE](licenses/tensorfold/NOTICE)
- [Original MIT notice](licenses/tensorfold/LICENSES/MIT.txt)

## DeepSeek-V4.1-Flash-1017

Copyright (c) DeepSeek. The recipe fetches DeepSeek's original checkpoint at revision `08c8657f02291e11b6f6b63e0e16014f8e748518` under DeepSeek's model license.

- [DeepSeek license](licenses/deepseek/LICENSE)

## BobClawblaw's Mac recipe

Copyright (c) 2026 BobClawblaw. The `recipe.toml` / `local.toml` and `python -m mac` command structure follow BobClawblaw's Mac recipe.

- [Original MIT license](licenses/bobclawblaw/LICENSE)

## Benchmark inputs

Long-context Python benchmark inputs reuse tokenized excerpts from oMLX's Apache-2.0 `code_python.txt` corpus, and short-prompt decode suites use TensorFold's `tools/bench_prompts.json`.

- [oMLX license](licenses/omlx/LICENSE)
- [oMLX corpus attribution](licenses/omlx/ATTRIBUTION.md)
