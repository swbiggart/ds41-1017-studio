# Credits

This recipe serves DeepSeek's original `DeepSeek-V4.1-Flash-1017` checkpoint on a 512 GB Apple Silicon Mac Studio using TensorFold `v1.3.3` with optimized native Metal 4 kernels.

## Core components

| Component | Source |
|---|---|
| DeepSeek-V4.1-Flash-1017 checkpoint | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/08c8657f02291e11b6f6b63e0e16014f8e748518) |
| Lossless TensorFold Metal 4.1 repack | [swbiggart/DeepSeek-V4.1-Flash-1017-TensorFold-Metal-Lossless](https://huggingface.co/swbiggart/DeepSeek-V4.1-Flash-1017-TensorFold-Metal-Lossless/tree/e9b17382756ff4d6ad988217539fc3799ef9c305) |
| TensorFold `v1.3.3` native Metal runtime & asset tools | [ashhart/TensorFold (`v1.3.3`, `1e4a32c`)](https://github.com/ashhart/TensorFold/tree/1e4a32cb5e80f2908e4642f9e7ffe71ae63792cc) |
| Mac CLI & recipe layout inspiration | [BobClawblaw/deepseek-v41-flash-mlx-m5-ultra](https://github.com/BobClawblaw/deepseek-v41-flash-mlx-m5-ultra/tree/5138e70f0c6caee16f963cf3ecf9862c13d40e5f) |
| 512 GB M5 Ultra baseline configuration | [LazyLocalLLM (TensorFold #3547)](https://github.com/ashhart/TensorFold/issues/3547) |
| Long-context Python benchmark corpus | [oMLX `code_python.txt`](https://github.com/jundot/omlx/blob/454c90d35fe41d6a4054074ad7f99323857b97e7/omlx/admin/bench_corpora/ATTRIBUTION.md) |
| Zig toolchain (`0.17.0`) | [ziglang.org](https://ziglang.org/) |

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license attributions.
