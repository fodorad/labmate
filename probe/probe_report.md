# Capability probe

Ollama `0.24.0` · 2026-09-26T10:31:42.571644+00:00

| Model | Check | Result | Detail | Metrics |
|---|---|---|---|---|
| `qwen3.6:35b-mlx` | declared capabilities | ℹ️ | completion, vision, thinking, tools |  |
| `qwen3.6:35b-mlx` | cold load + unload | ✅ | unloaded | cold_load_s=8.5, unload_s=0.6 |
| `qwen3.6:35b-mlx` | structured output | ℹ️ | (would fail) strict 0/10, lenient 0/10; first non-strict output: '**Main Result Claim:**\nLinMulT improves binary sentiment accuracy on the CMU-MOSEI benchmark while significantly reducin' | gen_tok_per_s=33.7 |
| `qwen3.6:35b-mlx` | tool calling | ✅ | use_paper_figure({'figure_id': 'fig3'}) |  |
| `qwen3.6:35b-mlx` | structured output (schema in prompt) | ✅ | strict 10/10, lenient 10/10 | gen_tok_per_s=33.8 |
| `qwen3.6:35b-mlx` | determinism | ✅ | 1 distinct output(s) over 3 runs |  |
| `qwen3.6:35b-mlx` | vision | ℹ️ | (would fail) answer: 'Black' |  |
| `gemma4:26b-mlx` | declared capabilities | ℹ️ | completion, tools, thinking |  |
| `gemma4:26b-mlx` | cold load + unload | ✅ | unloaded | cold_load_s=7.1, unload_s=0.6 |
| `gemma4:26b-mlx` | structured output | ℹ️ | (would fail) strict 0/10, lenient 0/10; first non-strict output: '**Main Result Claim:** LinMulT improves binary sentiment accuracy on the CMU-MOSEI benchmark from 82.1% to 84.6% while r' | gen_tok_per_s=32.6 |
| `gemma4:26b-mlx` | tool calling | ✅ | use_paper_figure({'figure_id': 'fig3'}) |  |
| `gemma4:26b-mlx` | structured output (schema in prompt) | ✅ | strict 10/10, lenient 10/10 | gen_tok_per_s=32.3 |
| `gemma4:26b-mlx` | determinism | ✅ | 1 distinct output(s) over 3 runs |  |
| `phi4-reasoning:plus` | declared capabilities | ℹ️ | completion |  |
| `phi4-reasoning:plus` | tool calling | ℹ️ | not declared |  |
| `phi4-reasoning:plus` | cold load + unload | ✅ | unloaded | cold_load_s=10.4, unload_s=0.5 |
| `phi4-reasoning:plus` | structured output | ℹ️ | (would pass) strict 10/10, lenient 10/10 | gen_tok_per_s=9.4 |
| `phi4-reasoning:plus` | structured output (schema in prompt) | ✅ | strict 10/10, lenient 10/10 | gen_tok_per_s=9.5 |
| `phi4-reasoning:plus` | determinism | ✅ | 1 distinct output(s) over 3 runs |  |
| `gemma4:e4b` | declared capabilities | ℹ️ | completion, vision, audio, tools, thinking |  |
| `gemma4:e4b` | cold load + unload | ✅ | unloaded | cold_load_s=6.6, unload_s=0.5 |
| `gemma4:e4b` | structured output | ℹ️ | (would pass) strict 10/10, lenient 10/10 | gen_tok_per_s=30.4 |
| `gemma4:e4b` | tool calling | ✅ | use_paper_figure({'figure_id': 'fig3'}) |  |
| `gemma4:e4b` | structured output (schema in prompt) | ✅ | strict 10/10, lenient 10/10 | gen_tok_per_s=30.1 |
| `gemma4:e4b` | determinism | ✅ | 1 distinct output(s) over 3 runs |  |
| `gemma4:e4b` | vision | ✅ | answer: 'Red' |  |
