# Model micro benchmark

arm64, 34 GB memory, Darwin 25.6.0

### Models

| Model | Size | Cold load | Reads | Writes | Structured | Tool calls | Judge: mistakes caught | Judge: false alarms | Judge: s per verdict |
|---|---|---|---|---|---|---|---|---|---|
| `gemma4:e2b` | 4.6 GB | 5 s | 655 tok/s | 35 tok/s | 5/5 | - | 15/15 | 1/9 | 1.3 s |
| `ornith-1.5:9b` | 6.6 GB | 2 s | 199 tok/s | 18 tok/s | 5/5 | 3/3 | 15/15 | 1/9 | 3.0 s |
| `gemma4:e4b` | 6.6 GB | 7 s | 333 tok/s | 24 tok/s | 5/5 | 3/3 | 15/15 | 0/9 | 2.0 s |
| `gemma4:26b-mlx` | 16.7 GB | 9 s | 414 tok/s | 30 tok/s | 5/5 | 3/3 | 15/15 | 0/9 | 1.5 s |
| `qwen3.8:27b-mlx` | 18.2 GB | 9 s | 61 tok/s | 10 tok/s | 5/5 | 3/3 | - | - | - |

### Profiles

| Profile | Writer | Judge | Together | Fits in 34 GB (keeping 5 GB free) |
|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 16.7 GB | yes |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 23.3 GB | yes |
| fast | `gemma4:26b-mlx` | `gemma4:e2b` | 21.3 GB | yes |
| small | `gemma4:e4b` | `gemma4:e4b` | 6.6 GB | yes |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 13.1 GB | yes |
| ornith-solo | `ornith-1.5:9b` | `ornith-1.5:9b` | 6.6 GB | yes |
| qwen | `qwen3.8:27b-mlx` | `gemma4:e4b` | 24.8 GB | yes |
