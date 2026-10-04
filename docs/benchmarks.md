# Models: speed, memory and quality

Measured on arm64, 34 GB memory, Darwin 25.6.0, Ollama 0.35.1, 2026-10-05. Every run starts cold (models unloaded) with the reply cache off, so it pays what a first request pays. The GPU may use about three quarters of the memory, 'fits' keeps 5 GB of that free for other programs, and a pair that does not fit is evicted and loaded again during a run. Quality is the mean of deterministic checks the code already has (no model grades another); the parts are listed. Peak memory is read from `ollama ps`, which under-reports models run by llama-server (`gemma4:e4b`, `gemma4:e2b`). Run-to-run variation is about 30% (the same ask run took 25 and 32 s). `make bench-micro`, `make bench` and `make bench-report` reproduce this page.

## Single models

### Models

| Model | Size | Cold load | Reads | Writes | Structured | Tool calls | Judge: mistakes caught | Judge: false alarms | Judge: s per verdict |
|---|---|---|---|---|---|---|---|---|---|
| `gemma4:e2b` | 4.6 GB | 5 s | 655 tok/s | 35 tok/s | 5/5 | - | 15/15 | 1/9 | 1.3 s |
| `ornith-1.5:9b` | 6.6 GB | 2 s | 199 tok/s | 18 tok/s | 5/5 | 3/3 | 15/15 | 1/9 | 3.0 s |
| `gemma4:e4b` | 6.6 GB | 7 s | 333 tok/s | 24 tok/s | 5/5 | 3/3 | 15/15 | 0/9 | 2.0 s |
| `gemma4:26b-mlx` | 16.7 GB | 9 s | 414 tok/s | 30 tok/s | 5/5 | 3/3 | 15/15 | 0/9 | 1.5 s |
| `qwen3.8:27b-mlx` | 18.2 GB | 9 s | 61 tok/s | 10 tok/s | 5/5 | 3/3 | - | - | - |

### Profiles

| Profile | Writer | Judge | Together | Fits in 26 GB GPU memory (keeping 5 GB free) |
|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 16.7 GB | yes |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 23.3 GB | **no, a model is evicted** |
| fast | `gemma4:26b-mlx` | `gemma4:e2b` | 21.3 GB | **no, a model is evicted** |
| small | `gemma4:e4b` | `gemma4:e4b` | 6.6 GB | yes |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 13.1 GB | yes |
| ornith-solo | `ornith-1.5:9b` | `ornith-1.5:9b` | 6.6 GB | yes |
| qwen | `qwen3.8:27b-mlx` | `gemma4:e4b` | 24.8 GB | **no, a model is evicted** |

## Use cases

### ask: agent

Target: under 30 s per question. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | `gemma4:e4b` | `gemma4:e4b` | 22 s | yes | 7 s | 0 | 1.0 GB | 9 | 14478/985 | **68%** | right answer or refusal 0.75, expected source cited 0.67, sentences kept 0.64 |
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 23 s | yes | 9 s | 0 | 21.3 GB | 10 | 14498/857 | **69%** | right answer or refusal 0.75, expected source cited 0.67, sentences kept 0.67 |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 33 s | **no** | 40 s | 3 | 18.6 GB | 10 | 14490/981 | **69%** | right answer or refusal 0.75, expected source cited 0.67, sentences kept 0.67 |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 57 s | **no** | 4 s | 0 | 7.5 GB | 13 | 34307/2447 | **86%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 0.59 |

### ask: graph

Target: under 30 s per question. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 32 s | **no** | 9 s | 0 | 23.5 GB | 14 | 23190/997 | **100%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 1.00 |
| fast | `gemma4:26b-mlx` | `gemma4:e2b` | 36 s | **no** | 15 s | 0 | 21.0 GB | 18 | 29499/1396 | **92%** | right answer or refusal 0.75, expected source cited 1.00, sentences kept 1.00 |
| small | `gemma4:e4b` | `gemma4:e4b` | 43 s | **no** | 8 s | 0 | 1.0 GB | 16 | 31876/1236 | **100%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 1.00 |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 45 s | **no** | 19 s | 0 | 20.4 GB | 15 | 26636/1233 | **100%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 1.00 |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 48 s | **no** | 8 s | 0 | 7.5 GB | 14 | 26839/1699 | **100%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 1.00 |
| ornith-solo | `ornith-1.5:9b` | `ornith-1.5:9b` | 55 s | **no** | 1 s | 0 | 7.1 GB | 14 | 27010/1421 | **100%** | right answer or refusal 1.00, expected source cited 1.00, sentences kept 1.00 |

### cv2job

Target: under 60 s. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small | `gemma4:e4b` | `gemma4:e4b` | 39 s | yes | 3 s | 0 | 0.4 GB | 10 | 4735/649 | **90%** | requirements classified right 0.80, PDFs produced 1.00 |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 54 s | yes | 2 s | 0 | 6.7 GB | 10 | 4581/563 | **90%** | requirements classified right 0.80, PDFs produced 1.00 |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 71 s | **no** | 8 s | 0 | 20.8 GB | 15 | 7922/971 | **100%** | requirements classified right 1.00, PDFs produced 1.00 |
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 73 s | **no** | 9 s | 0 | 20.8 GB | 15 | 7922/971 | **100%** | requirements classified right 1.00, PDFs produced 1.00 |

### paper2flow

Target: under 300 s. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 416 s | **no** | 19 s | 1 | 25.5 GB | 24 | 40485/6180 | **93%** | first-draft bullets passing 0.87, claims kept 0.91, cards kept 1.00 |

### paper2post

Target: under 300 s. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 476 s | **no** | 19 s | 1 | 25.6 GB | 28 | 44626/7039 | **100%** | first-draft sentences passing 1.00, sentences kept 1.00 |

### scout

Target: under 180 s. Quality parts are between 0 and 1.

| Profile | Writer | Judge | Time | Fast enough | Loading | Swaps | Peak memory | Calls | Tokens in/out | Quality | Parts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemma-26b | `gemma4:26b-mlx` | `gemma4:26b-mlx` | 49 s | yes | 9 s | 0 | 22.0 GB | 10 | 14895/565 | **80%** | notes written 1.00, papers cited (of 5) 0.60 |
| mixed | `gemma4:26b-mlx` | `gemma4:e4b` | 49 s | yes | 8 s | 0 | 22.0 GB | 10 | 14895/565 | **80%** | notes written 1.00, papers cited (of 5) 0.60 |
| small | `gemma4:e4b` | `gemma4:e4b` | 109 s | yes | 2 s | 0 | 0.4 GB | 11 | 23570/2495 | **100%** | notes written 1.00, papers cited (of 5) 1.00 |
| ornith | `ornith-1.5:9b` | `gemma4:e4b` | 276 s | **no** | 1 s | 0 | 6.7 GB | 10 | 48581/3835 | **100%** | notes written 1.00, papers cited (of 5) 1.00 |


## Triage deciders

36 cases: 18 arXiv papers, each labelled `deep`, `post` or `skip` by hand for two sets of interests (`evals/triage/golden.yaml`). A decision model scores relevance in one pass and code maps the score to an action (`>= 3.0` deep, `>= 1.75` post); a chat model is asked for the action as validated JSON. The two thresholds were picked on these cases; picked on one set of interests and tested on the other, `clef-flash` gets 30, `nimble` 29 and `tev1` 28 of 36 right. `clef` (27B, 18 GB) does not fit next to the writer.

| Decider | Kind | Size | Action right | Read or skip right | Deep or not right | Median per paper |
|---|---|---|---|---|---|---|
| `clef-flash` | decision | 10.9 GB | 30/36 | 33/36 | 33/36 | 2.2 s |
| `nimble` | decision | 9.5 GB | 29/36 | 32/36 | 32/36 | 2.4 s |
| `tev1` | decision | 4.5 GB | 28/36 | 33/36 | 31/36 | 1.2 s |
| `gemma4:e4b` | chat | 6.6 GB | 16/36 | 25/36 | 20/36 | 3.1 s |
| `tev1:0.8b` | decision | 0.8 GB | 7/36 | 16/36 | 8/36 | 0.2 s |

## Parallel requests

Two models loaded together both answer at once, but share the GPU (each at about half speed, 15% more in total). With `OLLAMA_NUM_PARALLEL=2`, two requests to `gemma4:26b-mlx` still run one after the other (the MLX runner ignores it), and two to `gemma4:e4b` run together at 10.3 tok/s each, against 19.7 alone: no gain in throughput, and each slot costs context memory. It stays at 1.
