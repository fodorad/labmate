# Gallery

Published runs, one folder per paper. Each folder holds the run's step artifacts, the
carousel, the LinkedIn post, the trace of the run that produced it and the cassettes of
every model call in that trace, so anyone can replay it without a model:

```bash
make verify                  # every entry
make verify PAPER=1706.03762 # one entry
```

Add a paper after a finished run with `make publish ARXIV=<id>`; `make site` builds the
static site that GitHub Pages serves.
