RRSI fixes agent harness overfitting via regularized self-improvement.

Agent harnesses optimize for benchmark scores, causing adaptive overfitting that fails on unseen tasks.

RRSI treats the harness as the variable while keeping the backbone policy fixed.

RRSI improves six held-out splits by up to 4.7 points using fewer tokens than unregularized evolution.

How do you currently prevent your agent systems from overfitting to specific evaluation benchmarks?

Paper: RRSI: Regularized Recursive Self-Improvement of Agent Harnesses
https://arxiv.org/abs/2609.24972

Made with paper2carousel: every claim on the slides is traced to a quote in the paper.
