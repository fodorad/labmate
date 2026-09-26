# Why AI agents fail at AAA games

Summary of *GameHorizon Suite: Multi-Horizon Data and Evaluation in Gameplay* (https://arxiv.org/abs/2609.25001)

## 1. Measuring Multi-Horizon Gameplay

- The paper introduces GameHorizon, a unified data and evaluation framework.
  - p. 1 (Introduction): "To address these challenges, as presented in Fig. 1, we introduce GameHorizon, a data and evaluation suite spanning multiple temporal horizons and AAA games."
- The corpus comprises 5,000 hours of human gameplay across 21 game titles.
  - p. 4 (GameHorizon Suite): "the corpus comprises 5,000 hours of human gameplay across 21 game titles"
- The suite evaluates long-horizon gameplay through order-dependent causal tasks and order-flexible thematic tasks.
  - p. 4 (GameHorizon Suite): "It evaluates long-horizon gameplay through order-dependent causal tasks and order-flexible thematic tasks, each comprising 2–6 verifiable short-horizon subtasks."
- Each task contains 2-6 verifiable subtasks to assess short-horizon capabilities.
  - p. 4 (GameHorizon Suite): "It evaluates long-horizon gameplay through order-dependent causal tasks and order-flexible thematic tasks, each comprising 2–6 verifiable short-horizon subtasks."

## 2. Narrow Coverage and Reproducibility Issues

- Existing datasets like GameWorld target only simple mini-games, resulting in narrow game coverage.
  - p. 1 (Introduction): "First, constrained by annotation cost, their game coverage is narrow. For instance, GameWorld (Ouyang et al., 2026) targets simple mini-games."
- Prior methods use small samples for low-confidence comparisons and online rollouts that are hard to reproduce.
  - p. 1 (Introduction): "Such small samples lead to low-confidence comparisons. Online results are sensitive to specific game environments and agent harnesses, making them difficult to reproduce."
- Earlier work focuses on Minecraft or non-AAA games.
  - p. 4 (GameHorizon Suite): "Prior work tests limited model types mainly on Minecraft (Fan et al., 2022; Zheng et al., 2025; Ju et al., 2026) or non-AAA games (Paglieri et al., 2025; Zhang et al., 2026; Ouyang et al., 2026)."

## 3. Annotator Pipeline and Benchmark Design

- Action-aware segmentation uses keyboard-mouse traces to identify key action transitions and determine clip boundaries.
  - p. 4 (GameHorizon Suite): "To address this issue, we perform action-aware segmentation using keyboard-mouse traces to identify key action transitions and determine clip boundaries."
- Dynamic programming enforces duration ranges: 1–5 seconds for L1, 1–2 minutes for L2, and 5–8 minutes for L3.
  - p. 4 (GameHorizon Suite): "Besides, we apply a dynamic programming algorithm (Bellman, 1966) to enforce level-specific duration ranges of 1–5 seconds for L1, 1–2 minutes for L2, and 5–8 minutes for L3."
- Unified multiple-choice questions ensure standardized evaluation where models directly output selections from four options.
  - p. 9 (Experiments): "We adopt unified MCQs for all offline tasks. Each model receives the same four-option questions and directly outputs the selections, ensuring standardized and reproducible evaluation."

## 4. Proprietary Models Lead, UMMs Lag

- Mean accuracy is 64.7%, substantially above the 25% random baseline.
  - p. 9 (Experiments): "The mean accuracy across our benchmark is 64.7%, substantially above the 25% random baseline, suggesting that the designed tasks are solvable yet challenging for meaningful evaluation."
- Proprietary model GPT-6-Astra ranks first with 80.2% overall accuracy.
  - p. 9 (Experiments): "GPT-6-Astra ranks first with 80.2% overall accuracy"
- Best UMM BAGEL-7B-MoT achieves 61.4%, falling into Tier 3 or Tier 4.
  - p. 9 (Experiments): "All four models fall into Tier 3 or Tier 4, indicating that current UMMs remain less competitive in gameplay tasks... BAGEL-7B-MoT achieves 61.4%"
