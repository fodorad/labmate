# Stop overfitting your agent harness to benchmarks

Summary of *RRSI: Regularized Recursive Self-Improvement of Agent Harnesses* (https://arxiv.org/abs/2609.24972)

## 1. Agent Harness Evolution Problem

- The paper defines an agent as a combination of a backbone policy and a harness managing prompts, control flow, tools, and memory.
  - p. 3 (Preliminaries): "We consider an agent 𝐴= (𝜋, 𝐻) built from a backbone policy 𝜋and a harness 𝐻. The harness is everything around the weights...: the system and task prompts, the control flow that decides when the agent plans, acts, reflects or stops, the tool interfaces and their descriptions, the memory and skill files the agent may consult, and the context management that decides what the policy sees at each step."
- Task performance is the expected score on a task set, while policy-token cost is the expected number of tokens consumed.
  - p. 3 (Preliminaries): "For a task set D, we measure task performance and policy-token cost as 𝑆(𝐻; D) = 𝔼𝑥∼D 𝔼𝜏∼𝐴(·|𝑥) [𝑟(𝑥, 𝜏)], 𝐶(𝐻; D) = 𝔼𝑥∼D 𝔼𝜏∼𝐴(·|𝑥) [𝑐(𝜏)], where 𝑐(𝜏) is the number of policy tokens consumed by the trajectory."
- Harness evolution treats the harness as the optimization variable while keeping the backbone policy fixed.
  - p. 3 (Preliminaries): "Harness evolution treats 𝐻as the optimization variable while keeping the backbone policy fixed (Lee et al., 2026b)."

## 2. Overfitting Risks in Iterative Evolution

- Iterative harness evolution uses finite feedback, creating adaptive overfitting where evolve-set performance fails to generalize to unseen tasks.
  - p. 1 (Introduction): "test-time harness evolution repeatedly proposes and selects edits using feedback from a finite evolve set, creating an adaptive overfitting risk: evolve-set performance may improve without corresponding gains on unseen tasks."
- Search encodes benchmark-specific patterns and promotes candidates favored by evaluation noise, accumulating complexity without improving the agent mechanism.
  - p. 1 (Introduction): "The evolution search may encode benchmark-specific patterns, promote candidates favored by the evaluation noise, or accumulate complexity that improves evolve-set scores without improving the underlying agent mechanism."
- Existing approaches rely on benchmark scores without generalization terms, so reported gains often do not survive changes in the evaluation suite.
  - p. 10 (Related Work): "Throughout, the search is driven by the score on the suite it optimizes against, with no term for generalization, and the cost is not hypothetical: reported gains often do not survive a change of suite"

## 3. RRSI Regularization Framework Details

- RRSI anneals update capacity per round, uses evidence-aware credit assignment, and structures exploration.
  - p. 3 (Preliminaries): "RRSI regularizes it in three ways: it anneals how much update capacity a single round may exercise, it makes credit assignment evidence-aware over the whole run, and it structures where that capacity is spent."
- Leakage screening rejects candidates encoding task names or specific answers before full evaluation.
  - p. 3 (Preliminaries): "Before full evaluation, a critic reads each candidate diff and rejects edits that explicitly encode task names, entity names, task-specific values, answers, or other logic specific to the evolve benchmark"
- Stability-aware acceptance uses a noise-adjusted floor to prevent selecting candidates due to stochastic variation.
  - p. 3 (Preliminaries): "A candidate must satisfy the noise-adjusted floor ˆS(H′) ≥ S★−δ. The floor prevents the search from walking downhill through a sequence of regressions that are individually small enough to be mistaken for noise."

## 4. Results, Gains, and Limitations

- RRSI improves six held-out splits by up to 4.7 points out of distribution using fewer policy tokens than unregularized evolution.
  - p. 1 (Introduction): "it gains up to 14.1 points on the evolving split and improves all six held-out splits, by up to 4.7 points out of distribution, on fewer policy tokens than unregularized evolution spends."
- The method gains 2.3 points on Harvey LAB and 1.8 points on SWE-bench Verified for in-distribution held-out tasks.
  - p. 6 (Experiments): "SWE-bench Verified gains 1.8 points although repository-level bug fixing was never scored. The in-distribution held-out split of Harvey LAB gains 2.3"
- Removing acceptance constraints raises the evolve-set score to 91.5 but drops out-of-distribution transfer from 43.6 to 41.0 and increases token cost.
  - p. 6 (Experiments): "Without the acceptance constraints the evolve-set score rises from 90.5 to 91.5 while the out-of-distribution average falls from 43.6 to 41.0 and token cost rises by half"
