# LLM agents fail at planning: here is the evaluation landscape

Summary of *Survey on Evaluation of LLM-based Agents* (https://arxiv.org/abs/2503.16416)

## 1. Mapping LLM Agent Evaluation Landscape

- LLM-based agents integrate models into multi-step workflows with external tools for autonomous planning.
  - p. 1 (Introduction): "LLM-based agents address those gaps by building on LLMs as a backbone, integrating them into multi-step workflows and equipping them with external tools (Wang et al., 2024a)."
- This survey presents the first overview of LLM-based agent evaluation by mapping the current landscape.
  - p. 1 (Introduction): "In this survey, we present the first overview of LLM-based agent evaluation. We aim to benefit developers, benchmark creators, practitioners, and researchers by mapping the current evaluation landscape and identifying key gaps for future research."
- The survey covers fundamental capabilities like planning, tool use, self-reflection, and memory.
  - p. 1 (Introduction): "We begin by discussing the evaluation of fundamental LLM-based agent capabilities (§2). These include planning, tool use, self-reflection, and memory. We then review benchmarks and evaluation strategies for prominent types of agentic applications: web agents, software engineering agents, scientific agents, and conversational agents (§3)."
- It reviews benchmarks for web, software engineering, scientific, and conversational agents.
  - p. 1 (Introduction): "We begin by discussing the evaluation of fundamental LLM-based agent capabilities (§2). These include planning, tool use, self-reflection, and memory. We then review benchmarks and evaluation strategies for prominent types of agentic applications: web agents, software engineering agents, scientific agents, and conversational agents (§3)."

## 2. Why Evaluating Agents Is Hard

- Evaluation must go beyond textual outputs to assess sequential decision-making in dynamic environments.
  - p. 1 (Introduction): "Such evaluation must go beyond measuring LLM textual outputs to assess an agent’s capacity for sequential decision-making and operation within dynamic environments."
- Benchmarks reveal current methods struggle with long-range consistency and dynamic memory handling.
  - p. 1 (Agent Capabilities Evaluation): "For semantic memory, benchmarks assess retrieval effectiveness and long-range understand-ing, revealing that current methods remain limited in maintaining long-range consistency and han-dling dynamic memory (Tan et al., 2025; Hu et al., 2025; Wu et al., 2025)."
- Recent work exhibits over-optimistic estimates, proposing Online-Mind2Web as a more rigorous alternative.
  - p. 3 (Application-Specific Agents Evaluation): "However, recent work suggests it exhibits over-optimistic performance estimates, and proposes Online-Mind2Web as a more rigorous alternative that remains challenging for current agents (Xue et al., 2025)."
- Most benchmarks conflate LLM capabilities with agent harness design, obscuring performance attribution.
  - p. 8 (Discussion): "Most current agent benchmarks conflate two distinct evaluation targets: (1) the inherent capabilities of the backbone LLM, and (2) the design of the agent Harness (a.k.a. scaffold)."

## 3. Benchmarks: WebArena, SWE-bench, Gaia

- WebArena provides dynamic environments with fully functional websites across multiple domains.
  - p. 3 (Application-Specific Agents Evaluation): "In contrast, WebArena introduces a dynamic environment featuring fully functional websites across multiple domains, enriched with auxiliary tools and knowledge sources."
- Gaia requires reasoning, multi-modality, web browsing, and tool-use on real-world questions.
  - p. 5 (Generalist Agent Evaluation): "Gaia is composed of real-world questions that require abilities such as reasoning, multi-modality handling, web browsing, and tool-use proficiency."

## 4. Performance Gaps and Critical Limitations

- State-of-the-art models struggle with long-horizon planning tasks.
  - p. 1 (Agent Capabilities Evaluation): "Results show that even SOTA models struggle with long-horizon planning."
- SWE-bench Pro performance remains below 25% Pass@1 for complex code changes.
  - p. 3 (Application-Specific Agents Evaluation): "Model performance remains below 25% Pass@1, highlighting current limitations in handling long-horizon, complex code changes."
- No standardized benchmark exists for assessing self-reflection capabilities.
  - p. 1 (Agent Capabilities Evaluation): "Despite these efforts, a standardized benchmark or methodology for assessing self-reflection remains a critical gap."
- Evaluations overlook cost, creating resource-intensive agents hard to deploy.
  - p. 8 (Discussion): "Current evaluations often prioritize performance while overlooking cost and efficiency measurements. This emphasis can inadvertently drive the development of highly capable but resource-intensive agents, limiting their practical deployment Kapoor et al. (2024b)."
