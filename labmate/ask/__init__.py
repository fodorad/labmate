"""ask: questions about your research, answered from your dissertation and papers.

An agent, built with LangGraph: tiered hybrid retrieval over a library (the dissertation
is the source of truth), grading and query-rewriting loops, a conflict check across
sources, answers with page citations, self-verification with the shared fact-check loop,
and multi-turn memory. A prebuilt LangChain tool-calling agent is kept as a baseline to
compare against.
"""
