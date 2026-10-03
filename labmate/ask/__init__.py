"""ask: questions about a library of PDFs, answered with citations.

Two ways to answer the same question. :mod:`labmate.ask.graph` is a LangGraph graph: hybrid
retrieval, a grade-and-rewrite loop, a clarification ``interrupt`` and a judge-model check of
every sentence, with the code fixing the path. :mod:`labmate.ask.agent` is a tool-calling
agent that decides its own searches. Both answer through the same sentence checks.
"""
