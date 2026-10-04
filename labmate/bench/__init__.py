"""bench: which model combination is fast enough, fits in memory, and answers well.

``labmate bench micro`` measures single models (loading, speed, structured output, tool calls, and
how well a model judges planted errors); ``labmate bench macro`` runs each use case on a small
fixed workload per model profile and scores the result. Neither is part of ``make check``.
"""
