"""scout: a research agent that finds papers on a topic and writes notes about them.

A tool-calling agent decides what to search on arXiv, which abstracts to read, which few papers
deserve a deep look (it can run paper2flow on them), and when it knows enough to write its notes.
The notes may cite only papers the agent has read.
"""
