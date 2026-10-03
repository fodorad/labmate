API Reference
=============

.. automodule:: labmate
   :members:

.. automodule:: labmate.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: labmate.cli
   :members:

Core
----

Shared by every feature: chat models with a reply cache, the LangChain building blocks,
ingest, claim extraction and the fact-check loop.

.. automodule:: labmate.core
.. automodule:: labmate.core.chat
   :members:
   :show-inheritance:
.. automodule:: labmate.core.structured
   :members:
.. automodule:: labmate.core.lc
   :members:
   :show-inheritance:
.. automodule:: labmate.core.schemas
   :members:
.. automodule:: labmate.core.ingest
   :members:
.. automodule:: labmate.core.figures
   :members:
.. automodule:: labmate.core.words
   :members:
.. automodule:: labmate.core.extract
   :members:
.. automodule:: labmate.core.factcheck
   :members:
.. automodule:: labmate.core.theme
   :members:

paper2flow
----------

A paper in, ``overview.pdf`` out: a LangChain chain.

.. automodule:: labmate.paper2flow
.. automodule:: labmate.paper2flow.chain
   :members:
.. automodule:: labmate.paper2flow.schemas
   :members:
   :exclude-members: Bullet, BulletCheck, BulletVerdict, Card, CardVerdicts, Cards, ClaimCard, ClaimDraft, Claims, FactCheckReport, FactChecked, Figure, Paper, Section, SectionClaims
.. automodule:: labmate.paper2flow.steps.write
   :members:
.. automodule:: labmate.paper2flow.steps.flow
   :members:
.. automodule:: labmate.paper2flow.steps.render
   :members:
.. automodule:: labmate.paper2flow.evals.metrics
   :members:

paper2post
----------

A paper in, ``post.pdf`` out: a LangChain chain that extends paper2flow's analysis.

.. automodule:: labmate.paper2post
.. automodule:: labmate.paper2post.chain
   :members:
.. automodule:: labmate.paper2post.schemas
   :members:
.. automodule:: labmate.paper2post.post
   :members:
.. automodule:: labmate.paper2post.render
   :members:

ask
---

Questions about a library of documents, answered with citations by a graph and by an agent.

.. automodule:: labmate.ask
.. automodule:: labmate.ask.library
   :members:
.. automodule:: labmate.ask.chunk
   :members:
.. automodule:: labmate.ask.embed
   :members:
.. automodule:: labmate.ask.index
   :members:
.. automodule:: labmate.ask.build
   :members:
.. automodule:: labmate.ask.evidence
   :members:
.. automodule:: labmate.ask.answer
   :members:
.. automodule:: labmate.ask.schemas
   :members:
.. automodule:: labmate.ask.graph
   :members:
.. automodule:: labmate.ask.agent
   :members:
.. automodule:: labmate.ask.evals
   :members:
.. automodule:: labmate.ask.session
   :members:
.. automodule:: labmate.ask.studio
   :members:
.. automodule:: labmate.ask.cli
   :members:
.. automodule:: labmate.diagrams
   :members:
