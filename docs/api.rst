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

Shared by every feature: local models with record/replay, the LangChain building blocks
over them, tracing, ingest, claim extraction and the fact-check loop.

.. automodule:: labmate.core
.. automodule:: labmate.core.llm.types
   :members:
   :show-inheritance:
.. automodule:: labmate.core.llm.client
   :members:
   :show-inheritance:
.. automodule:: labmate.core.llm.structured
   :members:
.. automodule:: labmate.core.llm.replay
   :members:
   :show-inheritance:
.. automodule:: labmate.core.model
   :members:
.. automodule:: labmate.core.chat
   :members:
   :show-inheritance:
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
.. automodule:: labmate.core.phases
   :members:
.. automodule:: labmate.core.tracing
   :members:
   :show-inheritance:
.. automodule:: labmate.core.theme
   :members:
.. automodule:: labmate.core.probe
   :members:
   :show-inheritance:

paper2flow
----------

A paper in, ``overview.pdf`` out: a LangChain chain.

.. automodule:: labmate.paper2flow
.. automodule:: labmate.paper2flow.chain
   :members:
.. automodule:: labmate.paper2flow.schemas
   :members:
   :exclude-members: Bullet, BulletCheck, BulletVerdict, Card, CardVerdicts, Cards, ClaimCard, ClaimDraft, Claims, FactCheckReport, FactChecked, Figure, Paper, Section, SectionClaims
.. automodule:: labmate.paper2flow.steps.publication
   :members:
.. automodule:: labmate.paper2flow.steps.route
   :members:
.. automodule:: labmate.paper2flow.steps.outline
   :members:
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

Questions about your research, answered from the dissertation and papers with citations.

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
.. automodule:: labmate.ask.schemas
   :members:
.. automodule:: labmate.ask.graph
   :members:
.. automodule:: labmate.ask.verify
   :members:
.. automodule:: labmate.ask.agent
   :members:
.. automodule:: labmate.ask.lc
   :members:
.. automodule:: labmate.ask.evals
   :members:
.. automodule:: labmate.ask.live
   :members:
.. automodule:: labmate.ask.session
   :members:
.. automodule:: labmate.ask.studio
   :members:
.. automodule:: labmate.ask.cli
   :members:
.. automodule:: labmate.ask.dashboard
   :members:
.. automodule:: labmate.diagrams
   :members:
