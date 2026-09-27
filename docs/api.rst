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

Shared by every feature: local models with record/replay, tracing, ingest, claim
extraction and the fact-check loop.

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
.. automodule:: labmate.core.parallel
   :members:
.. automodule:: labmate.core.tracing
   :members:
   :show-inheritance:
.. automodule:: labmate.core.traceview
   :members:
.. automodule:: labmate.core.theme
   :members:
.. automodule:: labmate.core.probe
   :members:
   :show-inheritance:

paper2flow
----------

A paper in, ``overview.pdf`` and ``post.pdf`` out.

.. automodule:: labmate.paper2flow
.. automodule:: labmate.paper2flow.schemas
   :members:
   :exclude-members: Bullet, BulletCheck, BulletVerdict, ClaimCard, ClaimDraft, Claims, FactCheckReport, FactChecked, Figure, Paper, Section, SectionClaims, SlideText, SlideVerdicts, WrittenSlides
.. automodule:: labmate.paper2flow.steps.publication
   :members:
.. automodule:: labmate.paper2flow.steps.route
   :members:
.. automodule:: labmate.paper2flow.steps.outline
   :members:
.. automodule:: labmate.paper2flow.steps.gate
   :members:
.. automodule:: labmate.paper2flow.steps.write
   :members:
.. automodule:: labmate.paper2flow.steps.post
   :members:
.. automodule:: labmate.paper2flow.steps.flow
   :members:
.. automodule:: labmate.paper2flow.steps.render
   :members:
.. automodule:: labmate.paper2flow.engines.common
   :members:
.. automodule:: labmate.paper2flow.engines.plain
   :members:
.. automodule:: labmate.paper2flow.engines.langgraph_engine
   :members:
.. automodule:: labmate.paper2flow.evals.metrics
   :members:
.. automodule:: labmate.paper2flow.evals.labels
   :members:
.. automodule:: labmate.paper2flow.evals.agreement
   :members:
.. automodule:: labmate.paper2flow.gallery
   :members:
