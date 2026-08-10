---
title: Planning & Autonomy
description: High-level task planning over trained robot controllers and measured simulator state.
---

# Planning & Autonomy

Planning and autonomy tasks choose and sequence higher-level robot actions while a lower-level controller executes
them. The current implementation is [LLM High-Level Planning](llm-high-level-planning), a registered evaluation task
family for language-model decisions over bounded primitives with physical feedback and safety checks.

The first adapter uses Go2+D1, but this category is organized by the planning objective rather than by robot platform.
