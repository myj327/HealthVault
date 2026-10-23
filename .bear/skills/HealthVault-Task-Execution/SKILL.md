---
name: HealthVault-Task-Execution
description: Execute read-only HealthVault tasks with explicit evidence and validation.
---

## When to Use
Use for HealthVault timeline, indicator comparison, knowledge search, and visit-brief tasks.

## Prerequisites
Use only the tools exposed by the HealthVault runtime and the injected profile context.

## Steps
1. Identify the requested HealthVault outcome before selecting a tool.
2. Run only approved read-only domain tools and preserve returned citations.
3. Return the tool status accurately; never present a rejection as success.

## Failure Handling
If evidence is insufficient or a tool rejects input, report that result without guessing.

## Verification
Check the final status, requested fields, and every citation before answering.

## Safety Boundaries
Never infer patient facts or medical advice, expose secrets, or invoke file, shell, MCP, or write tools.
