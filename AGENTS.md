# HealthVault development

HealthVault contains a Python application in `healthvault/`, the shared Agent runtime in `agents/`, a React frontend in `web-ui/`, and tests in `tests/`. Setup and demonstration instructions are in `README.md`.

Read the existing implementation before making a change. Keep changes focused, preserve unrelated work, and run the relevant offline tests. Tests use synthetic records and mocked model transports; real model calls require explicit authorization. Changes to the shared runtime must preserve its default behavior.

Confirmed health records belong in the structured database with their original evidence. Report drafts require explicit user confirmation before persistence. Account identity and tool permissions come from the server. Skills may describe workflow steps, but must not change medical facts, knowledge sources, account boundaries, or permissions. Personal Skills remain scoped to their owner.

Only source code, setup instructions, public knowledge snapshots, project Skills, and synthetic test inputs belong in this repository. Keep API keys, local configuration, databases, uploaded reports, conversation logs, model responses, evaluation results, and internal development notes outside Git. Check the staged files before committing. Never publish a branch containing private development history.
