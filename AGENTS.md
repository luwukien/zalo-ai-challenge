# Project Agent Instructions

## Working Rules

- Inspect relevant instructions, documentation, and code before making changes.
- Keep changes focused on the requested task. Preserve unrelated user changes.
- Treat `datasets/` as original input data. Do not modify it unless explicitly required by the task. Read each dataset's README before processing it.
- Exclude secrets, `.env` files, raw datasets, model weights, caches, and generated outputs from commits by default.
- Never commit or push project data, including `datasets/` and `data/`. Keep these directories ignored, and never use `git add -f` to bypass their exclusions. This also applies to data copied to other locations.
- Run checks appropriate to the change. Discover commands from the project rather than assuming a framework or test runner.
- If automated tests do not exist, perform relevant manual checks and report their limitations.
- Communicate with the user in Vietnamese unless requested otherwise. Keep this file in English.

## Commit and Push Policy

- Use `origin` at `https://github.com/luwukien/zalo-ai-challenge.git` as the authorized GitHub remote. The initial working branch is `codex/project-rules`; use task branches for subsequent work as described below.
- Automatically commit completed, coherent changes after relevant checks pass. Do not commit after every edit or file save.
- Keep each commit focused on one purpose. Large tasks may produce multiple independently valid commits.
- Use descriptive commit messages with prefixes such as `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, and `chore:`.
- Inspect the working tree and diff before committing. Stage only files or hunks belonging to the task; do not include unrelated user changes.
- Before each push, inspect the outgoing commits and their file lists to confirm they contain no project data or secrets. Ignore rules do not protect files already tracked by Git; if data appears in tracked files or outgoing history, stop the push and resolve it first without rewriting shared history.
- Automatically push new task commits after completing and validating the task, without requesting permission each time, once the repository and intended remote are configured.
- Continue on an appropriate existing task branch. When starting from the default branch, create `codex/<short-task-description>`. Do not automatically merge into the default branch.
- Never force-push, rewrite shared history, or bypass failing checks.
- Resolve task-related check failures before committing or pushing. If environment limitations, authentication failures, or remote conflicts prevent completion, report the blocker and preserve local work.
- Report the branch, commit hash, checks performed, and push result. Confirm success only after verifying the operation succeeded.
- Explicit user instructions for a task override these defaults.

## Validation and Git Setup

- For documentation-only changes, read the completed content to verify wording and consistency. Code tests are not required unless the change affects executable behavior.
- If the project has no Git repository or intended remote configured, report commit or push as unavailable. Do not initialize Git, create a GitHub repository, or select a remote unless the user requests that setup.
