# 0001 · One monorepo, packages and apps

Date: 2026-10-21 · Status: accepted

## Decision
**orb is one repository with two kinds of folders.**
- `packages/` holds reusable parts ("cookie-cutter components"). A package has no knowledge of a product.
  Another project can take one package and use it alone.
- `apps/` holds runnable products. An app wires packages together and owns its configuration.

Python packages use the src layout and their own `pyproject.toml`. They are installed in editable mode into
one conda env (`orb`). Go modules are joined by `go.work` at the repo root.

## Why
- The products that come next (Doc Brain, the onboarding agent, Slack) need the same parts: the Jev reader,
  the LLM client, the agent interface, speech, the voice pipeline. One repo keeps them in step.
- A package boundary makes reuse a rule, not a hope: a package must not import an app.

## How another project reuses a part
- Python: `pip install "git+<repo url>#subdirectory=packages/python/orb-discussion"`.
- Go: `go get github.com/beingPro007/orb/packages/go/orb-plugins` (the module path assumes the repo will live at
  github.com/beingPro007/orb; change it in one place, the go.mod files, if it lives elsewhere).

## What was removed
The GitHub-issues research (the cleaning pipeline, 498 MB of raw data, the notebook) and the old weighted chat
app. A full copy is at `~/orb-archive/datacleanerproj-2026-10-21.tar.gz` (490 MB). See 0004.
