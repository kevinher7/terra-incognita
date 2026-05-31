---
name: update-ssot
description: Update the single source of truth (.plans/) when the project's architecture or design decisions change.
allowed-tools: [Bash, Read, Edit, Write]
user-invocable: true
---

# Update SSoT Skill

`.plans/` is the **single source of truth** for how every repo in the camera-trap
project coordinates — shared contracts and each repo's design. It is a
**read-only mirror** of the [`terra-carta`](https://github.com/kevinher7/terra-carta)
repo, vendored into this repo via `git subtree`.

Because it is a mirror, **any change to architecture or design decisions must be
made in `terra-carta` first, then pulled back into `.plans/`**. Editing files
under `.plans/` directly in this repo does nothing useful — the edits are
overwritten on the next `subtree pull`.

## When to use

Use this skill whenever a change in this repo reflects or requires a change to
the shared design, such as:

- A new or changed cross-repo contract (e.g. `serving-io.md`, `bbox-format.md`,
  `model-registry.md`, `dataset-conventions.md`, `mlflow-topology.md`).
- A design decision in this training/serving repo that belongs in
  `.plans/training/` (`CONTEXT.md`, `PLAN.md`).
- Any architectural decision that affects how repos in the project coordinate.

If a change is purely internal to terra-incognita and touches no shared contract
or documented design, it does **not** need an SSoT update.

## How to update

### 1. Make the change upstream in `terra-carta`

The authoritative source is the `terra-carta` repo, **not** `.plans/`. Clone or
open `terra-carta` separately, edit the relevant file there, commit, and push to
its `main` branch.

```bash
# in a clone of terra-carta
git clone https://github.com/kevinher7/terra-carta   # if not already cloned
# edit the relevant plan/contract file
git commit -am "Describe the architecture/design change"
git push origin main
```

### 2. Refresh the mirror in this repo

Back in `terra-incognita`, pull the upstream changes into `.plans/` via subtree:

```bash
git subtree pull --prefix=.plans https://github.com/kevinher7/terra-carta main --squash
```

This produces a squashed merge commit that updates `.plans/` to match
`terra-carta`. Commit it (the subtree command commits for you with `--squash`).

### 3. Verify

- Confirm the changed file under `.plans/` now shows the upstream content.
- Confirm `git log` shows the squashed subtree merge commit.
- Update anything in this repo that depends on the changed contract/design.

## Rules

- **Never** edit files under `.plans/` directly in this repo. They are a mirror.
- **Never** redefine a contract locally in this repo; link to the file in
  `.plans/contracts/` instead.
- The flow is always: edit upstream → push → `subtree pull` here.
