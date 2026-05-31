# terra-incognita

The MLflow **training + serving** pipeline for the camera-trap project.

## SSoT

The single source of truth for cross-repo coordination, shared contracts, and
each repo's design lives in `.plans/`, a **read-only mirror** of the
[`terra-carta`](https://github.com/kevinher7/terra-carta) repo (vendored via
`git subtree`). When a question touches another repo's design or a shared
interface, read it from `.plans/` rather than guessing or restating it here.
