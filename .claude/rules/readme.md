---
description: Style guide for the plaquette README.
paths:
  - README.md
---

# plaquette README style guide

The README describes only what a newcomer can run today, in a neutral,
declarative tone, as a few short paragraphs of connected prose — readable in
under a minute. Follow the prose conventions in `CLAUDE.md`.

## Before writing

Confirm every concrete claim against the code: `src/lib.rs` (what the crate
exposes), the per-model examples under `examples/` (what a run does and prints)
and their `.toml` files (run parameters), `tests/` (what the end-to-end tests
check), and `Cargo.toml` (crate name and edition). If a claim can't be confirmed,
soften it to what the code shows and tell the user.

## Minimal and honest

Describe only what exists and runs today — no roadmap, unbuilt models, API
reference, or architecture tour. The device a model runs on (CPU/GPU) is a plain
fact for the Models table, not a performance pitch. Name the specific models
that run, but don't inflate them into a claim about the framework's reach: that
a Z2 gauge model runs today does not make plaquette a lattice gauge theory
framework, which stays a longer-term aim with much of the general machinery
still to build. Keep the tone neutral and plain, and
avoid hyperbole and absolute claims ("reproduces the same numbers", not
"bit-for-bit"). Write in a declarative technical register — state what the
software does rather than editorialize about its maturity or narrate what "runs
today". A brief, plain style invites conversational tics; keep them out: no
colon-hedges ("It is early-stage:"), no cleft constructions ("what runs today
is …"), no capability apologies. A plain verb ("implements", "provides") carries
the fact without them, and "currently" already implies more is coming. Keep
top-level headers to conventional labels (`Installation`, `Usage`, `License`),
not descriptive phrases.

## Content, in order

Short paragraphs of flowing prose, not a table or feature list:

1. An opening line stating the goal: a Rust framework for Monte Carlo simulation
   of lattice models. No status or maturity sentence — which models run shows
   through the Usage examples, not a narrated line up front.
2. An AI-assistance acknowledgment after it, in GitHub alert notation,
   generic (don't name the model):

   ```markdown
   > [!NOTE]
   > Developed with AI assistance.
   ```

3. `## Installation`: builds with a recent Rust toolchain, then `cargo build`.
4. `## Models`: a table of the models that run, columns for the model, its lattice
   dimension, the update implementation (Metropolis or checkerboard), and the
   device (CPU/GPU). Rows for what exists — no unbuilt combinations.
5. `## Usage`: note examples live in `examples/` (grouped one directory per
   model), then give a run for each model —
   `cargo run --example ising -- examples/ising/run.toml` and
   `cargo run --example gauge -- examples/gauge/gauge.toml`. Note runs are
   configured by a TOML file, not by editing source; name a few parameters (such
   as lattice shape, temperature, sample count) with "such as", and mention a
   seed makes a run reproducible.
6. `## Documentation`: model and method docs live in `docs/`.
7. `## Testing`: a short pointer that correctness is checked by the test suite —
   the unit tests plus the end-to-end tests under `tests/` that run each model
   against well-known results, with the reference physics in `docs/`. Give
   `cargo test`. Don't enumerate individual tests or the specific phenomena;
   point to where they live.
8. `## AI assistance`: developed with AI assistance; conventions live as
   `paths:`-scoped rules under `.claude/rules/`.
9. A license line only if a license file exists.

## Leave out

Swappable-seam design, generic-over-dimension structure, statistics methodology,
`docs/` internals, what a run reports, and anything aspirational — the code and
`docs/` carry that. A one-line pointer to the test suite is fine (see above); the
validation methodology is not.
