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

Describe only what exists and runs today — no roadmap, unbuilt models, or API
reference. The device a model runs on (CPU/GPU) is a plain
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

The sections below are the complete set. Adding, renaming, or removing one is
the maintainer's decision, never an agent's — content that does not fit an
existing section does not go in the README. Containment is the point: each
code-specific claim lives in exactly one section, so a change in the code means
rewriting one place rather than hunting the document for every paragraph that
assumed the old shape.

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
5. `## Architecture`: a short paragraph describing what a run does — sweeps, a
   snapshot every fixed number of them, a measurement per snapshot, an average
   with an error bar — in plain terms and naming no types, so a reader meets
   the flow before the vocabulary. Then two subsections and no others —
   `### Objects`, holding one
   fixed-width diagram of how they relate, and `### Snapshot`, holding a
   sentence that names the driver and states the chain is already thermalized,
   then a numbered walk-through of producing a single snapshot and measuring
   it. The list stops there; what spans a whole run, such as reducing many
   samples, goes in a closing sentence after it. The list runs past the
   diagram's last box, to measuring a snapshot and reducing the results, but
   every type the diagram does show must be named the same way in both. This is
   the only section that names the seams; keep it to structure — which object
   calls which — and leave design rationale to the code and `docs/`. No function
   signatures, generic parameters, or API-reference detail.
6. `## Usage`: note examples live in `examples/` (grouped one directory per
   model), then give a run for each model —
   `cargo run --example ising -- examples/ising/run.toml` and
   `cargo run --example gauge -- examples/gauge/gauge.toml`. Note runs are
   configured by a TOML file, not by editing source; name a few parameters (such
   as lattice shape, temperature, sample count) with "such as", and mention a
   seed makes a run reproducible.
7. `## Documentation`: model and method docs live in `docs/`.
8. `## Testing`: a short pointer that correctness is checked by the test suite —
   the unit tests plus the end-to-end tests under `tests/` that run each model
   against well-known results, with the reference physics in `docs/`. Give
   `cargo test`. Don't enumerate individual tests or the specific phenomena;
   point to where they live.
9. `## AI assistance`: developed with AI assistance; conventions live as
   `paths:`-scoped rules under `.claude/rules/`.
10. A license line only if a license file exists.

## Leave out

Swappable-seam design, generic-over-dimension structure, statistics methodology,
`docs/` internals, what a run reports, and anything aspirational — the code and
`docs/` carry that. These stay out of every section's prose and appear only
where a section explicitly owns the subject: the Models table names the update
implementation and the device because that is what the table is for, and the
prose around it elaborates on neither. A one-line pointer to the test suite is
fine (see above); the validation methodology is not.
