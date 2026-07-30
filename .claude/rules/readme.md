---
description: Style guide for the plaquette README.
paths:
  - README.md
---

# plaquette README style guide

The README describes only what a newcomer can run today, in a neutral tone, as a
few short flowing paragraphs — readable in under a minute. Follow the prose
conventions in `CLAUDE.md`.

## Before writing

Confirm every concrete claim against the code: `src/lib.rs` (what the crate
exposes), `examples/ising.rs` (what a run does and prints), `examples/run.toml`
(run parameters), `Cargo.toml` (crate name and edition). If a claim can't be
confirmed, soften it to what the code shows and tell the user.

## Minimal and honest

Describe only what exists and runs today — no roadmap, unbuilt models,
GPU/parallelism, API reference, or architecture tour. Lattice gauge theory is a
longer-term aim, not a current capability. Keep the tone neutral and plain, and
avoid hyperbole and absolute claims ("reproduces the same numbers", not
"bit-for-bit"). Keep top-level headers to conventional labels (`Installation`,
`Usage`, `License`), not descriptive phrases.

## Content, in order

Short paragraphs of flowing prose, not a table or feature list:

1. One line on the goal: a Rust framework for Monte Carlo simulation of lattice
   models.
2. An AI-assistance acknowledgment right after, in GitHub alert notation, generic
   (don't name the model):

   ```markdown
   > [!NOTE]
   > Developed with AI assistance.
   ```

3. One sentence of status: early-stage; 2D Ising, single-spin Metropolis on CPU is
   what runs today.
4. `## Installation`: builds with a recent Rust toolchain, then `cargo build`.
5. `## Usage`: note examples live in `examples/`, then give the Ising run —
   `cargo run --example ising -- examples/run.toml`. Note runs are configured by a
   TOML file, not by editing source; name a few parameters (such as lattice shape,
   temperature, sample count) with "such as", and mention a seed makes a run
   reproducible.
6. `## Documentation`: model and method docs live in `docs/`.
7. `## AI assistance`: developed with AI assistance; conventions live as
   `paths:`-scoped rules under `.claude/rules/`.
8. A license line only if a license file exists.

## Leave out

Swappable-seam design, generic-over-dimension structure, statistics methodology,
`docs/` internals, what a run reports, validation, and anything aspirational — the
code and `docs/` carry that.
