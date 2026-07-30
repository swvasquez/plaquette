# plaquette

plaquette is a Rust framework for Monte Carlo simulation of lattice models.

> [!NOTE]
> Developed with AI assistance.

It is early-stage: the 2D Ising model, sampled by single-spin Metropolis on the
CPU, is what runs today.

## Installation

plaquette builds with a recent Rust toolchain (edition 2024). From a clone of the
repository:

```bash
cargo build
```

## Usage

Runnable example code lives in the `examples/` folder. To run the 2D Ising model,
for example:

```bash
cargo run --example ising -- examples/run.toml
```

Model runs are set with a TOML config file rather than by editing source, so
trying a different run is an edit to that file rather than a recompile. The file
holds the run's parameters — such as the lattice shape, temperature, and number
of samples — along with a `seed` that makes the run reproducible.

## Documentation

Documentation describing the models and methods can be found in [`docs/`](docs/).

## AI assistance

plaquette is developed with AI assistance. The conventions that shape the code and
docs are written down as rules under `.claude/rules/`, each a Markdown file scoped
by a `paths:` glob to the files it applies to (for example, the README style guide
in `.claude/rules/readme.md`).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT
license](LICENSE-MIT) at your option.
