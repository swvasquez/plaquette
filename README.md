# plaquette

`plaquette` is a Rust framework for Monte Carlo simulation of lattice models.

> [!NOTE]
> Developed with AI assistance.

## Installation

`plaquette` is a Rust package and can be built via Cargo

```bash
cargo build
```

## Models

`plaquette` currently implements three models:

| Model    | Dimension | Implementation     | Device   |
| -------- | --------- | ------------------ | -------- |
| Ising    | D ≥ 1     | Metropolis         | CPU      |
| Ising    | D ≥ 1     | Site checkerboard  | CPU, GPU |
| Potts    | D ≥ 1     | Metropolis         | CPU      |
| Potts    | D ≥ 1     | Site checkerboard  | CPU, GPU |
| Z2 gauge | D ≥ 2     | Metropolis         | CPU      |
| Z2 gauge | D ≥ 2     | Link checkerboard  | CPU, GPU |

Note that memory use grows quickly with the dimension: the lattice stores about
`48 * D * (D - 1)` bytes of precomputed geometry per site. In 16 GB that allows a lattice of size 360³ sites in three dimensions, but only 5¹⁰ sites in ten.

Refer to [`docs/`](docs/) for descriptions of used algorithms.

## Usage

Model runs are configured via TOML. Runnable example code lives in [`examples/`](examples/).

To run the Ising model, call

```bash
cargo run --example ising -- examples/ising/run.toml
```

Similarly, run the q-state Potts model via

```bash
cargo run --example potts -- examples/potts/potts.toml
```

and the Z2 gauge model via

```bash
cargo run --example gauge -- examples/gauge/gauge.toml
```

The Potts example fixes the number of states in its own source rather than in the
config file, since it is a compile-time constant; `const Q` in
`examples/potts/potts.rs` names it.

## Documentation

Documentation describing the models and methods can be found in [`docs/`](docs/).

## Testing

Alongside the unit tests, end-to-end tests in [`tests/`](tests/) validate implementations using well-known results.

Run the whole suite with

```bash
cargo test
```

## AI assistance

`plaquette` is developed with AI assistance. The conventions that shape the code and docs are written down as rules under `.claude/rules/`, each a Markdown file scoped by a `paths:` glob to the files it applies to (for example, the README style guide in `.claude/rules/readme.md`).

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT
license](LICENSE-MIT) at your option.
