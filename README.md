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

`plaquette` currently implements two models:

| Model    | Dimension | Implementation     | Device   |
| -------- | --------- | ------------------ | -------- |
| Ising    | 2D        | Metropolis         | CPU      |
| Ising    | 2D        | Site checkerboard  | CPU, GPU |
| Z2 gauge | 3D        | Metropolis         | CPU      |
| Z2 gauge | 3D        | Link checkerboard  | CPU, GPU |

Refer to [`docs/`](docs/) for descriptions of used algorithms.

## Usage

Model runs are configured via TOML. Runnable example code lives in [`examples/`](examples/).

To run the 2D Ising model, call

```bash
cargo run --example ising -- examples/ising/run.toml
```

Similarly, run the 3D Z2 gauge model via

```bash
cargo run --example gauge -- examples/gauge/gauge.toml
```

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
