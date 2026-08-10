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
| Ising    | D ≥ 1     | Swendsen–Wang      | CPU, GPU |
| Potts    | D ≥ 1     | Metropolis         | CPU      |
| Potts    | D ≥ 1     | Site checkerboard  | CPU, GPU |
| Potts    | D ≥ 1     | Swendsen–Wang      | CPU, GPU |
| Z2 gauge | D ≥ 2     | Metropolis         | CPU      |
| Z2 gauge | D ≥ 2     | Link checkerboard  | CPU, GPU |

Note that memory use grows quickly with the dimension: the lattice stores about
`48 * D * (D - 1)` bytes of precomputed geometry per site. In 16 GB that allows a lattice of size 360³ sites in three dimensions, but only 5¹⁰ sites in ten.

Refer to [`docs/`](docs/) for descriptions of used algorithms.

## Architecture

A simulation generates near-independent snapshots of the system, measures
observables on each, and averages those measurements into a result with an
error bar. The snapshots come from evolving the system through sweeps of the
lattice. A local update makes as many attempts per sweep as the lattice has
variables, and accepts each one with a probability set by the energy change it
would cause. A cluster update instead changes a whole connected region in one
move and always accepts it. Consecutive states are still strongly correlated, so
snapshots are taken many sweeps apart, leaving each one close to independent of
the last.

### Objects

```
  the driver -- the code running the simulation
               |
               | asks for the next snapshot
               v
  +------- Chain --------+                +--------- Rng ----------+
  |  run a fixed number  |                |  random values for     |
  |  of sweeps, then     |                |  the proposal and      |
  |  yield a snapshot    |                |  the accept test       |
  +----------------------+                +------------------------+
               |                                      |
               | drives each sweep                    |
               |     +--------- random draws ---------+
               v     v
  +------ Updater -------+                +-------- Action --------+
  |  pick a variable,    |   a proposed   |  prices a move:        |
  |  propose a new value | ---- move ---> |  the energy change     |
  |  accept or reject    | <- its cost -- |  it would cause        |
  +----------------------+                +------------------------+
        |                                              ^
        | writes the                        reads the  |
        | new value                     current state  |
        v                                              |
  +--------------------------- the system ---------------------------+
  |                                                                  |
  |  +--- Configuration ----+            +------ Lattice -------+    |
  |  |  one State per cell  | -- on a -> |  shape, neighbors,   |    |
  |  |  (a site or a link)  |            |  links, plaquettes   |    |
  |  +----------------------+            +----------------------+    |
  |                                                                  |
  +------------------------------------------------------------------+
```

### Snapshot

The driver — an example program, or a model's sampler — has already thermalized
the system.

1. The driver asks `Chain` for the next snapshot.
2. `Chain` runs a fixed number of sweeps before handing one back, and each sweep
   is a single call into `Updater`.
3. `Updater` begins the sweep with a single update, picking a variable and
   asking `Rng` for a proposed new value.
4. `Action` prices that move, reading the geometry from `Lattice` and the
   affected values from `Configuration`, and returns the energy change it would
   cause.
5. `Updater` accepts a downhill move outright and an uphill one only with a
   probability set by that change and the temperature, writing an accepted value
   into `Configuration` and leaving it untouched on a rejection.
6. One sweep is as many of those updates as `Configuration` has variables.
7. After the last sweep, `Chain` hands back a copy of `Configuration` as it
   stands, and that copy is the snapshot.
8. The driver measures observables on it, producing one `Sample`.

After many snapshots, `statistics` averages the collected observables into an
`Estimate` with an error bar.

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
