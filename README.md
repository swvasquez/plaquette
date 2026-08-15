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

`plaquette` currently implements several models, each with multiple update
implementations across the CPU and GPU:

| Model        | Dimension | Implementation             | Device   |
| ------------ | --------- | -------------------------- | -------- |
| Ising, Potts | D ≥ 1     | Metropolis (random)        | CPU      |
| Ising, Potts | D ≥ 1     | Metropolis (checkerboard)  | CPU, GPU |
| Ising, Potts | D ≥ 1     | Heat bath (random)         | CPU      |
| Ising, Potts | D ≥ 1     | Heat bath (checkerboard)   | CPU, GPU |
| Ising, Potts | D ≥ 1     | Swendsen–Wang              | CPU, GPU |
| Ising, Potts | D ≥ 1     | Wolff                      | CPU, GPU |
| Z2 gauge     | D ≥ 2     | Metropolis (random)        | CPU      |
| Z2 gauge     | D ≥ 2     | Metropolis (checkerboard)  | CPU, GPU |
| Z2 gauge     | D ≥ 2     | Heat bath (random)         | CPU      |
| Z2 gauge     | D ≥ 2     | Heat bath (checkerboard)   | CPU, GPU |

Note that memory use grows quickly with the dimension: the lattice stores about
`48 * D * (D - 1)` bytes of precomputed geometry per site. In 16 GB that allows a lattice of size 360³ sites in three dimensions, but only 5¹⁰ sites in ten.

Refer to [`docs/`](docs/) for descriptions of used algorithms.

## Architecture

A simulation generates near-independent snapshots of the system, measures
observables on each, and averages those measurements into a result with an
error bar. The snapshots come from evolving the system through sweeps of the
lattice. A local update makes as many single-variable moves per sweep as the
lattice has variables, weighing each outcome by the energy change it would
cause. A cluster update instead changes a whole connected region in one move
and always accepts it. Consecutive states are still strongly correlated, so
snapshots are taken many sweeps apart, leaving each one close to independent of
the last.

### Objects

```
  the driver -- the code running the simulation
               |
               | asks for the next snapshot
               v
  +------- Chain --------+                +--------- Rng ----------+
  |  run a fixed number  |                |  random values for the |
  |  of sweeps, then     |                |  schedule's picks and  |
  |  yield a snapshot    |                |  the kernel's draws    |
  +----------------------+                +------------------------+
               |                                      |
               | drives each sweep                    |
               |     +--------- random draws ---------+
               v     v
  +--------------- Updater ----------------+
  |                                        |
  |  +---------- Schedule -----------+     |
  |  |  pick which variable is next  |     |
  |  |  (random or checkerboard)     |     |
  |  +--------------|----------------+     |
  |                 | that variable        |
  |                 v                      |
  |  +----------- Kernel ------------+     |
  |  |  choose the variable's new    |     |
  |  |  value (Metropolis or heat    |     |
  |  |  bath)                        |     |
  |  +-------------------------------+     |
  +------|----------------|-------------|--+
         |                | candidate   ^
         |                | moves       | their
         |                v             | costs
         |            +-------- Action --------+
  writes |            |  prices a move:        |
  the    |            |  the energy change     |
  new    |            |  it would cause        |
  value  |            +----------|-------------+
         |                       ^
         |                       | reads the
         |                       | current state
         v                       |
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
the system. The walk-through follows a local update; a cluster update keeps the
`Updater` box's outer arrows but replaces its interior with a single collective
move (`ClusterUpdate`, described in `docs/swendsen-wang.md` and `docs/wolff.md`).

1. The driver asks `Chain` for the next snapshot.
2. `Chain` runs a fixed number of sweeps before handing one back, and each sweep
   is a single call into `Updater`.
3. `Updater` begins the sweep with a single update. Its `Schedule` picks which
   variable is next — drawn from `Rng` on the random schedule, taken in fixed
   color order on the checkerboard — and hands it to the `Kernel`. The schedule
   reads only the system's shape (the variable count, and the coloring from
   `Lattice`), never the values in `Configuration`.
4. The `Kernel` forms its candidate moves — Metropolis proposes one alternative
   value (drawn from `Rng` when there is more than one), the heat bath considers
   every value the variable could take — and asks `Action` to price them.
   `Action` reads the geometry from `Lattice` and the affected values from
   `Configuration`, and returns the energy change each move would cause.
5. The `Kernel` then sets the variable: Metropolis accepts a downhill move
   outright and an uphill one only with a probability set by its cost and the
   temperature, leaving the variable untouched on a rejection; the heat bath
   draws one of its candidates with the matching Boltzmann weight. The deciding
   draw comes from `Rng`, and an accepted or drawn value is written into
   `Configuration`.
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

Claude Code can be run against this repository inside a Docker Sandboxes microVM, isolated from the host and with the local network denied. [`.sbx/kit/spec.yaml`](.sbx/kit/spec.yaml) describes the sandbox and the [`justfile`](justfile) drives it. Claude Code is configured in [`.sbx/kit/settings.json`](.sbx/kit/settings.json). To use the sandbox, install `just` and `sbx`.

| Command | Description |
| --- | --- |
| `just sbx-up` | Build and start the sandbox, replacing any existing one |
| `just sbx-login` | Sign in to Claude Code inside the sandbox |
| `just sbx-agent` | Attach Claude Code to the running sandbox |
| `just sbx-shell` | Open a login shell in the running sandbox |

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or [MIT
license](LICENSE-MIT) at your option.
