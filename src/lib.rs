//! A framework for Monte Carlo simulation of lattice models.
//!
//! A run is assembled from a few interchangeable pieces: a [`Lattice`] fixes the
//! geometry, an [`Action`] gives the energy of a [`Configuration`] on it, and an
//! [`Updater`] proposes and accepts changes. A [`Chain`] drives those pieces to
//! produce a stream of configurations, and [`measure`] turns each one into a
//! [`Sample`] of observables that [`reduce`] aggregates into estimates with
//! error bars.
//!
//! Two models run today, each by Metropolis or by a checkerboard sweep, and each
//! checkerboard on either the CPU or the GPU. The two colorings differ because
//! the variables do: the two-dimensional Ising model colors a *site* by the
//! parity of its coordinate sum, while the three-dimensional Z2 gauge model
//! colors a *link* by its direction as well as its base site's parity, so that
//! no two links updated together share a plaquette. Each has a sampler —
//! [`IsingSampler`] and [`GaugeSampler`] — that
//! builds a thermalized chain from a run configuration parsed from TOML, which
//! is how the examples under `examples/` are driven.
//!
//! The algorithms and the physics they rest on are written up under `docs/`.

// Lints are configured in Cargo.toml's `[lints]` table, which covers the tests
// and examples too rather than the library target alone.

pub mod chain;
pub mod config;
pub mod configuration;
pub mod device;
pub mod gauge_config;
pub mod gauge_gpu;
pub mod gauge_sampler;
pub mod ising_config;
pub mod ising_gpu;
pub mod ising_sampler;
pub mod lattice;
pub mod model;
pub mod observables;
pub mod rng;
pub mod state;
pub mod statistics;
pub mod updater;

pub use chain::Chain;
pub use config::{ConfigError, Start, UpdaterKind};
pub use configuration::{Cell, Configuration};
pub use device::Gpu;
pub use gauge_config::GaugeRunConfig;
pub use gauge_gpu::GpuGaugeChain;
pub use gauge_sampler::{AnyGaugeChain, GaugeSampler};
pub use ising_config::IsingRunConfig;
pub use ising_gpu::GpuIsingChain;
pub use ising_sampler::{AnyIsingChain, IsingSampler};
pub use lattice::{Lattice, Loop, Sign};
pub use model::{Action, AnyAction};
pub use observables::{
    Correlator, GaugeSample, Sample, WilsonRectangles, correlator, gauge_measure, measure,
    polyakov_loop, wilson_rectangles,
};
pub use rng::{RandRng, Rng};
pub use state::State;
pub use statistics::{
    Derived, Estimate, MIN_EFFECTIVE_SAMPLES, binder_cumulant, creutz_ratio, reduce, specific_heat,
    susceptibility,
};
pub use updater::{AnyUpdater, LinkCheckerboard, Metropolis, SiteCheckerboard, Updater};
