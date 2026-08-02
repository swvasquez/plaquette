//! A framework for Monte Carlo simulation of lattice models.
//!
//! A run is assembled from a few interchangeable pieces: a [`Lattice`] fixes the
//! geometry, an [`Action`] gives the energy of a [`Configuration`] on it, and an
//! [`Updater`] proposes and accepts changes. A [`Chain`] drives those pieces to
//! produce a stream of configurations, and [`measure`] turns each one into a
//! [`Sample`] of observables that [`reduce`] aggregates into estimates with
//! error bars.
//!
//! Two models run today. The two-dimensional Ising model is sampled by
//! single-spin Metropolis or by a checkerboard sweep, the latter on either the
//! CPU or the GPU; the three-dimensional Z2 gauge model is sampled by Metropolis
//! over links. Each has a sampler — [`IsingSampler`] and [`GaugeSampler`] — that
//! builds a thermalized chain from a run configuration parsed from TOML, which
//! is how the examples under `examples/` are driven.
//!
//! The algorithms and the physics they rest on are written up under `docs/`.

#![deny(missing_docs)]

pub mod chain;
pub mod config;
pub mod configuration;
pub mod gauge_config;
pub mod gauge_sampler;
pub mod gpu;
pub mod ising_config;
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
pub use gauge_config::GaugeRunConfig;
pub use gauge_sampler::GaugeSampler;
pub use gpu::{Gpu, GpuChain};
pub use ising_config::IsingRunConfig;
pub use ising_sampler::{AnyChain, IsingSampler};
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
pub use updater::{AnyUpdater, Checkerboard, Metropolis, Updater};
