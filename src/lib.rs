//! A framework for Monte Carlo simulation of lattice models.
//!
//! A run is assembled from a few interchangeable pieces: a [`Lattice`] fixes the
//! geometry, an [`Action`] gives the energy of a [`Configuration`] on it, and an
//! [`Updater`] proposes and accepts changes. A [`Chain`] drives those pieces to
//! produce a stream of configurations, and [`measure`] turns each one into a
//! [`Sample`] of observables that [`reduce`] aggregates into estimates with
//! error bars.
//!
//! Three models run today, each by Metropolis or by a checkerboard sweep, and
//! each checkerboard on either the CPU or the GPU. The two colorings differ
//! because the variables do: the Ising and Potts models color a *site* by the
//! parity of its coordinate sum, while the Z2 gauge model colors a *link* by its
//! direction as well as its base site's parity, so that no two links updated
//! together share a plaquette. Each has a sampler — [`IsingSampler`],
//! [`GaugeSampler`], [`PottsSampler`] — that builds a thermalized chain from a
//! run configuration parsed from TOML, which is how the examples under
//! `examples/` are driven.
//!
//! Every layer is generic over the lattice dimension `D`, which each model needs
//! enough of to have the cell its energy scores: Ising and Potts sum over
//! nearest-neighbor bonds and need one, the gauge action sums over plaquettes
//! and needs two. The state count `Q` is generic the same way, and
//! [`Potts`](model::Potts) is the model that exercises it — its labels stand for
//! nothing, so the energy compares them instead of decoding them into values,
//! while the other two are pinned at the two states `±1` names.
//!
//! Both are compile-time parameters throughout, so a driver names the pair it is
//! built for rather than reading either from a file. A config file's `shape`
//! still has to agree with the dimension, and `check_dimension` reports one that
//! does not; nothing in a file names the state count at all. Dispatching over
//! either at runtime would mean instantiating the whole stack once per
//! combination, and with two parameters that is a grid rather than a list —
//! fixing both at compile time is also what the lattice-gauge codes do.
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
pub mod potts_config;
pub mod potts_gpu;
pub mod potts_sampler;
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
    Correlator, GaugeSample, PottsSample, Sample, WilsonRectangles, correlator, gauge_measure,
    measure, polyakov_loop, potts_correlator, potts_measure, wilson_rectangles,
};
pub use potts_config::PottsRunConfig;
pub use potts_gpu::GpuPottsChain;
pub use potts_sampler::{AnyPottsChain, PottsSampler};
pub use rng::{RandRng, Rng};
pub use state::State;
pub use statistics::{
    Derived, Estimate, MIN_EFFECTIVE_SAMPLES, binder_cumulant, creutz_ratio, reduce, specific_heat,
    susceptibility,
};
pub use updater::{AnyUpdater, LinkCheckerboard, Metropolis, SiteCheckerboard, Updater};
