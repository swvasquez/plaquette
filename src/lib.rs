//! A framework for Monte Carlo simulation of lattice models.
//!
//! A run is assembled from a few interchangeable pieces: a [`Lattice`] fixes the
//! geometry, an [`Action`] gives the energy of a [`Configuration`] on it, and an
//! [`Updater`] proposes and accepts changes. A [`Chain`] drives those pieces to
//! produce a stream of configurations, and a per-model measure function turns
//! each one into a record of observables that [`reduce`] aggregates into
//! estimates with error bars.
//!
//! Those seams — and the geometry, statistics, and device plumbing they rest on
//! — live in the root modules and are re-exported here. The models live under
//! [`models`], one submodule each, and are reached by their own paths:
//! [`models::ising`], [`models::potts`], and [`models::gauge`] each hold the
//! action, its observables, a run-config schema, and the samplers that drive
//! it. Nothing at the root depends on anything under [`models`]; the models
//! depend only on the root.
//!
//! Three models run today, each by Metropolis or by a checkerboard sweep, and
//! each checkerboard on either the CPU or the GPU. The two colorings differ
//! because the variables do: the Ising and Potts models color a *site* by the
//! parity of its coordinate sum, while the Z2 gauge model colors a *link* by its
//! direction as well as its base site's parity, so that no two links updated
//! together share a plaquette. Each has a sampler —
//! [`IsingSampler`](models::ising::IsingSampler),
//! [`GaugeSampler`](models::gauge::GaugeSampler),
//! [`PottsSampler`](models::potts::PottsSampler) — that builds a thermalized
//! chain from a run configuration parsed from TOML, which is how the examples
//! under `examples/` are driven.
//!
//! Every layer is generic over the lattice dimension `D`, which each model needs
//! enough of to have the cell its energy scores: Ising and Potts sum over
//! nearest-neighbor bonds and need one, the gauge action sums over plaquettes
//! and needs two. The state count `Q` is generic the same way, and
//! [`Potts`](models::potts::Potts) is the model that exercises it — its labels
//! stand for nothing, so the energy compares them instead of decoding them into
//! values, while the other two are pinned at the two states `±1` names.
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

pub mod action;
pub mod chain;
pub mod config;
pub mod configuration;
pub mod device;
pub mod lattice;
pub mod models;
pub mod observables;
pub mod rng;
pub mod state;
pub mod statistics;
pub mod updater;

pub use action::Action;
pub use chain::Chain;
pub use config::{ConfigError, Start, UpdaterKind};
pub use configuration::{Cell, Configuration};
pub use device::Gpu;
pub use lattice::{Lattice, Loop, Sign};
pub use observables::Correlator;
pub use rng::{RandRng, Rng};
pub use state::State;
pub use statistics::{
    Derived, Estimate, MIN_EFFECTIVE_SAMPLES, binder_cumulant, creutz_ratio, reduce, specific_heat,
    susceptibility,
};
pub use updater::{AnyUpdater, LinkCheckerboard, Metropolis, SiteCheckerboard, Updater};
