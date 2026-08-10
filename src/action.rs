//! Action: the energy seam the sampler is built around.
//!
//! [`Action`] is what the updater depends on: energies only, never variable
//! values. An implementor owns just its physics parameters, borrowing the
//! lattice and configuration per call, and keeps no running energy.
//! Value-semantic observables stay inherent methods on the concrete models
//! rather than trait methods, so the trait stays energy-only.
//!
//! The models themselves — what the state indices *mean*, and everything that
//! reads them — live under [`models`](crate::models); this module is only the
//! vocabulary they are written in, which is what lets the updater name the seam
//! without naming a model.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::state::State;

/// The energy functional the sampler is built around.
///
/// Generic over the state count `Q` and the lattice dimension `D`, so the
/// updater can name the seam without naming a specific model.
pub trait Action<const Q: usize, const D: usize> {
    /// The energy `H` of `config` on `lattice`, computed from scratch — a full
    /// lattice scan, not the hot path.
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<Q>) -> f64;

    /// The energy change `ΔE = H(after) − H(before)` of poking the variable at
    /// `var` to `proposed`, without mutating `config`.
    ///
    /// The index names a cell of whatever kind `config` sits on, which is the
    /// model's own business — a site for [`Ising`](crate::models::ising::Ising), a link for [`Z2Gauge`](crate::models::gauge::Z2Gauge) — so
    /// the parameter stays grade-neutral rather than promising a site.
    ///
    /// The sampler's hot path: it reads only the terms incident to `var`, so it
    /// is `O(1)` in the lattice size rather than a rescan. It equals
    /// `energy(after) − energy(before)` by construction — exactly when the
    /// couplings and sums are integer-valued, up to rounding otherwise — and is
    /// the more accurate side of that comparison.
    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
        var: usize,
        proposed: State<Q>,
    ) -> f64;
}
