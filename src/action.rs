//! Action: the energy seam the sampler is built around.
//!
//! [`Action`] is what the updater depends on: energies only, never variable
//! values. An implementor owns just its physics parameters, borrowing the
//! lattice and configuration per call, and keeps no running energy.
//! Value-semantic observables stay inherent methods on the concrete models
//! rather than trait methods, so the trait stays energy-only.
//!
//! [`BondAction`] sits beside it as a second, narrower seam: not every model has
//! one, and a model that does declares the two numbers a cluster update needs
//! rather than another way of pricing a move.
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

/// A model whose energy is a sum over nearest-neighbor site bonds of a
/// two-valued term, symmetric under relabeling the `Q` states — the shape the
/// Fortuin–Kasteleyn cluster construction needs.
///
/// Implemented alongside [`Action`] rather than as a supertrait of it: a cluster
/// updater already receives an `&impl Action` in
/// [`Updater::sweep`](crate::updater::Updater::sweep), so the supertrait would
/// add nothing, and naming `D` here would leave it uninferable at a constructor
/// that takes only the model — [`Potts`](crate::models::potts::Potts) implements
/// `Action<Q, D>` for *every* `D`, so
/// [`SwendsenWang::for_model`](crate::updater::SwendsenWang::for_model) could not
/// resolve one and every call site would need a turbofish. See
/// `docs/swendsen-wang.md`.
///
/// Both numbers are constant over a run, which is what lets an updater read them
/// once at construction rather than querying the model per sweep. An algorithm
/// whose model dependence evolved with the configuration — heat bath, say —
/// would need a different seam.
pub trait BondAction<const Q: usize> {
    /// The per-bond energy gap `E(disagree) − E(agree)`, in the units of
    /// [`Action::energy`]. The bond probability is `1 − exp(−β · gap)`.
    ///
    /// One number for the whole lattice, with no bond index: both models here
    /// have uniform couplings. A disordered model with per-bond couplings would
    /// want the index; adding it before one exists would be a guess at what it
    /// should be keyed by.
    fn bond_energy_gap(&self) -> f64;

    /// Whether the energy is invariant under permuting the `Q` labels — `false`
    /// whenever an external field or per-label offset is set.
    ///
    /// The predicate is equality of labels, which is the only agreement test
    /// this trait knows about. A clock or O(n) model whose bonds agree by angle
    /// rather than by name would need that generalized too.
    fn relabel_invariant(&self) -> bool;
}
