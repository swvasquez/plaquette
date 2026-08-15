//! Updater: the rule that advances the Markov chain one sweep at a time.
//!
//! Where the [`Action`] says what a configuration costs, the updater says how
//! the chain moves. It is the seam the driver depends on: a chain calls
//! [`sweep`](Updater::sweep) without naming an algorithm. What differs between
//! algorithms is the *schedule* — which variables to update, in what order —
//! and that lives in each updater's `sweep`; the single-variable accept/reject
//! kernel `step` is a plain module function the schedules that have one call
//! into. The algorithm itself — the acceptance rule, why the proposal must be
//! symmetric, and why the checkerboard reorderings still sample the Boltzmann
//! distribution — is derived in `docs/metropolis.md`.
//!
//! [`HeatBath`] is the second kernel. Rather than proposing one alternative and
//! accepting or rejecting it, it prices every state the variable could take and
//! draws one from the conditional distribution they define, so the update lands
//! somewhere every time. It reaches the same Boltzmann distribution and asks
//! nothing of a model that [`Action::energy_delta`] does not already give, which
//! is what lets one kernel serve every model here, on links as readily as on
//! sites. `docs/heat-bath.md` describes it.
//!
//! [`SwendsenWang`] is the one updater here that is not a Metropolis schedule at
//! all. It builds its own set of variables stochastically, changes all of them
//! at once, and is accepted with probability one, so it shares no kernel with
//! the others — only the seam. `docs/swendsen-wang.md` derives it.
//!
//! The updater holds no chain state: the [`Configuration`] *is* the current
//! state, mutated in place, and `β` is passed per call so one updater serves a
//! whole temperature scan. It keeps no running energy either;
//! [`sweep`](Updater::sweep) returns the net realized ΔE, and re-anchoring
//! against a from-scratch [`Action::energy`] stays the driver's job.

use crate::action::{Action, BondAction};
use crate::cluster;
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::rng::Rng;
use crate::state::State;

/// The rule that advances the Markov chain by one sweep.
///
/// Generic over the field's state count `Q` and the lattice dimension `D`, so
/// the driver can name the seam without naming a specific algorithm.
pub trait Updater<const Q: usize, const D: usize> {
    /// Advance `config` in place by one sweep — the conventional unit of Monte
    /// Carlo time, sized to the lattice — returning the net realized `ΔE`
    /// summed over its updates. What one sweep does is the algorithm's own
    /// choice.
    ///
    /// The returned sum telescopes to `H(after) − H(before)` for this sweep
    /// alone. It is not a running energy — re-anchoring against a from-scratch
    /// [`Action::energy`] stays the driver's job.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64;
}

/// The shared single-variable Metropolis kernel, at a variable the schedule
/// hands in: propose a different state, price it with
/// [`Action::energy_delta`], and accept with `min(1, e^{-β ΔE})`, returning the
/// realized `ΔE` (`0.0` on rejection, leaving `config` unchanged).
///
/// `var` is a bare index into the configuration, so it names a site on a site
/// field and a link on a link field; that grade-neutrality is what lets
/// [`LinkCheckerboard`] reuse this unchanged.
fn step<const Q: usize, const D: usize>(
    config: &mut Configuration<Q>,
    lattice: &Lattice<D>,
    action: &impl Action<Q, D>,
    var: usize,
    beta: f64,
    rng: &mut impl Rng,
) -> f64 {
    let proposed = propose(config.peek(var), rng);
    let delta = action.energy_delta(lattice, config, var, proposed);

    // The `ΔE ≤ 0` short-circuit keeps downhill moves without a draw and, by
    // keeping the argument to `exp` non-positive, prevents overflow.
    if delta <= 0.0 || rng.next_f64() < (-beta * delta).exp() {
        config.poke(var, proposed);
        delta
    } else {
        0.0
    }
}

/// The shared single-variable **heat bath** kernel, at a variable the schedule
/// hands in: price all `Q` states the variable could take, then draw one from
/// the conditional distribution they define, returning the realized `ΔE`
/// (`0.0` when the draw lands back on the state it started from).
///
/// Where [`step`] proposes a single alternative and may reject it, this one
/// lands somewhere every time, because the draw *is* the conditional and there
/// is nothing left to accept against. It reads only [`Action::energy_delta`],
/// so like [`step`] it is grade-neutral — `var` names a site on a site field
/// and a link on a link field — and it carries no requirement on the model
/// beyond `Q ≥ 2`. See `docs/heat-bath.md`.
fn heat_bath_step<const Q: usize, const D: usize>(
    config: &mut Configuration<Q>,
    lattice: &Lattice<D>,
    action: &impl Action<Q, D>,
    var: usize,
    beta: f64,
    rng: &mut impl Rng,
) -> f64 {
    debug_assert!(Q >= 2, "a heat bath draw needs a state to choose between");

    let current = config.peek(var);

    // Every candidate priced against the *current* state, which is the common
    // factor the conditional is defined up to: the terms of `H` that do not
    // touch `var` cancel from the ratio, and what is left is one `ΔE` per state
    // with the current state's own entry at zero. That entry is written as zero
    // rather than asked for, which saves a call and does not rest on every
    // model short-circuiting a poke to the state already there.
    let deltas: [f64; Q] = std::array::from_fn(|index| {
        let candidate = State::new(index).expect("index < Q");
        if candidate == current {
            0.0
        } else {
            action.energy_delta(lattice, config, var, candidate)
        }
    });

    // Shifting by the smallest delta puts the largest exponent at zero, so the
    // biggest weight is exactly one and the rest fall away below it. Without the
    // shift a strongly downhill candidate at large β overflows `exp` while the
    // current state sits at weight one, and the walk below would be comparing
    // against an infinity.
    let floor = deltas.iter().copied().fold(f64::INFINITY, f64::min);
    let mut weights = [0.0; Q];
    let mut total = 0.0;
    for (weight, &delta) in weights.iter_mut().zip(deltas.iter()) {
        *weight = (-beta * (delta - floor)).exp();
        total += *weight;
    }

    // One uniform per variable whatever the outcome, walked against the
    // unnormalized cumulative sum so the weights never have to be divided
    // through. The last state is the structural fallback rather than a case:
    // it takes every target the earlier states do not claim, so rounding in
    // `total` cannot leave the walk without an answer.
    let target = rng.next_f64() * total;
    let mut chosen = Q - 1;
    let mut cumulative = 0.0;
    for (index, &weight) in weights.iter().enumerate().take(Q - 1) {
        cumulative += weight;
        if target < cumulative {
            chosen = index;
            break;
        }
    }

    config.poke(var, State::new(chosen).expect("chosen < Q"));
    deltas[chosen]
}

/// The single-variable Metropolis update with a **random-site** schedule: a
/// stateless unit struct, implementing [`Updater`] for any `Q` in any dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metropolis;

impl<const Q: usize, const D: usize> Updater<Q, D> for Metropolis {
    /// `n_vars` single-site `step`s at uniformly-random sites.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        let mut net = 0.0;
        for _ in 0..config.n_vars() {
            let site = rng.next_below(config.n_vars());
            net += step(config, lattice, action, site, beta, rng);
        }
        net
    }
}

/// The single-variable heat bath update with a **random-variable** schedule: a
/// stateless unit struct, implementing [`Updater`] for any `Q` in any dimension.
///
/// Deliberately the same schedule as [`Metropolis`], so that the two differ in
/// the kernel they call and in nothing else. It states no model requirement of
/// its own — no relabeling symmetry the way [`SwendsenWang`] needs one, and no
/// cell kind — because the conditional it draws from is built out of
/// [`Action::energy_delta`] and nothing else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeatBath;

impl<const Q: usize, const D: usize> Updater<Q, D> for HeatBath {
    /// `n_vars` single-variable heat bath draws at uniformly-random variables —
    /// the same schedule a [`Metropolis`] sweep walks, and the same count.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below two, since the draw needs states to choose
    /// between.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        assert!(
            Q >= 2,
            "the heat bath draws a state from among the Q available, which needs \
             at least two"
        );

        let mut net = 0.0;
        for _ in 0..config.n_vars() {
            let var = rng.next_below(config.n_vars());
            net += heat_bath_step(config, lattice, action, var, beta, rng);
        }
        net
    }
}

/// The single-variable Metropolis update with a **checkerboard** schedule: a
/// stateless unit struct, implementing [`Updater`] for any `Q` in any dimension.
///
/// A sweep updates every site of one color, then every site of the other, where
/// a site's color is the parity of its coordinate sum. On CPU that is just
/// [`Metropolis`] in a fixed order; its purpose is to be the sequential
/// reference for a parallel (GPU) sweep, where a whole color updates at once.
/// That parallel independence needs every extent even — an odd extent wraps two
/// same-color sites next to each other — but run sequentially any order is a
/// valid Metropolis schedule, so the CPU version is correct on any lattice.
/// See `docs/metropolis.md` for the argument.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SiteCheckerboard;

impl<const Q: usize, const D: usize> Updater<Q, D> for SiteCheckerboard {
    /// Two color passes, together attempting one `step` per site — the same
    /// `n_vars` updates a [`Metropolis`] sweep does, in a fixed order.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        let mut net = 0.0;
        for color in [0, 1] {
            for site in 0..config.n_vars() {
                if lattice.site_parity(site) == color {
                    net += step(config, lattice, action, site, beta, rng);
                }
            }
        }
        net
    }
}

/// The single-link Metropolis update with a **checkerboard** schedule for a
/// gauge model: a stateless unit struct, implementing [`Updater`] for any `Q` in
/// any dimension.
///
/// A sweep colors each link by the pair `(direction, parity of the base site's
/// coordinate sum)` — `2D` colors, each updated fully before the next — so
/// that no two links of one color share a plaquette. Its purpose is to be the
/// sequential reference for a parallel sweep, where a whole color updates at
/// once. That independence needs every extent even; run sequentially any link
/// order is a valid Metropolis schedule, so the CPU version is correct on any
/// lattice. See `docs/metropolis.md` for why base-site parity alone cannot
/// separate a plaquette's links and for the full argument.
///
/// It is a distinct type rather than a mode of [`SiteCheckerboard`] because
/// only the name is shared: the color rule, what a collision means, and the
/// pass count all differ.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinkCheckerboard;

impl LinkCheckerboard {
    /// Colors in one sweep: a direction paired with a base-site parity.
    ///
    /// Shared rather than derived twice — `GpuGaugeChain` turns it into
    /// dispatches per sweep and has to agree with the order
    /// [`sweep`](LinkCheckerboard::sweep) walks, since the CPU schedule is the
    /// reference the device kernel is checked against.
    pub(crate) const fn colors<const D: usize>() -> usize {
        2 * D
    }
}

impl<const Q: usize, const D: usize> Updater<Q, D> for LinkCheckerboard {
    /// `2D` color passes — each direction in turn, even base sites then odd —
    /// together attempting one `step` per link.
    ///
    /// # Panics
    ///
    /// Panics if `config` is not a link field, since the schedule reads each
    /// variable's index as a link.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        assert_eq!(
            config.cell(),
            Cell::Link,
            "the link checkerboard schedules links, so the configuration must be a link field"
        );

        let mut net = 0.0;
        for dir in 0..D {
            for color in [0, 1] {
                // Iterate over *sites* and address the one link each owns in
                // this direction, rather than scanning all `D * n_sites` links
                // and skipping other directions. The visiting order is the same
                // either way — `link_colors_are_visited_in_link_order` pins
                // that — and it is the mapping the GPU kernel launches one
                // thread per.
                for site in 0..lattice.n_sites() {
                    if lattice.site_parity(site) == color {
                        let link = lattice.site_link(site, dir);
                        net += step(config, lattice, action, link, beta, rng);
                    }
                }
            }
        }
        net
    }
}

/// The single-variable heat bath under a **site checkerboard** schedule: a
/// stateless unit struct, implementing [`Updater`] for any `Q` in any dimension.
///
/// The kernel-swapped twin of [`SiteCheckerboard`], walking the same colors in
/// the same order. Its purpose is the same too, and is the reason it exists at
/// all: it is the sequential reference for the parallel heat bath sweep the
/// device runs, so `GpuSiteCheckerboardHeatBath` has a host counterpart to be checked
/// against rather than only a Metropolis run at the same temperature.
///
/// The schedule is duplicated from [`SiteCheckerboard`] rather than shared,
/// because the color walk and the kernel call are fused in both. Factoring the
/// kernel out of the schedules would collapse this type and its link twin into
/// the Metropolis ones; that is a change to the seam and is deliberately not
/// made here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SiteCheckerboardHeatBath;

impl<const Q: usize, const D: usize> Updater<Q, D> for SiteCheckerboardHeatBath {
    /// Two color passes, together drawing once per site — the same schedule
    /// [`SiteCheckerboard`] walks, with `heat_bath_step` in place of `step`.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below two.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        assert!(
            Q >= 2,
            "the heat bath draws a state from among the Q available, which needs \
             at least two"
        );

        let mut net = 0.0;
        for color in [0, 1] {
            for site in 0..config.n_vars() {
                if lattice.site_parity(site) == color {
                    net += heat_bath_step(config, lattice, action, site, beta, rng);
                }
            }
        }
        net
    }
}

/// The single-variable heat bath under a **link checkerboard** schedule: a
/// stateless unit struct, implementing [`Updater`] for any `Q` in any dimension.
///
/// The kernel-swapped twin of [`LinkCheckerboard`], walking the same `2D` colors
/// in the same order, and the sequential reference for the device's
/// `GpuLinkCheckerboardHeatBath` for the reason [`SiteCheckerboardHeatBath`] gives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinkCheckerboardHeatBath;

impl<const Q: usize, const D: usize> Updater<Q, D> for LinkCheckerboardHeatBath {
    /// `2D` color passes — each direction in turn, even base sites then odd —
    /// together drawing once per link.
    ///
    /// # Panics
    ///
    /// Panics if `config` is not a link field, or if `Q` is below two.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        assert_eq!(
            config.cell(),
            Cell::Link,
            "the link checkerboard schedules links, so the configuration must be a link field"
        );
        assert!(
            Q >= 2,
            "the heat bath draws a state from among the Q available, which needs \
             at least two"
        );

        let mut net = 0.0;
        for dir in 0..D {
            for color in [0, 1] {
                for site in 0..lattice.n_sites() {
                    if lattice.site_parity(site) == color {
                        let link = lattice.site_link(site, dir);
                        net += heat_bath_step(config, lattice, action, link, beta, rng);
                    }
                }
            }
        }
        net
    }
}

/// The Swendsen–Wang cluster update: bond neighbors that agree, then give each
/// resulting cluster a fresh label.
///
/// Unlike the other updaters this one carries state — the model's bond gap, read
/// once at construction — so it is built with [`for_model`](SwendsenWang::for_model)
/// rather than named directly. Reading it once is what keeps [`Updater`] a
/// uniform capability rather than a relation between an updater and an action:
/// Swendsen–Wang's whole dependence on the model is two numbers that do not move
/// over a run. See `docs/swendsen-wang.md` for the algorithm and for why the
/// move is accepted unconditionally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwendsenWang {
    /// The model's `E(disagree) − E(agree)` per bond, from
    /// [`BondAction::bond_energy_gap`].
    bond_gap: f64,
}

impl SwendsenWang {
    /// Build a cluster updater for `model`, capturing its bond gap.
    ///
    /// # Panics
    ///
    /// Panics if the model is not invariant under relabeling — a per-label
    /// offset or an external field breaks the symmetry the cluster move relies
    /// on, and the update would sample the wrong distribution rather than fail.
    /// Panics too if the gap is negative or not finite: a negative gap makes the
    /// bond probability negative, so no bond would ever open and the chain would
    /// quietly sample the infinite-temperature model. Both are the loud
    /// counterparts of load-time rules in the run configs. See
    /// `docs/swendsen-wang.md`.
    pub fn for_model<const Q: usize, M: BondAction<Q>>(model: &M) -> Self {
        assert!(
            model.relabel_invariant(),
            "the cluster move relabels a whole cluster at once, which is only \
             weight-preserving when the energy is invariant under relabeling; \
             this model has a per-label offset or an external field set"
        );
        let bond_gap = model.bond_energy_gap();
        assert!(
            bond_gap.is_finite() && bond_gap >= 0.0,
            "the bond probability 1 - exp(-beta * gap) needs a finite, \
             non-negative gap, got {bond_gap}; an antiferromagnetic coupling is \
             frustrated rather than merely inverted and needs a different \
             construction"
        );
        SwendsenWang { bond_gap }
    }

    /// The probability `1 − exp(−β · gap)` that a bond between two *agreeing*
    /// sites is opened at inverse temperature `beta`.
    ///
    /// Public because the device backend needs the same number and must not
    /// arrive at it by its own arithmetic: a cluster chain that opened its bonds
    /// at a different rate would sample a different model while looking
    /// perfectly healthy. Going through here also means the device path inherits
    /// the construction guards rather than restating them.
    pub fn bond_probability(&self, beta: f64) -> f64 {
        1.0 - (-beta * self.bond_gap).exp()
    }
}

impl<const Q: usize, const D: usize> Updater<Q, D> for SwendsenWang {
    /// One cluster decomposition of the whole lattice, then one fresh label per
    /// cluster — a single move touching every site, not `n_vars` of them.
    ///
    /// The returned `ΔE` is a from-scratch difference rather than an
    /// accumulation, since a cluster move does not price itself one site at a
    /// time. That is two extra `O(D·V)` scans on top of a labeling pass that is
    /// already `Θ(D·V)` — a constant factor, paid to keep the seam uniform.
    ///
    /// # Panics
    ///
    /// Panics if `config` is not a site field, or if `Q` is below two.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        assert_eq!(
            config.cell(),
            Cell::Site,
            "the cluster update bonds nearest-neighbor sites, so the \
             configuration must be a site field"
        );
        assert!(
            Q >= 2,
            "the cluster update redraws a label among the Q states, which needs \
             at least two"
        );

        let before = action.energy(lattice, config);
        let p = self.bond_probability(beta);

        // The short-circuit is part of the contract, not an optimization: a
        // uniform is drawn only for a pair that *agrees*, so the position in the
        // stream depends on the configuration. Drawing unconditionally would
        // sample the same distribution and put every existing run on a different
        // stream.
        let clusters = cluster::site_clusters(lattice, |i, j| {
            config.peek(i) == config.peek(j) && rng.next_f64() < p
        });

        // A *redraw*, not a flip: a cluster may come back on the label it had.
        // At Q = 2 that differs from the textbook "flip each cluster with
        // probability one half" only in bookkeeping, and both are correct.
        let fresh: Vec<State<Q>> = (0..clusters.n_clusters())
            .map(|_| State::new(rng.next_below(Q)).expect("next_below(Q) < Q"))
            .collect();
        for (site, &label) in clusters.labels().iter().enumerate() {
            config.poke(site, fresh[label]);
        }

        action.energy(lattice, config) - before
    }
}

/// A runtime choice among the built-in updaters, so an updater named in a config
/// file can be selected without the caller committing to a type at compile time.
///
/// Implements [`Updater`] by forwarding `sweep` to whichever updater it wraps.
/// Its variants mirror [`UpdaterKind`](crate::config::UpdaterKind) — a closed
/// set, which is what makes the choice recordable.
///
/// It is `PartialEq` but not `Eq`, because [`SwendsenWang`] carries a bond gap
/// and floats have no total equality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyUpdater {
    /// The random-site schedule, [`Metropolis`].
    Metropolis(Metropolis),
    /// The random-variable heat bath, [`HeatBath`].
    HeatBath(HeatBath),
    /// The site checkerboard heat bath, [`SiteCheckerboardHeatBath`].
    SiteCheckerboardHeatBath(SiteCheckerboardHeatBath),
    /// The link checkerboard heat bath, [`LinkCheckerboardHeatBath`].
    LinkCheckerboardHeatBath(LinkCheckerboardHeatBath),
    /// The site checkerboard schedule, [`SiteCheckerboard`].
    SiteCheckerboard(SiteCheckerboard),
    /// The link checkerboard schedule, [`LinkCheckerboard`].
    LinkCheckerboard(LinkCheckerboard),
    /// The cluster update, [`SwendsenWang`].
    SwendsenWang(SwendsenWang),
}

impl<const Q: usize, const D: usize> Updater<Q, D> for AnyUpdater {
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        match self {
            AnyUpdater::Metropolis(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::HeatBath(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::SiteCheckerboardHeatBath(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::LinkCheckerboardHeatBath(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::SiteCheckerboard(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::LinkCheckerboard(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::SwendsenWang(u) => u.sweep(config, lattice, action, beta, rng),
        }
    }
}

/// Propose a state other than `current`, drawn uniformly from the `Q - 1` that
/// are not it — the *symmetric* proposal the acceptance rule rests on when it
/// drops the Hastings ratio (see `docs/metropolis.md`). The draw is mapped onto
/// the alternatives by skipping past `current`, a bijection, so no state is
/// offered twice or not at all.
///
/// `Q = 2` takes the deterministic flip rather than the general path, and not
/// only to save a draw: consuming randomness for a determined outcome would
/// shift every existing two-state chain onto a different stream while sampling
/// exactly the same distribution.
fn propose<const Q: usize>(current: State<Q>, rng: &mut impl Rng) -> State<Q> {
    debug_assert!(Q >= 2, "a proposal needs a state to move to");
    let index = if Q == 2 {
        1 - current.index()
    } else {
        let draw = rng.next_below(Q - 1);
        draw + usize::from(draw >= current.index())
    };
    State::new(index).expect("skipping past the current index stays below Q")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::gauge::Z2Gauge;
    use crate::models::ising::Ising;
    use crate::models::potts::Potts;
    use crate::rng::RandRng;

    /// A scripted [`Rng`] handing back preset answers; the consumption counters
    /// double as an assertion target for how many draws were made.
    struct ScriptedRng {
        sites: Vec<usize>,
        uniforms: Vec<f64>,
        site_i: usize,
        unif_i: usize,
    }

    impl ScriptedRng {
        fn new(sites: Vec<usize>, uniforms: Vec<f64>) -> Self {
            ScriptedRng {
                sites,
                uniforms,
                site_i: 0,
                unif_i: 0,
            }
        }
    }

    impl Rng for ScriptedRng {
        fn next_f64(&mut self) -> f64 {
            let v = self.uniforms[self.unif_i];
            self.unif_i += 1;
            v
        }

        fn next_below(&mut self, n: usize) -> usize {
            let v = self.sites[self.site_i];
            assert!(v < n, "scripted site {v} out of range for n = {n}");
            self.site_i += 1;
            v
        }
    }

    /// At two states the proposal is the flip and consumes no randomness, so
    /// existing two-state chains keep their streams.
    #[test]
    fn the_two_state_proposal_is_the_flip_and_draws_nothing() {
        let mut rng = ScriptedRng::new(vec![], vec![]);
        for index in [0usize, 1] {
            let current = State::<2>::new(index).unwrap();
            assert_eq!(propose(current, &mut rng).index(), 1 - index);
        }
        assert_eq!(rng.site_i, 0, "the two-state proposal draws no index");
        assert_eq!(rng.unif_i, 0, "the two-state proposal draws no uniform");
    }

    /// Above two states the proposal never hands the current state back and
    /// reaches every alternative about equally often — the symmetry the
    /// acceptance rule assumes.
    #[test]
    fn the_general_proposal_covers_every_other_state_uniformly() {
        const Q: usize = 4;
        const DRAWS: usize = 4_000;

        let mut rng = RandRng::seed_from_u64(2026);
        for index in 0..Q {
            let current = State::<Q>::new(index).unwrap();
            let mut seen = [0usize; Q];
            for _ in 0..DRAWS {
                seen[propose(current, &mut rng).index()] += 1;
            }

            assert_eq!(seen[index], 0, "state {index} was proposed back to itself");
            for (other, &count) in seen.iter().enumerate() {
                // Three alternatives share 4000 draws, so each expects about
                // 1333 with a standard deviation near 30; this window is wide
                // enough to be seed-robust and narrow enough that a proposal
                // favoring or skipping a state could not pass it.
                if other != index {
                    assert!(
                        (1_150..1_520).contains(&count),
                        "state {index} -> {other}: {count} of {DRAWS}"
                    );
                }
            }
        }
    }

    /// A downhill flip is accepted with no accept/reject draw and returns a
    /// realized ΔE equal to the from-scratch difference.
    #[test]
    fn accepts_a_downhill_flip_without_drawing() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);

        // One spin flipped against an aligned background, so flipping it back
        // is downhill.
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        let site = 5;
        config.poke(site, State::new(1).unwrap());

        let proposed = propose(config.peek(site), &mut ScriptedRng::new(vec![], vec![]));
        let expected = action.energy_delta(&lat, &config, site, proposed);
        assert!(expected < 0.0, "setup should make the flip downhill");
        let before = action.energy(&lat, &config);

        let mut rng = ScriptedRng::new(vec![], vec![]);
        let realized = step(&mut config, &lat, &action, site, 1.0, &mut rng);

        assert_eq!(realized, expected);
        assert_eq!(config.peek(site), proposed);
        assert_eq!(realized, action.energy(&lat, &config) - before);
        assert_eq!(
            rng.unif_i, 0,
            "downhill move must not consume an accept draw"
        );
    }

    /// An uphill flip whose draw falls above the Boltzmann factor is rejected.
    #[test]
    fn rejects_an_uphill_flip_above_the_boltzmann_factor() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site); // ground state: every flip uphill
        let untouched = config.clone();

        // β = 1, ΔE = +8 ⇒ e^{-βΔE} ≈ 3.4e-4; a draw of 0.5 is far above it.
        let mut rng = ScriptedRng::new(vec![], vec![0.5]);
        let realized = step(&mut config, &lat, &action, 5, 1.0, &mut rng);

        assert_eq!(realized, 0.0);
        assert_eq!(config, untouched);
    }

    /// An uphill flip whose draw falls below the Boltzmann factor is accepted —
    /// the mandatory uphill move.
    #[test]
    fn accepts_an_uphill_flip_below_the_boltzmann_factor() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        let site = 5;

        let proposed = propose(config.peek(site), &mut ScriptedRng::new(vec![], vec![]));
        let expected = action.energy_delta(&lat, &config, site, proposed);
        assert!(expected > 0.0, "setup should make the flip uphill");
        let before = action.energy(&lat, &config);

        // β = 0.25, ΔE = +8 ⇒ e^{-βΔE} = e^{-2} ≈ 0.135; a draw of 0.1 is below it.
        let mut rng = ScriptedRng::new(vec![], vec![0.1]);
        let realized = step(&mut config, &lat, &action, site, 0.25, &mut rng);

        assert_eq!(realized, expected);
        assert_eq!(config.peek(site), proposed);
        assert_eq!(realized, action.energy(&lat, &config) - before);
    }

    /// A Metropolis sweep attempts exactly `n_vars` steps, each drawing one
    /// accept uniform on the all-uphill ground state.
    #[test]
    fn metropolis_sweep_runs_n_vars_steps() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site); // ground state: all flips uphill
        let untouched = config.clone();
        let n = config.n_vars();

        // β = 1 ⇒ e^{-βΔE} ≈ 3.4e-4; a 0.9 draw rejects every step.
        let mut rng = ScriptedRng::new((0..n).collect(), vec![0.9; n]);
        let net = Metropolis.sweep(&mut config, &lat, &action, 1.0, &mut rng);

        assert_eq!(net, 0.0);
        assert_eq!(config, untouched);
        assert_eq!(rng.site_i, n, "sweep must attempt exactly n_vars steps");
        assert_eq!(
            rng.unif_i, n,
            "each uphill step draws exactly one accept uniform"
        );
    }

    /// The net ΔE a Metropolis sweep returns equals `H(after) − H(before)` for
    /// any accept/reject pattern; integer-valued couplings and sums make the
    /// comparison bit-exact.
    #[test]
    fn metropolis_sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5); // exactly representable; sums stay integer
        let mut config = Configuration::<2>::hot(&lat, Cell::Site, &mut RandRng::seed_from_u64(7));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(99);
        let net = Metropolis.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// A checkerboard sweep attempts one update per site and never touches the
    /// random-site generator — the schedule is deterministic.
    #[test]
    fn site_checkerboard_sweep_attempts_every_site_once() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site); // ground state: all flips uphill
        let untouched = config.clone();
        let n = config.n_vars();

        // β = 1 ⇒ e^{-βΔE} ≈ 3.4e-4; a 0.9 draw rejects every attempt. No site
        // draws are scripted, so a stray next_below would panic.
        let mut rng = ScriptedRng::new(vec![], vec![0.9; n]);
        let net = SiteCheckerboard.sweep(&mut config, &lat, &action, 1.0, &mut rng);

        assert_eq!(net, 0.0);
        assert_eq!(config, untouched);
        assert_eq!(
            rng.unif_i, n,
            "checkerboard must attempt each of the n sites once"
        );
        assert_eq!(rng.site_i, 0, "checkerboard picks no random sites");
    }

    /// A checkerboard sweep satisfies the same telescoping identity as the
    /// Metropolis sweep.
    #[test]
    fn site_checkerboard_sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let mut config = Configuration::<2>::hot(&lat, Cell::Site, &mut RandRng::seed_from_u64(7));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(99);
        let net = SiteCheckerboard.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// Both site schedules run a three-state model and still account exactly.
    /// Nothing below `Q = 3` exercises the drawn proposal at all — the
    /// two-state path never touches the generator — so this is what says the
    /// drawn candidate is in range and the action can price a move to it.
    #[test]
    fn the_site_schedules_run_a_three_state_model() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);

        for (label, updater) in [
            ("metropolis", AnyUpdater::Metropolis(Metropolis)),
            (
                "site checkerboard",
                AnyUpdater::SiteCheckerboard(SiteCheckerboard),
            ),
        ] {
            let mut rng = RandRng::seed_from_u64(5);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
            let before = action.energy(&lat, &config);
            let net = updater.sweep(&mut config, &lat, &action, 0.6, &mut rng);

            assert_eq!(net, action.energy(&lat, &config) - before, "{label}");
            assert_ne!(
                net, 0.0,
                "{label}: a hot three-state field should accept something"
            );
        }
    }

    /// On an even lattice no site shares a color with any of its neighbors —
    /// the property a parallel sweep relies on.
    #[test]
    fn site_colors_separate_neighbors() {
        let lat = Lattice::new([4, 4]);
        for site in 0..lat.n_sites() {
            let c = lat.site_parity(site);
            for &nbr in lat.site_neighbors(site) {
                assert_ne!(c, lat.site_parity(nbr), "neighbors must differ in color");
            }
        }
    }

    /// A link's color under [`LinkCheckerboard`]: its direction paired with its
    /// base site's parity.
    fn link_color<const D: usize>(lattice: &Lattice<D>, link: usize) -> (usize, usize) {
        (
            lattice.link_direction(link),
            lattice.site_parity(lattice.link_site(link)),
        )
    }

    /// No two links of the same color ever share a plaquette — the property a
    /// parallel gauge sweep would rely on.
    #[test]
    fn link_colors_separate_plaquette_partners() {
        let lat = Lattice::new([4, 4, 4]);
        for link in 0..lat.n_links() {
            let color = link_color(&lat, link);
            for plaquette in lat.link_plaquettes(link) {
                for partner in lat.plaquette_links(plaquette) {
                    if partner != link {
                        assert_ne!(
                            color,
                            link_color(&lat, partner),
                            "links {link} and {partner} share plaquette {plaquette} and a color"
                        );
                    }
                }
            }
        }
    }

    /// The link coloring stays collision-free at every dimension, where a link
    /// gains plaquette partners linearly while the colors grow as `2D`. Every
    /// extent is even, which the periodic wrap requires.
    ///
    /// Only the link half is here: the site coloring is `Lattice::site_parity`,
    /// and `lattice::parity_alternates_between_neighbors_in_every_dimension`
    /// already sweeps it over a superset of these shapes.
    #[test]
    fn the_link_coloring_stays_collision_free_in_every_dimension() {
        fn links<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            for link in 0..lat.n_links() {
                let color = link_color(&lat, link);
                for plaquette in lat.link_plaquettes(link) {
                    for partner in lat.plaquette_links(plaquette) {
                        if partner != link {
                            assert_ne!(
                                color,
                                link_color(&lat, partner),
                                "{shape:?}: links {link} and {partner} share \
                                 plaquette {plaquette} and a color"
                            );
                        }
                    }
                }
            }
        }

        links([4, 6]);
        links([2, 4, 6]);
        links([2, 4, 2, 4]);
        links([2, 2, 4, 2, 2]);
        links([2, 4, 2, 2, 4, 2]);
    }

    /// A link pass covers every link exactly once, at every dimension — the
    /// counterpart of the collision-free property, and an independent failure
    /// mode. The site pass is not asserted: it partitions the sites by a
    /// two-valued function, so coverage is arithmetic rather than a property of
    /// the schedule.
    #[test]
    fn a_link_pass_covers_every_link_once() {
        fn links<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let mut seen: Vec<usize> = Vec::with_capacity(lat.n_links());
            // The pass order the sweep walks: direction outermost, parity inner.
            for dir in 0..D {
                for color in [0, 1] {
                    seen.extend(
                        (0..lat.n_sites())
                            .filter(|&site| lat.site_parity(site) == color)
                            .map(|site| lat.site_link(site, dir)),
                    );
                }
            }
            assert_eq!(seen.len(), lat.n_links(), "{shape:?}: wrong pass total");
            seen.sort_unstable();
            assert_eq!(seen, (0..lat.n_links()).collect::<Vec<_>>(), "{shape:?}");
        }

        links([4, 6]);
        links([2, 4, 6]);
        links([2, 4, 2, 4]);
        links([2, 4, 2, 2, 4, 2]);
    }

    /// The sweep's site-major iteration visits exactly the links a link-major
    /// scan would, in the same order — pinning them equal is what makes the
    /// site-major walk an optimization rather than a schedule change.
    #[test]
    fn link_colors_are_visited_in_link_order() {
        let lat = Lattice::new([4, 6, 4]);
        for dir in 0..3 {
            for color in [0, 1] {
                let site_major: Vec<usize> = (0..lat.n_sites())
                    .filter(|&site| lat.site_parity(site) == color)
                    .map(|site| lat.site_link(site, dir))
                    .collect();
                let link_major: Vec<usize> = (0..lat.n_links())
                    .filter(|&link| {
                        lat.link_direction(link) == dir
                            && lat.site_parity(lat.link_site(link)) == color
                    })
                    .collect();

                assert_eq!(site_major, link_major, "dir {dir}, color {color}");
                assert!(
                    site_major.windows(2).all(|w| w[0] < w[1]),
                    "the shared order should be ascending in link index"
                );
            }
        }
    }

    /// A link checkerboard sweep attempts one update per link and never touches
    /// the random-variable generator — the schedule is deterministic.
    #[test]
    fn link_checkerboard_sweep_attempts_every_link_once() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Link); // ground state: all flips uphill
        let untouched = config.clone();
        let n = config.n_vars();

        // Flipping one link flips its 2(D−1) = 4 plaquettes, so ΔE = +8 at j = 1;
        // at β = 1 that is e^{-8} ≈ 3.4e-4, and a 0.9 draw rejects every attempt.
        let mut rng = ScriptedRng::new(vec![], vec![0.9; n]);
        let net = LinkCheckerboard.sweep(&mut config, &lat, &action, 1.0, &mut rng);

        assert_eq!(net, 0.0);
        assert_eq!(config, untouched);
        assert_eq!(
            rng.unif_i, n,
            "the link checkerboard must attempt each of the n links once"
        );
        assert_eq!(
            rng.site_i, 0,
            "the link checkerboard picks nothing at random"
        );
    }

    /// The link checkerboard sweep satisfies the same telescoping identity as
    /// the site schedules, on a link field.
    #[test]
    fn link_checkerboard_sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0); // integer-valued, so the comparison is bit-exact
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut RandRng::seed_from_u64(7));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(99);
        let net = LinkCheckerboard.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// An odd extent is no obstacle to the sequential schedule: the coloring's
    /// independence fails there, but any sequential order is a valid Metropolis
    /// schedule, so the sweep still runs and still accounts correctly.
    #[test]
    fn link_checkerboard_runs_on_odd_extents() {
        let lat = Lattice::new([3, 5, 3]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut RandRng::seed_from_u64(3));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(11);
        let net = LinkCheckerboard.sweep(&mut config, &lat, &action, 0.5, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// A real generator with a tally of the draws taken through it — what a
    /// [`ScriptedRng`] cannot be for the cluster update, whose draw *count*
    /// depends on the configuration it is handed.
    struct Counting<R> {
        inner: R,
        uniforms: usize,
        below: usize,
    }

    impl<R: Rng> Counting<R> {
        fn new(inner: R) -> Self {
            Counting {
                inner,
                uniforms: 0,
                below: 0,
            }
        }
    }

    impl<R: Rng> Rng for Counting<R> {
        fn next_f64(&mut self) -> f64 {
            self.uniforms += 1;
            self.inner.next_f64()
        }

        fn next_below(&mut self, n: usize) -> usize {
            self.below += 1;
            self.inner.next_below(n)
        }
    }

    /// A model whose energy is not invariant under relabeling is refused at
    /// construction rather than sampled wrongly.
    #[test]
    #[should_panic(expected = "invariant under relabeling")]
    fn the_cluster_update_refuses_a_potts_model_with_offsets() {
        SwendsenWang::for_model(&Potts::<3>::new(1.0, [0.5, 0.0, 0.0]));
    }

    /// The same rule reached through the other implementor, whose symmetry is
    /// broken by a field rather than by a per-label offset.
    #[test]
    #[should_panic(expected = "invariant under relabeling")]
    fn the_cluster_update_refuses_an_ising_model_with_a_field() {
        SwendsenWang::for_model(&Ising::new(1.0, 0.25));
    }

    /// An antiferromagnetic coupling is refused too. Nothing would fail on its
    /// own: the bond probability would come out negative, no bond would open,
    /// and the chain would sample the infinite-temperature model while reporting
    /// the coupling it was given.
    #[test]
    #[should_panic(expected = "non-negative gap")]
    fn the_cluster_update_refuses_an_antiferromagnetic_coupling() {
        SwendsenWang::for_model(&Potts::<3>::symmetric(-1.0));
    }

    /// The two implementors' gaps differ by the factor of two their conventions
    /// differ by, which is what makes a Potts run at `2J` and an Ising run at
    /// `J` open their bonds with the same probability.
    #[test]
    fn the_two_conventions_give_the_same_bond_probability() {
        let j = 0.75;
        let potts = SwendsenWang::for_model(&Potts::<2>::symmetric(2.0 * j));
        let ising = SwendsenWang::for_model(&Ising::new(j, 0.0));
        assert_eq!(potts, ising);
        assert_eq!(potts.bond_gap, 2.0 * j);
    }

    /// At `beta = 0` no bond opens, so every site is its own cluster and gets an
    /// independent fresh label — the infinite-temperature limit, and the case
    /// where the cluster update degenerates to resampling the whole lattice.
    ///
    /// The cluster count is read off the generator: one label draw per cluster.
    #[test]
    fn at_zero_beta_every_site_is_its_own_cluster() {
        fn probe<const Q: usize, const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = SwendsenWang::for_model(&action);
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            let mut rng = Counting::new(RandRng::seed_from_u64(4));

            updater.sweep(&mut config, &lat, &action, 0.0, &mut rng);

            assert_eq!(
                rng.below,
                lat.n_sites(),
                "{shape:?} at Q = {Q}: one label draw per cluster, and at p = 0 \
                 there are as many clusters as sites"
            );
            // A cold start agrees on every bond, so each of the `D * n_sites`
            // bonds still costs its uniform even though none of them opens.
            assert_eq!(rng.uniforms, D * lat.n_sites(), "{shape:?} at Q = {Q}");
        }

        probe::<2, 2>([4, 6]);
        probe::<3, 2>([4, 6]);
        probe::<4, 2>([4, 6]);
        probe::<3, 3>([3, 4, 5]);
    }

    /// A uniform is drawn for an agreeing pair and for no other, which is what
    /// fixes where a cluster run sits in the generator's stream.
    ///
    /// The short-circuit in `sweep` is a contract rather than an optimization.
    /// Drawing unconditionally would sample exactly the same distribution — a
    /// disagreeing bond cannot open whatever the draw says — but it would shift
    /// every existing chain onto a different stream, so every recorded run would
    /// stop reproducing while every test of the physics kept passing. Nothing
    /// else here would notice, which is why the draw count is asserted against a
    /// number counted independently.
    #[test]
    fn a_uniform_is_drawn_for_an_agreeing_pair_and_no_other() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = SwendsenWang::for_model(&action);

        let mut setup = RandRng::seed_from_u64(20260811);
        let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut setup);

        // Count the agreeing bonds by the same forward-column walk, before the
        // sweep touches the configuration.
        let mut agreeing = 0usize;
        for site in 0..lat.n_sites() {
            for &partner in lat.site_neighbors(site).iter().step_by(2) {
                agreeing += usize::from(config.peek(site) == config.peek(partner));
            }
        }
        let bonds = 2 * lat.n_sites(); // D * n_sites
        assert!(
            agreeing > 0 && agreeing < bonds,
            "the fixture must hold bonds of both kinds, got {agreeing} of {bonds}"
        );

        let mut rng = Counting::new(RandRng::seed_from_u64(1));
        updater.sweep(&mut config, &lat, &action, 0.7, &mut rng);

        assert_eq!(
            rng.uniforms,
            agreeing,
            "a uniform per agreeing bond and none for the {} that disagree",
            bonds - agreeing
        );
    }

    /// One seed, one sweep, one answer — including through the data-dependent
    /// path above, where how many draws are taken depends on the configuration.
    #[test]
    fn a_cluster_sweep_is_reproducible_from_its_seed() {
        let lat = Lattice::new([6, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = SwendsenWang::for_model(&action);

        let run = |seed: u64| {
            let mut setup = RandRng::seed_from_u64(5);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut setup);
            let mut rng = RandRng::seed_from_u64(seed);
            let mut net = 0.0;
            for _ in 0..8 {
                net += updater.sweep(&mut config, &lat, &action, 0.8, &mut rng);
            }
            (config, net)
        };

        assert_eq!(run(42), run(42));
        assert_ne!(
            run(42).0,
            run(43).0,
            "a different seed should give a different chain"
        );
    }

    /// At very large `beta` every agreeing bond opens, so a uniform start comes
    /// back as one cluster carrying one label everywhere.
    ///
    /// A uniform start is the point: at large `beta` a *disagreeing* bond never
    /// opens whatever the coupling, so only a configuration that already agrees
    /// everywhere collapses to a single cluster.
    #[test]
    fn at_large_beta_a_uniform_lattice_is_one_cluster() {
        fn probe<const Q: usize, const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = SwendsenWang::for_model(&action);
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            let mut rng = Counting::new(RandRng::seed_from_u64(9));

            updater.sweep(&mut config, &lat, &action, 40.0, &mut rng);

            assert_eq!(rng.below, 1, "{shape:?} at Q = {Q}: one cluster, one draw");
            let first = config.peek(0);
            assert!(
                config.variables().iter().all(|&s| s == first),
                "{shape:?} at Q = {Q}: one cluster must land on one label"
            );
        }

        probe::<2, 2>([4, 6]);
        probe::<3, 2>([4, 6]);
        probe::<4, 2>([4, 6]);
        probe::<3, 3>([3, 4, 5]);
    }

    /// The net `ΔE` a cluster sweep returns equals `H(after) − H(before)`, the
    /// invariant every other updater is held to.
    ///
    /// Trivially true as written, since the sweep computes exactly that
    /// difference — which is the point: a regression that stopped computing the
    /// return value at all would show up here and nowhere else.
    #[test]
    fn the_cluster_sweep_net_delta_equals_the_energy_change() {
        fn probe<const Q: usize, const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = SwendsenWang::for_model(&action);
            let mut rng = RandRng::seed_from_u64(20260810);
            let mut config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let before = action.energy(&lat, &config);

            let net = updater.sweep(&mut config, &lat, &action, 0.6, &mut rng);

            assert_eq!(net, action.energy(&lat, &config) - before, "{shape:?}");
            assert_ne!(net, 0.0, "{shape:?}: a hot lattice should move");
        }

        probe::<2, 2>([8, 8]);
        probe::<3, 2>([8, 8]);
        probe::<4, 2>([8, 8]);
        probe::<3, 3>([4, 4, 4]);
    }

    /// The cluster update runs on an odd lattice, where no parallel coloring
    /// could: there is no coloring to collide, only a graph to label.
    #[test]
    fn the_cluster_update_runs_on_odd_extents() {
        let lat = Lattice::new([3, 5, 3]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = SwendsenWang::for_model(&action);
        let mut rng = RandRng::seed_from_u64(3);
        let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        let net = updater.sweep(&mut config, &lat, &action, 0.5, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// The cluster update reaches the same distribution the single-site schedule
    /// does, on a lattice small enough that the mean order parameter is a sharp
    /// enough statistic to separate a wrong bond probability.
    ///
    /// A factor of two in the gap — the likeliest implementation error — moves
    /// this well outside the window.
    #[test]
    fn the_cluster_update_matches_the_metropolis_distribution() {
        fn mean_order(updater: &AnyUpdater, seed: u64) -> f64 {
            let lat = Lattice::new([8, 8]);
            let action = Potts::<3>::symmetric(1.0);
            let mut rng = RandRng::seed_from_u64(seed);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
            let mut chain =
                crate::chain::Chain::new(&mut config, &lat, &action, updater, 1.1, &mut rng, 2);
            chain.advance(200);
            let n = 400;
            chain
                .take(n)
                .map(|c| action.order_parameter(&c))
                .sum::<f64>()
                / n as f64
        }

        let action = Potts::<3>::symmetric(1.0);
        let metropolis = mean_order(&AnyUpdater::Metropolis(Metropolis), 17);
        let cluster = mean_order(
            &AnyUpdater::SwendsenWang(SwendsenWang::for_model(&action)),
            23,
        );

        assert!(
            (metropolis - cluster).abs() < 0.04,
            "mean order parameter: metropolis {metropolis}, swendsen-wang {cluster}"
        );
    }

    /// A link field is rejected rather than silently misread as a lattice of
    /// sites, the same guard the link schedule carries the other way round.
    #[test]
    #[should_panic(expected = "must be a site field")]
    fn the_cluster_update_rejects_a_link_field() {
        let lat = Lattice::new([4, 4]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = SwendsenWang::for_model(&action);
        let mut config = Configuration::<3>::cold(&lat, Cell::Link);
        updater.sweep(
            &mut config,
            &lat,
            &action,
            1.0,
            &mut RandRng::seed_from_u64(1),
        );
    }

    /// A site field is rejected rather than silently misread as links.
    #[test]
    #[should_panic(expected = "must be a link field")]
    fn link_checkerboard_rejects_a_site_field() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        LinkCheckerboard.sweep(
            &mut config,
            &lat,
            &action,
            1.0,
            &mut RandRng::seed_from_u64(1),
        );
    }

    /// Both checkerboard heat baths satisfy the telescoping identity every
    /// updater here is held to, on the grade each schedules.
    ///
    /// They exist to be the sequential reference for the device kernels, so what
    /// matters most is that they are the *same* schedule as their Metropolis
    /// twins: `site_checkerboard_sweep_attempts_every_site_once` and
    /// `a_link_pass_covers_every_link_once` already pin those color walks, and
    /// these were copied from them unchanged.
    #[test]
    fn the_checkerboard_heat_baths_net_delta_equals_the_energy_change() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let mut rng = RandRng::seed_from_u64(31);
        let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);
        let net = SiteCheckerboardHeatBath.sweep(&mut config, &lat, &action, 0.6, &mut rng);
        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot lattice should move");

        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut rng = RandRng::seed_from_u64(32);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
        let before = action.energy(&lat, &config);
        let net = LinkCheckerboardHeatBath.sweep(&mut config, &lat, &action, 0.6, &mut rng);
        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot gauge field should move");
    }

    /// Each checkerboard heat bath draws exactly one uniform per variable and
    /// picks nothing at random, the way its Metropolis twin does — the schedule
    /// is deterministic and only the kernel consumes the generator.
    #[test]
    fn the_checkerboard_heat_baths_draw_one_uniform_per_variable() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let mut config = Configuration::<3>::cold(&lat, Cell::Site);
        let mut rng = Counting::new(RandRng::seed_from_u64(2));
        SiteCheckerboardHeatBath.sweep(&mut config, &lat, &action, 0.7, &mut rng);
        assert_eq!(rng.uniforms, lat.n_sites(), "one uniform per site");
        assert_eq!(rng.below, 0, "the site checkerboard picks no random sites");

        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        let mut rng = Counting::new(RandRng::seed_from_u64(2));
        LinkCheckerboardHeatBath.sweep(&mut config, &lat, &action, 0.7, &mut rng);
        assert_eq!(rng.uniforms, lat.n_links(), "one uniform per link");
        assert_eq!(
            rng.below, 0,
            "the link checkerboard picks nothing at random"
        );
    }

    /// A site field is rejected by the link checkerboard heat bath, the same
    /// guard its Metropolis twin carries.
    #[test]
    #[should_panic(expected = "must be a link field")]
    fn the_link_checkerboard_heat_bath_rejects_a_site_field() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        LinkCheckerboardHeatBath.sweep(
            &mut config,
            &lat,
            &action,
            1.0,
            &mut RandRng::seed_from_u64(1),
        );
    }

    /// At `beta = 0` the conditional is flat, so the draw is uniform over the
    /// `Q` states — the infinite-temperature limit, and the case that catches a
    /// cumulative walk skewed toward either end of the weight array.
    #[test]
    fn at_zero_beta_the_heat_bath_draws_uniformly() {
        const Q: usize = 4;
        const DRAWS: usize = 4_000;

        let lat = Lattice::new([4, 4]);
        let action = Potts::<Q>::symmetric(1.0);
        let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
        let mut rng = RandRng::seed_from_u64(11);

        let mut seen = [0usize; Q];
        for _ in 0..DRAWS {
            heat_bath_step(&mut config, &lat, &action, 0, 0.0, &mut rng);
            seen[config.peek(0).index()] += 1;
        }

        for (index, &count) in seen.iter().enumerate() {
            // Four states share 4000 draws, so each expects 1000 with a standard
            // deviation near 27; this window is wide enough to be seed-robust
            // and far too narrow for a walk that favored a state to pass.
            assert!(
                (870..1_130).contains(&count),
                "state {index}: {count} of {DRAWS}"
            );
        }
    }

    /// At large `beta` the draw collapses onto the lowest-energy candidate, and
    /// the arithmetic that gets there stays finite. Both halves matter: the
    /// shift by the smallest delta is the only thing keeping `exp` from
    /// overflowing on a strongly downhill state, and an infinite weight would
    /// poison the cumulative walk rather than fail outright.
    #[test]
    fn at_large_beta_the_heat_bath_lands_on_the_lowest_energy_state() {
        let lat = Lattice::new([4, 4]);
        let action = Potts::<3>::symmetric(1.0);
        let mut config = Configuration::<3>::cold(&lat, Cell::Site);
        // One site moved off the aligned background, so agreeing with its four
        // neighbors again is the unique lowest-energy candidate.
        config.poke(0, State::new(2).unwrap());
        let mut rng = RandRng::seed_from_u64(3);

        for draw in 0..32 {
            let delta = heat_bath_step(&mut config, &lat, &action, 0, 500.0, &mut rng);
            assert!(
                delta.is_finite(),
                "draw {draw}: the weights must stay finite"
            );
            assert_eq!(
                config.peek(0).index(),
                0,
                "draw {draw}: the aligned state should win at large beta"
            );
        }
    }

    /// A heat bath sweep draws exactly one uniform per variable whatever the
    /// configuration. That is the contract that separates it from a Metropolis
    /// sweep, whose downhill moves skip the accept draw and so leave the stream
    /// position depending on the state.
    #[test]
    fn a_heat_bath_sweep_draws_one_uniform_per_variable() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let n = lat.n_sites();

        // Both a cold start, where every step is uphill, and a hot one, where
        // the mix varies site to site: the count must not notice the difference.
        for cold in [true, false] {
            let mut setup = RandRng::seed_from_u64(8);
            let mut config = if cold {
                Configuration::<3>::cold(&lat, Cell::Site)
            } else {
                Configuration::<3>::hot(&lat, Cell::Site, &mut setup)
            };

            let mut rng = Counting::new(RandRng::seed_from_u64(2));
            HeatBath.sweep(&mut config, &lat, &action, 0.7, &mut rng);

            assert_eq!(rng.uniforms, n, "cold = {cold}: one uniform per variable");
            assert_eq!(rng.below, n, "cold = {cold}: one variable pick per step");
        }
    }

    /// The net `ΔE` a heat bath sweep returns equals `H(after) − H(before)`, the
    /// invariant every updater here is held to. The couplings are integer-valued
    /// so the comparison is bit-exact.
    #[test]
    fn heat_bath_sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let mut rng = RandRng::seed_from_u64(7);
        let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        let net = HeatBath.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot lattice should move");
    }

    /// The same kernel runs on a link field with no change of its own, which is
    /// what grade-neutrality buys: the gauge model is served by the code that
    /// serves the spin models, and not by a variant of it.
    #[test]
    fn the_heat_bath_runs_on_a_link_field() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut rng = RandRng::seed_from_u64(5);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
        let before = action.energy(&lat, &config);

        let net = HeatBath.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot gauge field should move");
    }

    /// A model whose energy is not invariant under relabeling runs here without
    /// comment, which is the whole difference from the cluster update:
    /// `SwendsenWang::for_model` refuses an Ising field outright, while the heat
    /// bath simply carries it in `ΔE` and tilts the conditional.
    #[test]
    fn the_heat_bath_accepts_a_model_with_a_field() {
        let lat = Lattice::new([6, 6]);
        let action = Ising::new(1.0, 0.25); // exactly representable; sums stay exact
        let mut rng = RandRng::seed_from_u64(4);
        let mut config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        let net = HeatBath.sweep(&mut config, &lat, &action, 0.5, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// The heat bath reaches the same distribution the Metropolis schedule
    /// does, on a lattice small enough that the mean order parameter is a sharp
    /// enough statistic to separate a mispriced conditional.
    ///
    /// A sign error in the exponent — the likeliest implementation mistake, and
    /// one that leaves every other test here passing — moves this far outside
    /// the window.
    #[test]
    fn the_heat_bath_matches_the_metropolis_distribution() {
        fn mean_order<U: Updater<3, 2>>(updater: &U, seed: u64) -> f64 {
            let lat = Lattice::new([8, 8]);
            let action = Potts::<3>::symmetric(1.0);
            let mut rng = RandRng::seed_from_u64(seed);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
            let mut chain =
                crate::chain::Chain::new(&mut config, &lat, &action, updater, 1.1, &mut rng, 2);
            chain.advance(200);
            let n = 400;
            chain
                .take(n)
                .map(|c| action.order_parameter(&c))
                .sum::<f64>()
                / n as f64
        }

        let metropolis = mean_order(&Metropolis, 17);
        let heat_bath = mean_order(&HeatBath, 23);

        assert!(
            (metropolis - heat_bath).abs() < 0.04,
            "mean order parameter: metropolis {metropolis}, heat bath {heat_bath}"
        );
    }
}
