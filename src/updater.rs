//! Updater: the rule that advances the Markov chain one sweep at a time.
//!
//! Where the [`Action`] says what a configuration costs, the updater says how
//! the chain moves. It is the seam the driver depends on: a chain calls
//! [`sweep`](Updater::sweep) without naming an algorithm.
//!
//! The whole local-update family is one type, [`LocalUpdate`], composed from
//! two orthogonal choices. A [`Kernel`] is what happens at one variable —
//! propose an alternative and accept or reject it ([`Kernel::Metropolis`],
//! derived in `docs/metropolis.md`), or price every state the variable could
//! take and draw from the conditional they define ([`Kernel::HeatBath`],
//! described in `docs/heat-bath.md`). A [`Schedule`] is which variables are
//! visited in what order — uniformly at random ([`Schedule::Random`]) or every
//! variable exactly once, color by color ([`Schedule::Checkerboard`]). The two
//! compose freely because the kernel reads only a bare variable index and
//! [`Action::energy_delta`], never learning which model it is running or
//! whether the variable is a site or a link, and the schedule only walks,
//! never prices. The checkerboard reads which cell the field lives on and
//! colors accordingly — site parity on a site field, direction paired with
//! base-site parity on a link field — so "checkerboard" names one idea, not
//! one per grade.
//!
//! [`ClusterUpdate`] is the second family, with a composition of its own
//! that mirrors the first: an [`Extent`] says which clusters move together,
//! and a [`Relabel`] rule says what happens to each — Swendsen–Wang is every
//! cluster, freshly redrawn (`docs/swendsen-wang.md`), and Wolff is one
//! seeded cluster, forced onto a different label (`docs/wolff.md`). Its
//! variables are built stochastically from the current state, changed many at
//! once, and accepted with probability one, which is why the family stands
//! beside the local composition rather than inside it, sharing only the
//! [`Updater`] seam.
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
/// field and a link on a link field; that grade-neutrality is what lets every
/// [`Schedule`] hand it variables of either grade unchanged.
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

/// Which single-variable rule a [`LocalUpdate`] applies at each variable its
/// schedule hands in.
///
/// The kernel is deliberately blind to everything else: it receives a bare
/// variable index, prices a change through [`Action::energy_delta`], and never
/// learns which model it is running or whether the variable is a site or a
/// link. That blindness is what makes one Metropolis serve every model. The
/// device backend applies the same two rules in its shaders, chosen by this
/// same enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Kernel {
    /// Propose one alternative state and accept it with `min(1, e^{-β ΔE})` —
    /// the rule `docs/metropolis.md` derives.
    #[default]
    Metropolis,
    /// Price every state the variable could take and draw one from the
    /// conditional distribution they define — the rule `docs/heat-bath.md`
    /// describes. Lands somewhere every time, since
    /// the draw *is* the conditional and there is nothing left to accept
    /// against.
    HeatBath,
}

/// Which variables a [`LocalUpdate`] visits in one sweep, and in what order.
///
/// The schedule owns the walk and nothing else: it hands bare variable indices
/// to the kernel and prices nothing itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Schedule {
    /// `n_vars` picks at uniformly random variables per sweep — the
    /// conventional random schedule.
    #[default]
    Random,
    /// Every variable exactly once, color by color, where the coloring puts no
    /// two interacting variables in one color.
    ///
    /// The coloring follows from which cell the field lives on, read off the
    /// configuration rather than named by the caller. A site field colors each
    /// site by the parity of its coordinate sum — two passes; a link field
    /// colors each link by its direction paired with its base site's parity —
    /// `2D` passes, because base-site parity alone cannot separate a
    /// plaquette's links (see `docs/metropolis.md`).
    ///
    /// Run sequentially, any order is a valid schedule, so this sweep is
    /// correct on any lattice; its purpose is to be the sequential reference
    /// for the parallel device sweep, where a whole color updates at once and
    /// the independence needs every extent even — an odd extent wraps two
    /// same-color variables next to each other.
    ///
    /// The independence argument also fixes how far a model's interaction may
    /// reach: `energy_delta` at a variable must read only variables of *other*
    /// colors — nearest neighbors on a site field, one plaquette's links on a
    /// link field. A longer-ranged action (a next-nearest-neighbor coupling,
    /// an improved gauge action with rectangle terms) breaks the coloring's
    /// independence, and only this sequential sweep stays correct until a
    /// finer coloring is built for it. Every action in the crate today is
    /// within range; the constraint binds the *parallel* backend.
    Checkerboard,
}

/// Color passes a checkerboard sweep makes over a field on `cell` in
/// `dimension` dimensions: two site parities, or `2 * dimension`
/// direction–parity pairs on links.
///
/// Shared with the device backend, which turns it into dispatches per sweep
/// and must agree with the order the CPU sweep walks, since the CPU schedule
/// is the reference the device kernels are checked against.
pub(crate) const fn checkerboard_colors(cell: Cell, dimension: usize) -> usize {
    match cell {
        Cell::Site => 2,
        Cell::Link => 2 * dimension,
    }
}

/// A single-variable update: a [`Kernel`] applied under a [`Schedule`],
/// implementing [`Updater`] for any `Q` in any dimension and for any model —
/// the model enters only through [`Action::energy_delta`].
///
/// The four compositions are all valid and all sample the Boltzmann
/// distribution; they differ in mixing and in what can run in parallel, not in
/// physics. The checkerboard compositions double as the sequential references
/// the device kernels are checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LocalUpdate {
    /// What happens at one variable.
    pub kernel: Kernel,
    /// Which variables, in what order.
    pub schedule: Schedule,
}

impl LocalUpdate {
    /// Compose a kernel and a schedule. `const`, so named combinations can be
    /// constants.
    pub const fn new(kernel: Kernel, schedule: Schedule) -> Self {
        LocalUpdate { kernel, schedule }
    }

    /// One kernel application at `var`, whichever rule this update composes.
    fn apply<const Q: usize, const D: usize>(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        var: usize,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        match self.kernel {
            Kernel::Metropolis => step(config, lattice, action, var, beta, rng),
            Kernel::HeatBath => heat_bath_step(config, lattice, action, var, beta, rng),
        }
    }
}

impl<const Q: usize, const D: usize> Updater<Q, D> for LocalUpdate {
    /// One kernel application per variable — `n_vars` random picks, or every
    /// variable once in color order.
    ///
    /// # Panics
    ///
    /// Panics if the kernel is [`Kernel::HeatBath`] and `Q` is below two,
    /// since the draw needs states to choose between.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        if self.kernel == Kernel::HeatBath {
            assert!(
                Q >= 2,
                "the heat bath draws a state from among the Q available, which needs \
                 at least two"
            );
        }

        let mut net = 0.0;
        match self.schedule {
            Schedule::Random => {
                for _ in 0..config.n_vars() {
                    let var = rng.next_below(config.n_vars());
                    net += self.apply(config, lattice, action, var, beta, rng);
                }
            }
            Schedule::Checkerboard => match config.cell() {
                Cell::Site => {
                    for color in [0, 1] {
                        for site in 0..config.n_vars() {
                            if lattice.site_parity(site) == color {
                                net += self.apply(config, lattice, action, site, beta, rng);
                            }
                        }
                    }
                }
                Cell::Link => {
                    for dir in 0..D {
                        for color in [0, 1] {
                            // Iterate over *sites* and address the one link each
                            // owns in this direction, rather than scanning all
                            // `D * n_sites` links and skipping other directions.
                            // The visiting order is the same either way —
                            // `link_colors_are_visited_in_link_order` pins that —
                            // and it is the mapping the GPU kernel launches one
                            // thread per.
                            for site in 0..lattice.n_sites() {
                                if lattice.site_parity(site) == color {
                                    let link = lattice.site_link(site, dir);
                                    net += self.apply(config, lattice, action, link, beta, rng);
                                }
                            }
                        }
                    }
                }
            },
        }
        net
    }
}

/// How a cluster sweep selects the clusters it updates — the cluster family's
/// analogue of a [`Schedule`], answering "which variables move together".
///
/// Unlike a schedule it is not state-blind: the clusters are built from the
/// current configuration and fresh randomness, which is exactly why the
/// cluster family stands beside the [`LocalUpdate`] composition rather than
/// inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Extent {
    /// Decompose the whole lattice into clusters and update every one — the
    /// Swendsen–Wang extent.
    All,
    /// Grow one cluster outward from a uniformly random seed site and update
    /// only that — the Wolff extent. Seeding at a uniformly random *site*
    /// picks a cluster with probability proportional to its size, which is
    /// what aims the move at the large clusters that decorrelate slowest; see
    /// `docs/wolff.md`. On the CPU the growth touches only the cluster, so a
    /// move costs the cluster rather than the volume.
    Seeded,
}

/// What happens to each selected cluster's label — the cluster family's
/// analogue of a [`Kernel`], answering "what happens to what was selected".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relabel {
    /// Draw a fresh label uniformly among all `Q`, so a cluster may come back
    /// on the label it had. Given the bonds, uniform *is* the exact
    /// conditional — the heat bath archetype, one level up.
    Redraw,
    /// Draw a fresh label uniformly among the `Q − 1` labels the cluster does
    /// not carry — the Wolff algorithm's rule (the propose-and-accept
    /// archetype, with acceptance identically one). At `Q = 2` this is the
    /// flip, and like the local kernels' proposal — whose draw it shares — it
    /// then consumes no randomness. Given the
    /// bonds it preserves the uniform label conditional — from any label,
    /// every *other* label is equally likely, a symmetric doubly stochastic
    /// kernel — so the move is exact without being the conditional itself.
    ForcedChange,
}

/// A cluster update: an [`Extent`] applied with a [`Relabel`] rule — bond
/// neighbors that agree, then relabel the clusters the bonds define.
///
/// The cluster family's composition, mirroring how [`LocalUpdate`] composes a
/// kernel under a schedule; Swendsen–Wang is [`Extent::All`] with
/// [`Relabel::Redraw`] (`docs/swendsen-wang.md`) and Wolff is
/// [`Extent::Seeded`] with [`Relabel::ForcedChange`] (`docs/wolff.md`). The
/// off-diagonal compositions are valid too, with one exception refused at
/// construction (see [`new`](ClusterUpdate::new)).
/// The axes share no types with the local family because they obey different
/// contracts: an extent reads the state and consumes randomness where a
/// schedule is state-blind, and a relabel rule's conditional rests on the
/// bond variables where a kernel prices through
/// [`Action::energy_delta`].
///
/// Unlike the local updaters this one carries state — the model's bond gap,
/// read once at construction — so it is built with
/// [`swendsen_wang`](ClusterUpdate::swendsen_wang) (or composed via
/// [`new`](ClusterUpdate::new)) rather than named directly. Reading it once is
/// what keeps [`Updater`] a uniform capability rather than a relation between
/// an updater and an action: the whole dependence on the model is two numbers
/// that do not move over a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterUpdate {
    /// The model's `E(disagree) − E(agree)` per bond, from
    /// [`BondAction::bond_energy_gap`].
    bond_gap: f64,
    /// Which clusters a sweep updates.
    pub extent: Extent,
    /// What happens to each selected cluster's label.
    pub relabel: Relabel,
}

impl ClusterUpdate {
    /// Compose a cluster update for `model` from its two axes, capturing the
    /// model's bond gap.
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
    ///
    /// Panics, finally, on the one composition that is not a valid algorithm:
    /// [`Extent::All`] with [`Relabel::ForcedChange`] at `Q = 2`. There the
    /// forced change is the deterministic flip, so the move is a function of
    /// the bonds alone — and a configuration that agrees everywhere is one
    /// cluster under any bond draw, so it flips whole and lands on the other
    /// uniform configuration, which flips straight back. The chain is trapped
    /// in that two-cycle the moment it touches either ordered state, sampling
    /// the wrong distribution without ever failing. At `Q ≥ 3` the forced
    /// change has real choices and the composition is exact and ergodic.
    pub fn new<const Q: usize, M: BondAction<Q>>(
        model: &M,
        extent: Extent,
        relabel: Relabel,
    ) -> Self {
        assert!(
            model.relabel_invariant(),
            "the cluster move relabels a whole cluster at once, which is only \
             weight-preserving when the energy is invariant under relabeling; \
             this model has a per-label offset or an external field set"
        );
        assert!(
            !(Q == 2 && extent == Extent::All && relabel == Relabel::ForcedChange),
            "at Q = 2 a forced change of every cluster is the deterministic \
             global flip on any uniform configuration, so the chain falls into \
             a two-cycle and stops being ergodic; use Relabel::Redraw for the \
             all-clusters extent, or the seeded extent for a forced change"
        );
        let bond_gap = model.bond_energy_gap();
        assert!(
            bond_gap.is_finite() && bond_gap >= 0.0,
            "the bond probability 1 - exp(-beta * gap) needs a finite, \
             non-negative gap, got {bond_gap}; an antiferromagnetic coupling is \
             frustrated rather than merely inverted and needs a different \
             construction"
        );
        ClusterUpdate {
            bond_gap,
            extent,
            relabel,
        }
    }

    /// The Swendsen–Wang composition: every cluster, freshly redrawn.
    pub fn swendsen_wang<const Q: usize, M: BondAction<Q>>(model: &M) -> Self {
        Self::new(model, Extent::All, Relabel::Redraw)
    }

    /// The Wolff composition: one seeded cluster, forced onto a different
    /// label. See `docs/wolff.md`.
    pub fn wolff<const Q: usize, M: BondAction<Q>>(model: &M) -> Self {
        Self::new(model, Extent::Seeded, Relabel::ForcedChange)
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

    /// One label under this update's relabel rule, for a cluster about to be
    /// written; `current` is the label the cluster carries, which every member
    /// shares since bonds open only between agreeing sites.
    fn fresh_label<const Q: usize>(&self, current: State<Q>, rng: &mut impl Rng) -> State<Q> {
        match self.relabel {
            // A *redraw*, not a flip: a cluster may come back on the label it
            // had. At Q = 2 that differs from the textbook "flip each cluster
            // with probability one half" only in bookkeeping, and both are
            // correct.
            Relabel::Redraw => State::new(rng.next_below(Q)).expect("next_below(Q) < Q"),
            // The same symmetric draw over the Q − 1 alternatives the local
            // kernels use, including its Q = 2 contract of consuming nothing.
            Relabel::ForcedChange => propose(current, rng),
        }
    }
}

impl<const Q: usize, const D: usize> Updater<Q, D> for ClusterUpdate {
    /// One pass of the composed extent — under [`Extent::All`], a cluster
    /// decomposition of the whole lattice and then one label per cluster from
    /// the relabel rule; under [`Extent::Seeded`], one grown cluster and one
    /// label. Either way a single move, not `n_vars` of them — and a seeded
    /// sweep is *one cluster*, deliberately much less work than a sweep of any
    /// other updater here; `docs/wolff.md` covers what that does to sweep
    /// accounting.
    ///
    /// The returned `ΔE` is a from-scratch difference rather than an
    /// accumulation, since a cluster move does not price itself one site at a
    /// time. That is two extra `O(D·V)` scans — for [`Extent::All`] a constant
    /// factor on a labeling pass that is already `Θ(D·V)`, paid to keep the
    /// seam uniform. For [`Extent::Seeded`] the scans dominate a small
    /// cluster's growth, which is accepted for the same reason: the seam
    /// promises the realized `ΔE`, and pricing a cluster boundary
    /// incrementally would put an energy method on the extent.
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

        match self.extent {
            Extent::All => {
                // The short-circuit is part of the contract, not an
                // optimization: a uniform is drawn only for a pair that
                // *agrees*, so the position in the stream depends on the
                // configuration. Drawing unconditionally would sample the same
                // distribution and put every existing run on a different
                // stream.
                let clusters = cluster::site_clusters(lattice, |i, j| {
                    config.peek(i) == config.peek(j) && rng.next_f64() < p
                });

                // Each cluster's current label, read off its first member —
                // labels are compacted in site order, so first encounters walk
                // the labels in order. A scan, not a draw, so the Redraw
                // stream is untouched by the forced change needing it.
                let mut current: Vec<State<Q>> = Vec::with_capacity(clusters.n_clusters());
                for (site, &label) in clusters.labels().iter().enumerate() {
                    if label == current.len() {
                        current.push(config.peek(site));
                    }
                }
                let fresh: Vec<State<Q>> = current
                    .iter()
                    .map(|&label| self.fresh_label(label, rng))
                    .collect();
                for (site, &label) in clusters.labels().iter().enumerate() {
                    config.poke(site, fresh[label]);
                }
            }
            Extent::Seeded => {
                // A uniformly random site, not a uniformly random cluster:
                // the seed lands in a cluster with probability proportional
                // to its size, and that size bias is part of the algorithm.
                let seed = rng.next_below(config.n_vars());
                let members = cluster::grow_cluster(lattice, seed, |inside, outside| {
                    // The same agreement short-circuit as above, with the same
                    // stream contract: one uniform per offered bond whose far
                    // end agrees, none otherwise.
                    config.peek(inside) == config.peek(outside) && rng.next_f64() < p
                });
                let fresh = self.fresh_label(config.peek(seed), rng);
                for &site in &members {
                    config.poke(site, fresh);
                }
            }
        }

        action.energy(lattice, config) - before
    }
}

/// A runtime choice among the built-in updaters, so an updater named in a config
/// file can be selected without the caller committing to a type at compile time.
///
/// Implements [`Updater`] by forwarding `sweep` to whichever updater it wraps.
/// Two variants cover everything: a [`LocalUpdate`] is any kernel under any
/// schedule, and a [`ClusterUpdate`] is any extent with any relabel rule.
///
/// It is `PartialEq` but not `Eq`, because [`ClusterUpdate`] carries a bond
/// gap and floats have no total equality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyUpdater {
    /// A kernel under a schedule, [`LocalUpdate`].
    Local(LocalUpdate),
    /// An extent with a relabel rule, [`ClusterUpdate`].
    Cluster(ClusterUpdate),
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
            AnyUpdater::Local(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::Cluster(u) => u.sweep(config, lattice, action, beta, rng),
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

    /// The four kernel × schedule compositions, named once for the tests.
    const METROPOLIS: LocalUpdate = LocalUpdate::new(Kernel::Metropolis, Schedule::Random);
    const HEAT_BATH: LocalUpdate = LocalUpdate::new(Kernel::HeatBath, Schedule::Random);
    const CHECKERBOARD: LocalUpdate = LocalUpdate::new(Kernel::Metropolis, Schedule::Checkerboard);
    const CHECKERBOARD_HEAT_BATH: LocalUpdate =
        LocalUpdate::new(Kernel::HeatBath, Schedule::Checkerboard);

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
        let net = METROPOLIS.sweep(&mut config, &lat, &action, 1.0, &mut rng);

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
        let net = METROPOLIS.sweep(&mut config, &lat, &action, 0.6, &mut rng);

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
        let net = CHECKERBOARD.sweep(&mut config, &lat, &action, 1.0, &mut rng);

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
        let net = CHECKERBOARD.sweep(&mut config, &lat, &action, 0.6, &mut rng);

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
            ("metropolis", AnyUpdater::Local(METROPOLIS)),
            ("site checkerboard", AnyUpdater::Local(CHECKERBOARD)),
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

    /// A link's color under the link checkerboard walk: its direction paired with its
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
        let net = CHECKERBOARD.sweep(&mut config, &lat, &action, 1.0, &mut rng);

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
        let net = CHECKERBOARD.sweep(&mut config, &lat, &action, 0.6, &mut rng);

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
        let net = CHECKERBOARD.sweep(&mut config, &lat, &action, 0.5, &mut rng);

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
        ClusterUpdate::swendsen_wang(&Potts::<3>::new(1.0, [0.5, 0.0, 0.0]));
    }

    /// The same rule reached through the other implementor, whose symmetry is
    /// broken by a field rather than by a per-label offset.
    #[test]
    #[should_panic(expected = "invariant under relabeling")]
    fn the_cluster_update_refuses_an_ising_model_with_a_field() {
        ClusterUpdate::swendsen_wang(&Ising::new(1.0, 0.25));
    }

    /// An antiferromagnetic coupling is refused too. Nothing would fail on its
    /// own: the bond probability would come out negative, no bond would open,
    /// and the chain would sample the infinite-temperature model while reporting
    /// the coupling it was given.
    #[test]
    #[should_panic(expected = "non-negative gap")]
    fn the_cluster_update_refuses_an_antiferromagnetic_coupling() {
        ClusterUpdate::swendsen_wang(&Potts::<3>::symmetric(-1.0));
    }

    /// The two implementors' gaps differ by the factor of two their conventions
    /// differ by, which is what makes a Potts run at `2J` and an Ising run at
    /// `J` open their bonds with the same probability.
    #[test]
    fn the_two_conventions_give_the_same_bond_probability() {
        let j = 0.75;
        let potts = ClusterUpdate::swendsen_wang(&Potts::<2>::symmetric(2.0 * j));
        let ising = ClusterUpdate::swendsen_wang(&Ising::new(j, 0.0));
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
            let updater = ClusterUpdate::swendsen_wang(&action);
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
        let updater = ClusterUpdate::swendsen_wang(&action);

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
        let updater = ClusterUpdate::swendsen_wang(&action);

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
            let updater = ClusterUpdate::swendsen_wang(&action);
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
            let updater = ClusterUpdate::swendsen_wang(&action);
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
        let updater = ClusterUpdate::swendsen_wang(&action);
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
        let metropolis = mean_order(&AnyUpdater::Local(METROPOLIS), 17);
        let cluster = mean_order(
            &AnyUpdater::Cluster(ClusterUpdate::swendsen_wang(&action)),
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
        let updater = ClusterUpdate::swendsen_wang(&action);
        let mut config = Configuration::<3>::cold(&lat, Cell::Link);
        updater.sweep(
            &mut config,
            &lat,
            &action,
            1.0,
            &mut RandRng::seed_from_u64(1),
        );
    }

    /// The one invalid composition is refused at construction: at two states a
    /// forced change of every cluster is deterministic given the bonds, and
    /// the chain two-cycles between the uniform configurations.
    #[test]
    #[should_panic(expected = "stops being ergodic")]
    fn all_clusters_with_a_forced_change_is_refused_at_two_states() {
        ClusterUpdate::new(
            &Potts::<2>::symmetric(1.0),
            Extent::All,
            Relabel::ForcedChange,
        );
    }

    /// The same composition at three states is exact and runs: at `beta = 0`
    /// every site is its own cluster and the forced change moves every one of
    /// them, drawing one alternative per cluster.
    #[test]
    fn at_three_states_all_clusters_with_a_forced_change_moves_every_site() {
        let lat = Lattice::new([4, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = ClusterUpdate::new(&action, Extent::All, Relabel::ForcedChange);
        let mut config = Configuration::<3>::cold(&lat, Cell::Site);
        let before = config.clone();
        let mut rng = Counting::new(RandRng::seed_from_u64(6));

        updater.sweep(&mut config, &lat, &action, 0.0, &mut rng);

        assert!(
            (0..lat.n_sites()).all(|s| config.peek(s) != before.peek(s)),
            "a forced change must move every singleton cluster"
        );
        // One agreement uniform per bond (a cold start agrees everywhere) and
        // one alternative draw per cluster; at p = 0 there are as many
        // clusters as sites.
        assert_eq!(rng.uniforms, 2 * lat.n_sites());
        assert_eq!(rng.below, lat.n_sites());
    }

    /// The Wolff constructor inherits the symmetry guard the composed
    /// constructor carries, on the model whose symmetry a field breaks.
    #[test]
    #[should_panic(expected = "invariant under relabeling")]
    fn the_wolff_update_refuses_an_ising_model_with_a_field() {
        ClusterUpdate::wolff(&Ising::new(1.0, 0.25));
    }

    /// At `beta = 0` no bond opens, so a Wolff move is the seed alone: exactly
    /// one site changes, and it changes to a *different* label. The draw
    /// counts pin the stream contract — one seed pick, one agreement uniform
    /// per neighbor of the seed, and one alternative draw at `Q = 3` against
    /// none at `Q = 2`, where the forced change is the flip.
    #[test]
    fn at_zero_beta_a_wolff_move_is_the_seed_alone() {
        fn probe<const Q: usize>(expected_label_draws: usize) {
            let lat = Lattice::new([4, 6]);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = ClusterUpdate::wolff(&action);
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            let before = config.clone();
            let mut rng = Counting::new(RandRng::seed_from_u64(12));

            updater.sweep(&mut config, &lat, &action, 0.0, &mut rng);

            let changed: Vec<usize> = (0..lat.n_sites())
                .filter(|&s| config.peek(s) != before.peek(s))
                .collect();
            assert_eq!(changed.len(), 1, "Q = {Q}: only the seed moves at p = 0");
            assert_eq!(
                rng.uniforms,
                2 * 2, // 2D agreeing neighbors of the seed, each offered once
                "Q = {Q}: one uniform per offered bond"
            );
            assert_eq!(rng.below, 1 + expected_label_draws, "Q = {Q}");
        }

        probe::<2>(0);
        probe::<3>(1);
    }

    /// At very large `beta` a uniform start grows the whole lattice into the
    /// seeded cluster, and the forced change repaints all of it in one other
    /// label — the global move a local update can essentially never make.
    #[test]
    fn at_large_beta_a_wolff_move_repaints_a_uniform_lattice() {
        fn probe<const Q: usize>() {
            let lat = Lattice::new([4, 6]);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = ClusterUpdate::wolff(&action);
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            let before = config.peek(0);
            let mut rng = RandRng::seed_from_u64(9);

            updater.sweep(&mut config, &lat, &action, 40.0, &mut rng);

            let first = config.peek(0);
            assert_ne!(first, before, "Q = {Q}: the label must change");
            assert!(
                config.variables().iter().all(|&s| s == first),
                "Q = {Q}: one cluster must land on one label"
            );
        }

        probe::<2>();
        probe::<3>();
    }

    /// The net `ΔE` a Wolff sweep returns equals `H(after) − H(before)`, the
    /// invariant every updater here is held to.
    #[test]
    fn the_wolff_sweep_net_delta_equals_the_energy_change() {
        fn probe<const Q: usize, const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let action = Potts::<Q>::symmetric(1.0);
            let updater = ClusterUpdate::wolff(&action);
            let mut rng = RandRng::seed_from_u64(20260815);
            let mut config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let before = action.energy(&lat, &config);

            let net = updater.sweep(&mut config, &lat, &action, 0.9, &mut rng);

            assert_eq!(net, action.energy(&lat, &config) - before, "{shape:?}");
        }

        probe::<2, 2>([8, 8]);
        probe::<3, 2>([8, 8]);
        probe::<3, 3>([4, 4, 4]);
    }

    /// One seed, one chain, one answer — through a path whose draw count
    /// depends on the configuration, as for Swendsen–Wang above.
    #[test]
    fn a_wolff_run_is_reproducible_from_its_seed() {
        let lat = Lattice::new([6, 6]);
        let action = Potts::<3>::symmetric(1.0);
        let updater = ClusterUpdate::wolff(&action);

        let run = |seed: u64| {
            let mut setup = RandRng::seed_from_u64(5);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut setup);
            let mut rng = RandRng::seed_from_u64(seed);
            let mut net = 0.0;
            for _ in 0..32 {
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

    /// The Wolff update reaches the same distribution the Metropolis schedule
    /// does. One Wolff sweep is one cluster rather than a lattice of attempts,
    /// so the chain decorrelates with more sweeps between samples than the
    /// Swendsen–Wang comparison needs.
    #[test]
    fn the_wolff_update_matches_the_metropolis_distribution() {
        fn mean_order(updater: &AnyUpdater, seed: u64, sweeps_between: usize) -> f64 {
            let lat = Lattice::new([8, 8]);
            let action = Potts::<3>::symmetric(1.0);
            let mut rng = RandRng::seed_from_u64(seed);
            let mut config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = crate::chain::Chain::new(
                &mut config,
                &lat,
                &action,
                updater,
                1.1,
                &mut rng,
                sweeps_between,
            );
            chain.advance(400);
            let n = 400;
            chain
                .take(n)
                .map(|c| action.order_parameter(&c))
                .sum::<f64>()
                / n as f64
        }

        let action = Potts::<3>::symmetric(1.0);
        let metropolis = mean_order(&AnyUpdater::Local(METROPOLIS), 17, 2);
        let wolff = mean_order(&AnyUpdater::Cluster(ClusterUpdate::wolff(&action)), 23, 8);

        assert!(
            (metropolis - wolff).abs() < 0.04,
            "mean order parameter: metropolis {metropolis}, wolff {wolff}"
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
        let net = CHECKERBOARD_HEAT_BATH.sweep(&mut config, &lat, &action, 0.6, &mut rng);
        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot lattice should move");

        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut rng = RandRng::seed_from_u64(32);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
        let before = action.energy(&lat, &config);
        let net = CHECKERBOARD_HEAT_BATH.sweep(&mut config, &lat, &action, 0.6, &mut rng);
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
        CHECKERBOARD_HEAT_BATH.sweep(&mut config, &lat, &action, 0.7, &mut rng);
        assert_eq!(rng.uniforms, lat.n_sites(), "one uniform per site");
        assert_eq!(rng.below, 0, "the site checkerboard picks no random sites");

        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        let mut rng = Counting::new(RandRng::seed_from_u64(2));
        CHECKERBOARD_HEAT_BATH.sweep(&mut config, &lat, &action, 0.7, &mut rng);
        assert_eq!(rng.uniforms, lat.n_links(), "one uniform per link");
        assert_eq!(
            rng.below, 0,
            "the link checkerboard picks nothing at random"
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
            HEAT_BATH.sweep(&mut config, &lat, &action, 0.7, &mut rng);

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

        let net = HEAT_BATH.sweep(&mut config, &lat, &action, 0.6, &mut rng);

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

        let net = HEAT_BATH.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
        assert_ne!(net, 0.0, "a hot gauge field should move");
    }

    /// A model whose energy is not invariant under relabeling runs here without
    /// comment, which is the whole difference from the cluster update:
    /// `ClusterUpdate::new` refuses an Ising field outright, while the heat
    /// bath simply carries it in `ΔE` and tilts the conditional.
    #[test]
    fn the_heat_bath_accepts_a_model_with_a_field() {
        let lat = Lattice::new([6, 6]);
        let action = Ising::new(1.0, 0.25); // exactly representable; sums stay exact
        let mut rng = RandRng::seed_from_u64(4);
        let mut config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        let net = HEAT_BATH.sweep(&mut config, &lat, &action, 0.5, &mut rng);

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

        let metropolis = mean_order(&METROPOLIS, 17);
        let heat_bath = mean_order(&HEAT_BATH, 23);

        assert!(
            (metropolis - heat_bath).abs() < 0.04,
            "mean order parameter: metropolis {metropolis}, heat bath {heat_bath}"
        );
    }
}
