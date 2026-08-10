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
//! The updater holds no chain state: the [`Configuration`] *is* the current
//! state, mutated in place, and `β` is passed per call so one updater serves a
//! whole temperature scan. It keeps no running energy either;
//! [`sweep`](Updater::sweep) returns the net realized ΔE, and re-anchoring
//! against a from-scratch [`Action::energy`] stays the driver's job.

use crate::action::Action;
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

/// A runtime choice among the built-in updaters, so an updater named in a config
/// file can be selected without the caller committing to a type at compile time.
///
/// Implements [`Updater`] by forwarding `sweep` to whichever updater it wraps.
/// Its variants mirror [`UpdaterKind`](crate::config::UpdaterKind) — a closed
/// set, which is what makes the choice recordable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnyUpdater {
    /// The random-site schedule, [`Metropolis`].
    Metropolis(Metropolis),
    /// The site checkerboard schedule, [`SiteCheckerboard`].
    SiteCheckerboard(SiteCheckerboard),
    /// The link checkerboard schedule, [`LinkCheckerboard`].
    LinkCheckerboard(LinkCheckerboard),
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
            AnyUpdater::SiteCheckerboard(u) => u.sweep(config, lattice, action, beta, rng),
            AnyUpdater::LinkCheckerboard(u) => u.sweep(config, lattice, action, beta, rng),
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
}
