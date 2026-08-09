//! Updater: the rule that advances the Markov chain one sweep at a time.
//!
//! Where the [`Action`] says what a configuration *costs*, the updater says how
//! the chain *moves*, so that the configurations visited over a long run are
//! Boltzmann-distributed. It is the seam the driver depends on: a chain calls
//! [`sweep`](Updater::sweep) without naming an algorithm, so heat-bath, cluster,
//! or checkerboard moves can replace Metropolis without touching it.
//!
//! [`sweep`](Updater::sweep) is the trait's whole obligation, because it is the
//! one thing every updater shares: each advances the chain by one sweep, whatever
//! happens inside. What differs between algorithms is the *schedule* — which
//! sites to update, in what order — and that lives in each updater's `sweep`. The
//! single-site accept/reject kernel `step` is a plain module function that the
//! schedules which have one call into; a cluster update would write a `sweep`
//! that never mentions it.
//!
//! [`Metropolis`] is the single-spin-flip update, the `Q = 2` case for any `D`.
//! Its sweep visits `N = n_vars` uniformly-random sites; each `step` proposes
//! the flipped state, prices the move with [`Action::energy_delta`], and accepts
//! with `min(1, e^{-β ΔE})`. The uniform pick is what makes the proposal
//! symmetric and so lets acceptance drop the Hastings ratio; pricing by local
//! delta rather than absolute energy is what cancels the partition function.
//! Uphill moves surviving with probability `e^{-β ΔE} > 0` is mandatory, not an
//! optimization — it is what lets the chain climb out of local minima.
//!
//! [`SiteCheckerboard`] keeps that accept/reject rule and changes only the order the
//! sites are visited in: it colors each site by coordinate-sum parity and updates
//! one color fully before the other, so no two sites updated together are
//! neighbors. On CPU that is just Metropolis in a fixed order, sampling the same
//! distribution; its purpose is to be the sequential form of a parallel (GPU)
//! sweep, where a whole color updates at once. See `docs/metropolis.md` for why
//! the reordering samples the same distribution and when the coloring's
//! independence holds.
//!
//! [`LinkCheckerboard`] is the same idea for a gauge model, where the variables
//! sit on links and the unit of interaction is the plaquette rather than the
//! bond. Base-site parity alone cannot separate the four links of a plaquette,
//! since two of them share a base site, so the coloring splits by direction
//! first and by parity second — `2D` colors and a `2D`-pass sweep instead of
//! two. It is a distinct type rather than a mode of [`SiteCheckerboard`] because
//! only the name is shared: the color rule, what a collision means, and the pass
//! count all differ, so one `sweep` covering both would be two unrelated bodies
//! behind a branch. `docs/metropolis.md` carries the argument for the rule.
//!
//! The updater holds no chain state: the [`Configuration`] *is* the current
//! state, mutated in place, and `β` is passed per call so one updater serves a
//! whole temperature scan. It keeps no running energy either, since the
//! configuration already determines its own and measurements are infrequent
//! enough for a driver to recompute [`Action::energy`] at sample points.
//! [`sweep`](Updater::sweep) still returns the net realized ΔE so a driver can
//! accumulate it if profiling ever justifies it.

use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::model::Action;
use crate::rng::Rng;
use crate::state::State;

/// The rule that advances the Markov chain by one sweep.
///
/// Generic over the field's state count `Q` and the lattice dimension `D`, so
/// the driver can name the seam without naming a specific algorithm — the peer
/// of [`Action`] on the sampling side.
///
/// [`sweep`](Updater::sweep) is the only requirement: it is what the chain calls
/// and the one obligation every algorithm shares. How a sweep is scheduled — and
/// whether it is built from single-site updates at all — is each updater's own
/// business.
pub trait Updater<const Q: usize, const D: usize> {
    /// One **sweep**: advance `config` in place by one sweep's worth of updates,
    /// returning the net realized `ΔE` summed over them. `beta` is passed per
    /// call so one updater serves a whole temperature scan.
    ///
    /// A sweep is the conventional unit of Monte Carlo time, sized to the lattice
    /// so autocorrelation is measured in sweeps rather than raw steps. What one
    /// sweep *does* is the algorithm's own choice: [`Metropolis`] attempts
    /// `n_vars` single-site updates at random sites; [`SiteCheckerboard`] attempts
    /// one per site in color order; an HMC trajectory would be a single
    /// whole-lattice move.
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

/// The single-variable-flip Metropolis update at a **given** variable: propose
/// the flip, price it with [`Action::energy_delta`], and accept with
/// `min(1, e^{-β ΔE})`, returning the realized `ΔE` (`0.0` on rejection, leaving
/// `config` unchanged).
///
/// This is the shared single-variable kernel, with the *variable handed in*
/// rather than chosen here — selecting it is the schedule's job, not the
/// kernel's. `var` is a bare index into the configuration, so it names a site on
/// a site field and a link on a link field; [`Action::energy_delta`] is likewise
/// grade-neutral, which is what lets [`LinkCheckerboard`] reuse this unchanged.
/// [`Metropolis`] (random order), [`SiteCheckerboard`] (parity order), and
/// [`LinkCheckerboard`] (direction-and-parity order) all drive their sweeps by
/// calling it, and differ only in which variables they pass and in what order.
fn step<const D: usize>(
    config: &mut Configuration<2>,
    lattice: &Lattice<D>,
    action: &impl Action<2, D>,
    var: usize,
    beta: f64,
    rng: &mut impl Rng,
) -> f64 {
    let proposed = propose(config.peek(var), rng);
    let delta = action.energy_delta(lattice, config, var, proposed);

    // Accept with min(1, e^{-β ΔE}). The `ΔE ≤ 0` short-circuit keeps downhill
    // moves without a draw and, by keeping the argument to `exp` non-positive,
    // also prevents overflow.
    if delta <= 0.0 || rng.next_f64() < (-beta * delta).exp() {
        config.poke(var, proposed);
        delta
    } else {
        0.0
    }
}

/// The single-spin-flip Metropolis update with a **random-site** schedule: a
/// stateless unit struct, implementing [`Updater`] for `Q = 2` in any dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metropolis;

impl<const D: usize> Updater<2, D> for Metropolis {
    /// `N = n_vars` single-site `step`s at uniformly-random sites. Drawing the
    /// site here — rather than inside `step` — is what makes the kernel reusable
    /// by schedules that choose sites differently, like [`SiteCheckerboard`].
    fn sweep(
        &self,
        config: &mut Configuration<2>,
        lattice: &Lattice<D>,
        action: &impl Action<2, D>,
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

/// The single-spin-flip Metropolis update with a **checkerboard** schedule: a
/// stateless unit struct, implementing [`Updater`] for `Q = 2` in any dimension.
///
/// A sweep updates every site of one color, then every site of the other, where
/// a site's color is the parity of its coordinate sum. The accept/reject rule is
/// identical to [`Metropolis`] — only the order differs — so on CPU it samples
/// the same distribution. Its purpose is to be the sequential reference for a
/// parallel sweep: within one color no two sites are neighbors, so a whole color
/// can be updated at once (on a GPU) without any site seeing another change.
///
/// That non-adjacency holds only when every extent is even; on an odd extent the
/// periodic wrap puts two same-color sites next to each other. This matters only
/// for a *parallel* sweep — here, updating sequentially, any site order is a
/// valid Metropolis schedule, so the CPU version is correct on any lattice.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SiteCheckerboard;

impl<const D: usize> Updater<2, D> for SiteCheckerboard {
    /// Two color passes: every site of color 0, then every site of color 1, each
    /// a single-site `step`. Together they attempt one update per site, so a
    /// checkerboard sweep does the same `n_vars` updates a [`Metropolis`] sweep
    /// does — only the order differs.
    fn sweep(
        &self,
        config: &mut Configuration<2>,
        lattice: &Lattice<D>,
        action: &impl Action<2, D>,
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

/// The single-link-flip Metropolis update with a **checkerboard** schedule for a
/// gauge model: a stateless unit struct, implementing [`Updater`] for `Q = 2` in
/// any dimension.
///
/// A link is a base site and a direction, and a sweep colors it by the pair
/// `(direction, parity of the base site's coordinate sum)` — `2D` colors, each
/// updated fully before the next. Splitting by direction is what base-site parity
/// alone cannot do: the four links of a plaquette include two that share a base
/// site, so a rule reading only the site would color them alike. Fixing the
/// direction freezes every link of another direction, and a plaquette then holds
/// exactly two links of the pass's direction, whose base sites differ by one step
/// and so carry opposite parity.
///
/// The accept/reject rule is [`Metropolis`]'s, so on CPU this samples the same
/// distribution and differs only in autocorrelation. Its purpose is to be the
/// sequential reference for a parallel sweep: within one color no two links share
/// a plaquette, so a whole color could be updated at once without any link seeing
/// another change.
///
/// That independence holds only when every extent is even; on an odd extent the
/// periodic wrap puts two links of a shared plaquette back in the same color.
/// This matters only for a *parallel* sweep — here, updating sequentially, any
/// link order is a valid Metropolis schedule, so the CPU version is correct on
/// any lattice. See `docs/metropolis.md` for the full argument.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinkCheckerboard;

impl LinkCheckerboard {
    /// Colors in one sweep: a direction paired with a base-site parity.
    ///
    /// Unlike the site coloring's two, this grows with the dimension, and the
    /// number is shared rather than derived twice — `GpuGaugeChain` turns it
    /// into dispatches per sweep and has to agree with the order
    /// [`sweep`](LinkCheckerboard::sweep) walks, since the CPU schedule is the
    /// reference the device kernel is checked against.
    pub(crate) const fn colors<const D: usize>() -> usize {
        2 * D
    }
}

impl<const D: usize> Updater<2, D> for LinkCheckerboard {
    /// `2D` color passes: for each direction in turn, every link of that
    /// direction based on an even site, then every one based on an odd site, each
    /// a single-variable `step`. Together they attempt one update per link, so a
    /// sweep does the same `n_vars` updates a [`Metropolis`] sweep does — only
    /// the order differs.
    ///
    /// # Panics
    ///
    /// Panics if `config` is not a link field, since the schedule reads each
    /// variable's index as a link.
    fn sweep(
        &self,
        config: &mut Configuration<2>,
        lattice: &Lattice<D>,
        action: &impl Action<2, D>,
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
                // Iterate over *sites*, not links, and address the one link each
                // site owns in this direction. Scanning every link and skipping
                // those of another direction would walk `D * n_sites` entries
                // per pass to perform `n_sites` updates. The visiting order is
                // the same either way, since the packing is monotonic in the base
                // site at fixed direction — `link_colors_are_visited_in_link_order`
                // pins that — and it is the mapping the GPU kernel launches one
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
/// The types are fixed at compile time, but which one a run uses is a value read
/// from a file — so the two have to meet at a single type. `AnyUpdater` is that
/// type: it implements [`Updater`] by forwarding `sweep` to whichever updater it
/// wraps, which is what lets [`IsingSampler`](crate::ising_sampler::IsingSampler) hold one field
/// and [`Chain`](crate::chain::Chain) stay generic while the algorithm is chosen
/// at runtime. Its variants mirror
/// [`UpdaterKind`](crate::config::UpdaterKind) — a closed set, which is what
/// makes the choice recordable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnyUpdater {
    /// The random-site schedule, [`Metropolis`].
    Metropolis(Metropolis),
    /// The site checkerboard schedule, [`SiteCheckerboard`].
    SiteCheckerboard(SiteCheckerboard),
    /// The link checkerboard schedule, [`LinkCheckerboard`].
    LinkCheckerboard(LinkCheckerboard),
}

impl<const D: usize> Updater<2, D> for AnyUpdater {
    /// Forward to the wrapped updater's sweep. The match is the whole cost of
    /// runtime dispatch — one branch per algorithm, resolved once per sweep.
    fn sweep(
        &self,
        config: &mut Configuration<2>,
        lattice: &Lattice<D>,
        action: &impl Action<2, D>,
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

/// Propose the single-site flip `0 ↔ 1`.
///
/// The one `Q`-specific piece of the update, isolated so a general-`Q` proposal
/// can replace it without touching the accept/reject core. The deterministic
/// two-state flip does not use `rng`, but a general-`Q` proposal would draw its
/// candidate from it, so it stays in the signature.
fn propose(current: State<2>, _rng: &mut impl Rng) -> State<2> {
    State::new(1 - current.index()).expect("1 - index is 0 or 1, always < Q")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Ising, Z2Gauge};
    use crate::rng::RandRng;

    /// A scripted [`Rng`] handing back preset answers, so a test can pin whether
    /// an accept draw passes (and, for a random-site schedule, which site is
    /// visited). The consumption counters double as an assertion target for how
    /// many draws were made.
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

    /// A downhill flip is accepted with no accept/reject draw, mutates the
    /// config, and returns a realized ΔE equal to the from-scratch difference.
    #[test]
    fn accepts_a_downhill_flip_without_drawing() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);

        // One spin flipped against an otherwise-aligned background: flipping it
        // back to the aligned state lowers the energy.
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

    /// An uphill flip whose draw falls above the Boltzmann factor is rejected:
    /// the config is untouched and the realized ΔE is 0.0.
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

    /// A Metropolis sweep is exactly `N = n_vars` steps: on the ground state
    /// every flip is uphill, so each step draws one accept uniform, and rejecting
    /// all of them leaves the config untouched with a net ΔE of 0.0.
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

    /// The net ΔE a Metropolis sweep returns equals `H(after) − H(before)`. This
    /// holds for *any* accept/reject pattern, so a real seeded RNG is fine;
    /// integer-valued couplings and sums make the comparison bit-exact.
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

    /// A checkerboard sweep attempts one update per site: on the ground state
    /// every flip is uphill, so each of the `n_vars` attempts draws exactly one
    /// accept uniform. The random-site generator is never touched — the schedule
    /// is deterministic — and rejecting all leaves the config untouched.
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

    /// The net ΔE a checkerboard sweep returns equals `H(after) − H(before)`, the
    /// same telescoping identity as the Metropolis sweep — the schedule differs
    /// but the accounting does not.
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

    /// The two colors partition the lattice: every site is exactly one color, and
    /// on an even lattice no site shares a color with any of its neighbors — the
    /// property a parallel sweep relies on.
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
    /// base site's parity, the `2D`-valued rule the sweep's pass order walks.
    fn link_color<const D: usize>(lattice: &Lattice<D>, link: usize) -> (usize, usize) {
        (
            lattice.link_direction(link),
            lattice.site_parity(lattice.link_site(link)),
        )
    }

    /// The link coloring is collision-free: no two links of the same color ever
    /// share a plaquette. This is the property a parallel gauge sweep would rely
    /// on, and the reason base-site parity alone is not enough — within a
    /// plaquette, two links do share a base site, and only the direction half of
    /// the pair separates them.
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

    /// The link coloring stays collision-free at every dimension.
    ///
    /// This is the property a parallel pass rests on, and it grows with `D` in a
    /// way the fixed-dimension test above cannot show: a link gains plaquette
    /// partners linearly while the coloring's own `2D` colors grow alongside. A
    /// coloring that stopped separating partners in higher dimensions would
    /// break detailed balance on the GPU silently — every measured quantity
    /// would still look reasonable.
    ///
    /// Only the link half is here. The site coloring is `Lattice::site_parity`
    /// rather than anything this module owns, and
    /// `lattice::parity_alternates_between_neighbors_in_every_dimension` already
    /// sweeps it over a superset of these shapes.
    ///
    /// Every extent is even, which the wrap requires: an odd one puts a variable
    /// next to a same-colored copy of itself across the boundary.
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

    /// A link pass covers every link exactly once, at every dimension.
    ///
    /// The counterpart of the collision-free property: a coloring that separated
    /// partners but skipped or double-counted variables would also break the
    /// chain, and the two failures are independent. Only the link pass is worth
    /// asserting — the site pass partitions the sites by a two-valued function,
    /// so reproducing the whole range is arithmetic rather than a property of
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
    /// scan would, in the same order — the property that lets it address one
    /// link per site instead of scanning every link and discarding those of
    /// another direction. Both orders are valid schedules, but pinning them
    /// equal is what makes that an optimization rather than a change.
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

    /// A link checkerboard sweep attempts one update per link: on the ground
    /// state every flip is uphill, so each of the `n_vars` attempts draws exactly
    /// one accept uniform. The random-variable generator is never touched — the
    /// schedule is deterministic — and rejecting all leaves the config untouched.
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

    /// The net ΔE a link checkerboard sweep returns equals `H(after) − H(before)`
    /// — the same telescoping identity the site schedules satisfy, on a link
    /// field.
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
    /// independence fails there — the periodic wrap puts two links of a shared
    /// plaquette in one color — but updating in sequence makes any order a valid
    /// Metropolis schedule, so the sweep still runs and still accounts correctly.
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

    /// A site field is rejected rather than silently misread: the schedule treats
    /// every index as a link, so running it on sites would color by a direction
    /// the variables do not have.
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
