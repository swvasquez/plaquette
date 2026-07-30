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
//! [`Checkerboard`] keeps that accept/reject rule and changes only the order the
//! sites are visited in: it colors each site by coordinate-sum parity and updates
//! one color fully before the other, so no two sites updated together are
//! neighbors. On CPU that is just Metropolis in a fixed order, sampling the same
//! distribution; its purpose is to be the sequential form of a parallel (GPU)
//! sweep, where a whole color updates at once. See `docs/metropolis.md` for why
//! the reordering samples the same distribution and when the coloring's
//! independence holds.
//!
//! The updater holds no chain state: the [`Configuration`] *is* the current
//! state, mutated in place, and `β` is passed per call so one updater serves a
//! whole temperature scan. It keeps no running energy either, since the
//! configuration already determines its own and measurements are infrequent
//! enough for a driver to recompute [`Action::energy`] at sample points.
//! [`sweep`](Updater::sweep) still returns the net realized ΔE so a driver can
//! accumulate it if profiling ever justifies it.

use crate::configuration::Configuration;
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
    /// `n_vars` single-site updates at random sites; [`Checkerboard`] attempts
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

/// The single-spin-flip Metropolis update at a **given** site: propose the flip,
/// price it with [`Action::energy_delta`], and accept with `min(1, e^{-β ΔE})`,
/// returning the realized `ΔE` (`0.0` on rejection, leaving `config` unchanged).
///
/// This is the shared single-site kernel, with the *site handed in* rather than
/// chosen here — site selection is the schedule's job, not the kernel's. Both
/// [`Metropolis`] (random sites) and [`Checkerboard`] (color order) drive their
/// sweeps by calling this; the two differ only in which sites they pass and in
/// what order.
fn step<const D: usize>(
    config: &mut Configuration<2>,
    lattice: &Lattice<D>,
    action: &impl Action<2, D>,
    site: usize,
    beta: f64,
    rng: &mut impl Rng,
) -> f64 {
    let proposed = propose(config.peek(site), rng);
    let delta = action.energy_delta(lattice, config, site, proposed);

    // Accept with min(1, e^{-β ΔE}). The `ΔE ≤ 0` short-circuit keeps downhill
    // moves without a draw and, by keeping the argument to `exp` non-positive,
    // also prevents overflow.
    if delta <= 0.0 || rng.next_f64() < (-beta * delta).exp() {
        config.poke(site, proposed);
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
    /// by schedules that choose sites differently, like [`Checkerboard`].
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
pub struct Checkerboard;

impl<const D: usize> Updater<2, D> for Checkerboard {
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
                if parity(lattice, site) == color {
                    net += step(config, lattice, action, site, beta, rng);
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
    /// The checkerboard schedule, [`Checkerboard`].
    Checkerboard(Checkerboard),
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
            AnyUpdater::Checkerboard(u) => u.sweep(config, lattice, action, beta, rng),
        }
    }
}

/// A site's checkerboard color: the parity of its coordinate sum, `0` or `1`.
///
/// Two sites of the same color are never nearest neighbors on an even lattice,
/// because a single step along any axis changes exactly one coordinate by one and
/// so flips the parity.
fn parity<const D: usize>(lattice: &Lattice<D>, site: usize) -> usize {
    lattice.site_coords(site).iter().sum::<usize>() % 2
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
    use crate::configuration::Cell;
    use crate::model::Ising;
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
    fn checkerboard_sweep_attempts_every_site_once() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site); // ground state: all flips uphill
        let untouched = config.clone();
        let n = config.n_vars();

        // β = 1 ⇒ e^{-βΔE} ≈ 3.4e-4; a 0.9 draw rejects every attempt. No site
        // draws are scripted, so a stray next_below would panic.
        let mut rng = ScriptedRng::new(vec![], vec![0.9; n]);
        let net = Checkerboard.sweep(&mut config, &lat, &action, 1.0, &mut rng);

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
    fn checkerboard_sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let mut config = Configuration::<2>::hot(&lat, Cell::Site, &mut RandRng::seed_from_u64(7));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(99);
        let net = Checkerboard.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }

    /// The two colors partition the lattice: every site is exactly one color, and
    /// on an even lattice no site shares a color with any of its neighbors — the
    /// property a parallel sweep relies on.
    #[test]
    fn colors_partition_and_separate_neighbors() {
        let lat = Lattice::new([4, 4]);
        for site in 0..lat.n_sites() {
            let c = parity(&lat, site);
            for &nbr in lat.site_neighbors(site) {
                assert_ne!(c, parity(&lat, nbr), "neighbors must differ in color");
            }
        }
    }
}
