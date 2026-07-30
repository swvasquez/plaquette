//! Updater: the rule that advances the Markov chain one move at a time.
//!
//! Where the [`Action`] says what a configuration *costs*, the updater says how
//! the chain *moves*, so that the configurations visited over a long run are
//! Boltzmann-distributed. It is the seam the driver depends on: a driver calls
//! [`step`](Updater::step) or [`sweep`](Updater::sweep) without naming an
//! algorithm, so heat-bath, cluster, or HMC moves can replace Metropolis without
//! touching it.
//!
//! [`Metropolis`] is the single-spin-flip update, the `Q = 2` case for any `D`.
//! One step picks a site uniformly, proposes the flipped state, prices the move
//! with [`Action::energy_delta`], and accepts with `min(1, e^{-β ΔE})`. The
//! uniform pick is what makes the proposal symmetric and so lets acceptance drop
//! the Hastings ratio; pricing by local delta rather than absolute energy is
//! what cancels the partition function. Uphill moves surviving with probability
//! `e^{-β ΔE} > 0` is mandatory, not an optimization — it is what lets the chain
//! climb out of local minima.
//!
//! The updater holds no chain state: the [`Configuration`] *is* the current
//! state, mutated in place, and `β` is passed per call so one updater serves a
//! whole temperature scan. It keeps no running energy either, since the
//! configuration already determines its own and measurements are infrequent
//! enough for a driver to recompute [`Action::energy`] at sample points.
//! [`step`](Updater::step) still returns the realized ΔE so a driver can
//! accumulate it if profiling ever justifies it.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::Action;
use crate::rng::Rng;
use crate::state::State;

/// The rule that advances the Markov chain by one move.
///
/// Generic over the field's state count `Q` and the lattice dimension `D`, so
/// the driver can name the seam without naming a specific algorithm — the peer
/// of [`Action`] on the sampling side.
pub trait Updater<const Q: usize, const D: usize> {
    /// Attempt one single-site update of `config` in place, returning the
    /// *realized* energy change: `ΔE` if the move was accepted, `0.0` if it was
    /// rejected (the configuration is then unchanged). `beta` is passed per call
    /// so one updater serves a whole temperature scan.
    fn step(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64;

    /// One **sweep**: `N = n_sites` single-site [`step`](Updater::step)s at random
    /// sites, returning the net realized `ΔE` summed over them.
    ///
    /// A sweep is the conventional unit of Monte Carlo time, sized to the lattice
    /// so autocorrelation is measured in sweeps rather than raw steps. The
    /// default composes `step`; an algorithm that sweeps differently (a
    /// full-lattice HMC trajectory, say) overrides it.
    ///
    /// The sum telescopes to `H(after) − H(before)` for this sweep alone. It is
    /// not a running energy — re-anchoring against a from-scratch
    /// [`Action::energy`] stays the driver's job.
    fn sweep(
        &self,
        config: &mut Configuration<Q>,
        lattice: &Lattice<D>,
        action: &impl Action<Q, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        let mut net = 0.0;
        for _ in 0..config.n_sites() {
            net += self.step(config, lattice, action, beta, rng);
        }
        net
    }
}

/// The single-spin-flip Metropolis update: a stateless unit struct with no
/// parameters, implementing [`Updater`] for `Q = 2` in any dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metropolis;

impl<const D: usize> Updater<2, D> for Metropolis {
    fn step(
        &self,
        config: &mut Configuration<2>,
        lattice: &Lattice<D>,
        action: &impl Action<2, D>,
        beta: f64,
        rng: &mut impl Rng,
    ) -> f64 {
        let site = rng.next_below(config.n_sites());
        let proposed = propose(config.peek(site), rng);
        let delta = action.energy_delta(lattice, config, site, proposed);

        // Accept with min(1, e^{-β ΔE}). The `ΔE ≤ 0` short-circuit keeps
        // downhill moves without a draw and, by keeping the argument to `exp`
        // non-positive, also prevents overflow.
        if delta <= 0.0 || rng.next_f64() < (-beta * delta).exp() {
            config.poke(site, proposed);
            delta
        } else {
            0.0
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
    use crate::model::Ising;
    use crate::rng::RandRng;

    /// A scripted [`Rng`] handing back preset answers, so a test can pin which
    /// site a step visits and whether its accept draw passes. The consumption
    /// counters double as an assertion target for how many draws were made.
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
        let mut config = Configuration::<2>::cold(&lat);
        let site = 5;
        config.poke(site, State::new(1).unwrap());

        let proposed = propose(config.peek(site), &mut ScriptedRng::new(vec![], vec![]));
        let expected = action.energy_delta(&lat, &config, site, proposed);
        assert!(expected < 0.0, "setup should make the flip downhill");
        let before = action.energy(&lat, &config);

        let mut rng = ScriptedRng::new(vec![site], vec![]);
        let realized = Metropolis.step(&mut config, &lat, &action, 1.0, &mut rng);

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
        let mut config = Configuration::<2>::cold(&lat); // ground state: every flip uphill
        let untouched = config.clone();

        // β = 1, ΔE = +8 ⇒ e^{-βΔE} ≈ 3.4e-4; a draw of 0.5 is far above it.
        let mut rng = ScriptedRng::new(vec![5], vec![0.5]);
        let realized = Metropolis.step(&mut config, &lat, &action, 1.0, &mut rng);

        assert_eq!(realized, 0.0);
        assert_eq!(config, untouched);
    }

    /// An uphill flip whose draw falls below the Boltzmann factor is accepted —
    /// the mandatory uphill move.
    #[test]
    fn accepts_an_uphill_flip_below_the_boltzmann_factor() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat);
        let site = 5;

        let proposed = propose(config.peek(site), &mut ScriptedRng::new(vec![], vec![]));
        let expected = action.energy_delta(&lat, &config, site, proposed);
        assert!(expected > 0.0, "setup should make the flip uphill");
        let before = action.energy(&lat, &config);

        // β = 0.25, ΔE = +8 ⇒ e^{-βΔE} = e^{-2} ≈ 0.135; a draw of 0.1 is below it.
        let mut rng = ScriptedRng::new(vec![site], vec![0.1]);
        let realized = Metropolis.step(&mut config, &lat, &action, 0.25, &mut rng);

        assert_eq!(realized, expected);
        assert_eq!(config.peek(site), proposed);
        assert_eq!(realized, action.energy(&lat, &config) - before);
    }

    /// A sweep is exactly `N = n_sites` steps: on the ground state every flip is
    /// uphill, so each step draws one accept uniform, and rejecting all of them
    /// leaves the config untouched with a net ΔE of 0.0.
    #[test]
    fn sweep_runs_n_sites_steps() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat); // ground state: all flips uphill
        let untouched = config.clone();
        let n = config.n_sites();

        // β = 1 ⇒ e^{-βΔE} ≈ 3.4e-4; a 0.9 draw rejects every step.
        let mut rng = ScriptedRng::new((0..n).collect(), vec![0.9; n]);
        let net = Metropolis.sweep(&mut config, &lat, &action, 1.0, &mut rng);

        assert_eq!(net, 0.0);
        assert_eq!(config, untouched);
        assert_eq!(rng.site_i, n, "sweep must attempt exactly n_sites steps");
        assert_eq!(
            rng.unif_i, n,
            "each uphill step draws exactly one accept uniform"
        );
    }

    /// The net ΔE a sweep returns equals `H(after) − H(before)`. This holds for
    /// *any* accept/reject pattern, so a real seeded RNG is fine; integer-valued
    /// couplings and sums make the comparison bit-exact.
    #[test]
    fn sweep_net_delta_equals_energy_change() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5); // exactly representable; sums stay integer
        let mut config = Configuration::<2>::hot(&lat, &mut RandRng::seed_from_u64(7));
        let before = action.energy(&lat, &config);

        let mut rng = RandRng::seed_from_u64(99);
        let net = Metropolis.sweep(&mut config, &lat, &action, 0.6, &mut rng);

        assert_eq!(net, action.energy(&lat, &config) - before);
    }
}
