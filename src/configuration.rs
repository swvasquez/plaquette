//! Configuration: the mutable field state living on a [`Lattice`].
//!
//! A `Configuration<Q>` is the `N` per-site variables of a field valued in a
//! finite set of `Q` states (see [`State`]), stored as a flat array indexed by
//! the lattice's own linear site index. It does **not** own or reference the
//! lattice — geometry is shared context passed alongside — so many
//! configurations (replicas, checkpoints, hot/cold starts) can run against a
//! single shared lattice.
//!
//! Access follows the lattice-field-theory convention: [`peek`](Configuration::peek)
//! reads a site's variable, [`poke`](Configuration::poke) writes one.

use crate::lattice::Lattice;
use crate::rng::Rng;
use crate::state::State;

/// The `N = n_sites` per-site variables as mutable state, each one of `Q`
/// states. Variable `i` sits at lattice site `i`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Configuration<const Q: usize> {
    /// One variable per site, indexed by linear site index.
    variables: Vec<State<Q>>,
}

impl<const Q: usize> Configuration<Q> {
    /// A **cold** start: every variable set to state `0` (an ordered state).
    /// The lattice is borrowed only for its site count.
    pub fn cold<const D: usize>(lattice: &Lattice<D>) -> Self {
        let ground = State::new(0).expect("Q must be >= 1");
        Configuration {
            variables: vec![ground; lattice.n_sites()],
        }
    }

    /// A **hot** start: each variable drawn independently and uniformly from the
    /// `Q` states (a disordered state). The `rng` is injected so the caller
    /// controls the source and can seed it for reproducibility.
    pub fn hot<const D: usize>(lattice: &Lattice<D>, rng: &mut impl Rng) -> Self {
        let variables = (0..lattice.n_sites())
            .map(|_| State::new(rng.next_below(Q)).expect("next_below(Q) < Q"))
            .collect();
        Configuration { variables }
    }

    /// The number of variables — the site count of the lattice it was built on.
    pub fn n_sites(&self) -> usize {
        self.variables.len()
    }

    /// Read the variable at `site` (`peekSite` in QDP++/Grid).
    pub fn peek(&self, site: usize) -> State<Q> {
        self.variables[site]
    }

    /// Write `value` into `site` (`pokeSite` in QDP++/Grid).
    pub fn poke(&mut self, site: usize, value: State<Q>) {
        self.variables[site] = value;
    }

    /// All variables in site order, for whole-lattice scans (total energy,
    /// magnetization).
    pub fn variables(&self) -> &[State<Q>] {
        &self.variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_is_all_ground_state() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat);
        assert_eq!(config.n_sites(), lat.n_sites());
        assert!(config.variables().iter().all(|s| s.index() == 0));
    }

    #[test]
    fn hot_start_stays_within_the_state_set() {
        let lat = Lattice::new([8, 8]);
        let mut rng = crate::rng::RandRng::seed_from_u64(0);
        // Q = 3 exercises the finite-set generality.
        let config = Configuration::<3>::hot(&lat, &mut rng);
        assert_eq!(config.n_sites(), lat.n_sites());
        assert!(config.variables().iter().all(|s| s.index() < 3));
    }

    #[test]
    fn hot_start_is_reproducible_from_seed() {
        let lat = Lattice::new([8, 8]);
        let mut rng_a = crate::rng::RandRng::seed_from_u64(123);
        let mut rng_b = crate::rng::RandRng::seed_from_u64(123);
        let a = Configuration::<2>::hot(&lat, &mut rng_a);
        let b = Configuration::<2>::hot(&lat, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn accessors_report_size_and_variables() {
        let lat = Lattice::new([3, 3]);
        let config = Configuration::<2>::cold(&lat);
        assert_eq!(config.n_sites(), 9);
        assert_eq!(config.variables().len(), 9);
        assert_eq!(config.peek(4).index(), 0);
    }

    #[test]
    fn poke_writes_a_single_site() {
        let lat = Lattice::new([3, 3]);
        let mut config = Configuration::<2>::cold(&lat);
        config.poke(4, State::new(1).unwrap());
        assert_eq!(config.peek(4).index(), 1);
        assert_eq!(config.peek(0).index(), 0); // neighbors untouched
    }

    #[test]
    fn poke_round_trips_a_two_state_spin() {
        let lat = Lattice::new([3, 3]);
        let mut config = Configuration::<2>::cold(&lat);
        config.poke(4, State::new(1).unwrap());
        assert_eq!(config.peek(4).index(), 1);
        config.poke(4, State::new(0).unwrap());
        assert_eq!(config.peek(4).index(), 0); // written back to the ground state
    }
}
