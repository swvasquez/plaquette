//! Configuration: the mutable field state living on a [`Lattice`].
//!
//! A `Configuration<Q>` is a field of variables valued in a finite set of `Q`
//! states (see [`State`]), stored as a flat array in the lattice's own linear
//! index order. Which cells those variables sit on is part of what a
//! configuration *is* rather than something a reader supplies later: the Ising
//! model puts a spin on every site, a Z2 gauge theory a variable on every link,
//! and a configuration records that choice as its [`Cell`]. An action then says
//! how a field evolves, never what it is.
//!
//! It does not own or reference the lattice — geometry is shared context passed
//! alongside — so many configurations (replicas, checkpoints, hot/cold starts)
//! can run against a single shared lattice. The lattice is borrowed at
//! construction only, and only to be measured: a lattice and a cell kind fix
//! exactly one correct length, so the constructor derives it rather than taking
//! a count on trust.
//!
//! Access follows the lattice-field-theory convention:
//! [`peek`](Configuration::peek) reads one variable,
//! [`poke`](Configuration::poke) writes one.

use crate::lattice::Lattice;
use crate::rng::Rng;
use crate::state::State;

/// The kind of lattice cell a field's variables sit on.
///
/// Only the kinds that actually carry variables are named. A plaquette is where
/// a gauge energy is *evaluated* — the product of the four link variables around
/// a face — but nothing is stored on it, so it is not a field kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// One variable per lattice site, as in the Ising model.
    Site,
    /// One variable per forward link, as in a Z2 gauge theory.
    Link,
}

impl Cell {
    /// How many cells of this kind `lattice` has, and hence how many variables a
    /// field on it holds.
    pub fn count<const D: usize>(&self, lattice: &Lattice<D>) -> usize {
        match self {
            Cell::Site => lattice.n_sites(),
            Cell::Link => lattice.n_links(),
        }
    }
}

/// A field of variables as mutable state: one per cell of a single [`Cell`]
/// kind, each valued in `Q` states. Variable `i` sits at cell `i` of that kind,
/// in the lattice's index order.
///
/// A configuration carries one field and only one. A physical configuration
/// holding several at once — gauge variables on links together with matter on
/// sites — is a bundle of these rather than a wider container, so the atom stays
/// a single field that knows what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Configuration<const Q: usize> {
    /// Which kind of cell the variables sit on, and so what an index means.
    cell: Cell,
    /// One variable per cell of that kind, in linear index order.
    variables: Vec<State<Q>>,
}

impl<const Q: usize> Configuration<Q> {
    /// A **cold** start on `lattice`: one variable per `cell`, every one set to
    /// state `0` (an ordered state). The lattice is borrowed only to count its
    /// cells of that kind.
    pub fn cold<const D: usize>(lattice: &Lattice<D>, cell: Cell) -> Self {
        let ground = State::new(0).expect("Q must be >= 1");
        Configuration {
            cell,
            variables: vec![ground; cell.count(lattice)],
        }
    }

    /// A **hot** start on `lattice`: one variable per `cell`, each drawn
    /// independently and uniformly from the `Q` states (a disordered state). The
    /// `rng` is injected so the caller controls the source and can seed it for
    /// reproducibility.
    pub fn hot<const D: usize>(lattice: &Lattice<D>, cell: Cell, rng: &mut impl Rng) -> Self {
        let variables = (0..cell.count(lattice))
            .map(|_| State::new(rng.next_below(Q)).expect("next_below(Q) < Q"))
            .collect();
        Configuration { cell, variables }
    }

    /// Which kind of cell this field's variables sit on.
    pub fn cell(&self) -> Cell {
        self.cell
    }

    /// The number of variables — the count of the cells the field was built on.
    pub fn n_vars(&self) -> usize {
        self.variables.len()
    }

    /// Read the variable at `index` (`peekSite` in QDP++/Grid).
    pub fn peek(&self, index: usize) -> State<Q> {
        self.variables[index]
    }

    /// Write `value` into `index` (`pokeSite` in QDP++/Grid).
    pub fn poke(&mut self, index: usize, value: State<Q>) {
        self.variables[index] = value;
    }

    /// All variables in index order, for whole-lattice scans (total energy,
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
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(config.n_vars(), lat.n_sites());
        assert!(config.variables().iter().all(|s| s.index() == 0));
    }

    #[test]
    fn hot_start_stays_within_the_state_set() {
        let lat = Lattice::new([8, 8]);
        let mut rng = crate::rng::RandRng::seed_from_u64(0);
        // Q = 3 exercises the finite-set generality.
        let config = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
        assert_eq!(config.n_vars(), lat.n_sites());
        assert!(config.variables().iter().all(|s| s.index() < 3));
    }

    #[test]
    fn hot_start_is_reproducible_from_seed() {
        let lat = Lattice::new([8, 8]);
        let mut rng_a = crate::rng::RandRng::seed_from_u64(123);
        let mut rng_b = crate::rng::RandRng::seed_from_u64(123);
        let a = Configuration::<2>::hot(&lat, Cell::Site, &mut rng_a);
        let b = Configuration::<2>::hot(&lat, Cell::Site, &mut rng_b);
        assert_eq!(a, b);
    }

    #[test]
    fn accessors_report_size_and_variables() {
        let lat = Lattice::new([3, 3]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(config.n_vars(), 9);
        assert_eq!(config.variables().len(), 9);
        assert_eq!(config.peek(4).index(), 0);
    }

    #[test]
    fn poke_writes_a_single_variable() {
        let lat = Lattice::new([3, 3]);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        config.poke(4, State::new(1).unwrap());
        assert_eq!(config.peek(4).index(), 1);
        assert_eq!(config.peek(0).index(), 0); // neighbors untouched
    }

    #[test]
    fn poke_round_trips_a_two_state_spin() {
        let lat = Lattice::new([3, 3]);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        config.poke(4, State::new(1).unwrap());
        assert_eq!(config.peek(4).index(), 1);
        config.poke(4, State::new(0).unwrap());
        assert_eq!(config.peek(4).index(), 0); // written back to the ground state
    }

    #[test]
    fn a_link_field_is_sized_to_the_links_not_the_sites() {
        // The whole point of the cell kind: on the same lattice a link field is
        // D times a site field, and each reports which it is.
        let lat = Lattice::new([3, 3, 3]);
        let sites = Configuration::<2>::cold(&lat, Cell::Site);
        let links = Configuration::<2>::cold(&lat, Cell::Link);

        assert_eq!(sites.cell(), Cell::Site);
        assert_eq!(links.cell(), Cell::Link);
        assert_eq!(sites.n_vars(), lat.n_sites());
        assert_eq!(links.n_vars(), lat.n_links());
        assert_eq!(links.n_vars(), 3 * sites.n_vars());
    }

    #[test]
    fn fields_on_different_cells_are_never_equal() {
        // Both are all-ground-state arrays, so only the cell kind distinguishes
        // them — and it must, or a site field could pass for a link field on a
        // one-dimensional lattice, where the two counts coincide.
        let lat = Lattice::new([5]);
        let sites = Configuration::<2>::cold(&lat, Cell::Site);
        let links = Configuration::<2>::cold(&lat, Cell::Link);
        assert_eq!(sites.n_vars(), links.n_vars());
        assert_ne!(sites, links);
    }
}
