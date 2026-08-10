//! The built-in models, one submodule each — its action, its observables, its
//! run-config schema, and its samplers, end to end.
//!
//! Everything below this module is meaning-free ([`State`] an index,
//! [`Configuration`] a flat array of indices, [`Lattice`] geometry alone); a
//! model is the first place that says what the indices *are*. This level holds
//! only what spans models — `decode`, `axis_pair_sums`, and [`AnyAction`] —
//! and nothing at the crate root reaches back down into this module.

pub mod gauge;
pub mod ising;
pub mod potts;

use crate::action::Action;
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::state::State;
use gauge::Z2Gauge;
use ising::Ising;

/// Map a two-state index to the value it stands for: `0 → +1`, `1 → −1`.
///
/// One function rather than a copy per model, and not by coincidence: an Ising
/// spin and a Z2 gauge link are the same variable, an element of `{+1, −1}`
/// under multiplication, moved from the sites to the links.
// TODO: value semantics belong to a type, not a function — prior art (Grid,
// QDP++) carries a per-cell value type owning its own product. Build that when
// a second value appears that is not `±1`, since one example cannot decide its
// shape; until then this stays inside `models` so the move costs one file.
pub(crate) fn decode(state: State<2>) -> i32 {
    1 - 2 * state.index() as i32
}

/// Sum `pair(s_i, s_{i+r})` over every site, for each axis and each displacement
/// `r = 0..=L_mu/2` — the scan both models' correlators are built from.
///
/// The two differ only in what one pair contributes: a product of signs for
/// [`Ising`], an equality test for [`Potts`](potts::Potts). The shared part is
/// what is easy to get subtly wrong twice — half of each axis suffices because
/// `C_r = C_{L_mu - r}`, and the accumulation stays in an integer so the final
/// scaling is the only rounding. The counts come back raw; what to divide by,
/// and whether to subtract a floor, differs between the two conventions and is
/// the caller's.
pub(crate) fn axis_pair_sums<const Q: usize, const D: usize>(
    lattice: &Lattice<D>,
    config: &Configuration<Q>,
    pair: impl Fn(State<Q>, State<Q>) -> i64,
) -> [Vec<i64>; D] {
    let shape = lattice.shape();
    let mut sums: [Vec<i64>; D] = std::array::from_fn(|mu| vec![0i64; shape[mu] / 2 + 1]);
    for site in 0..config.n_vars() {
        let s_i = config.peek(site);
        for (mu, row) in sums.iter_mut().enumerate() {
            for (r, cell) in row.iter_mut().enumerate() {
                *cell += pair(s_i, config.peek(lattice.site_shift(site, mu, r)));
            }
        }
    }
    sums
}

/// A runtime choice among the built-in actions, so a model named in a config
/// file can be selected without the caller committing to a type at compile
/// time — the counterpart of [`AnyUpdater`](crate::updater::AnyUpdater).
///
/// It unifies the energy seam and only that: the models do not measure the same
/// quantities, so a caller that measures still branches on the model via
/// [`ising::measure`] and [`gauge::gauge_measure`].
///
/// [`Potts`](potts::Potts) is absent, and not by oversight: it is generic over
/// `Q`, which the other two satisfy only at two, so admitting it would mean
/// carrying `Q` on the enum.
///
/// Nothing consumes it yet, by design. The Ising and gauge runtimes are
/// separate siblings today; this is the seam the eventual unified runtime will
/// hold, built ahead of that consumer only because `AnyUpdater` has already
/// fixed its shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyAction {
    /// The site model, [`Ising`].
    Ising(Ising),
    /// The link model, [`Z2Gauge`].
    Z2Gauge(Z2Gauge),
}

impl AnyAction {
    /// The cell kind a configuration must sit on for this action to price it.
    ///
    /// A caller building a configuration for an action picked at runtime has no
    /// other way to ask, and the `debug_assert!`s in the energies are compiled
    /// out of a release build.
    pub fn cell(&self) -> Cell {
        match self {
            AnyAction::Ising(_) => Cell::Site,
            AnyAction::Z2Gauge(_) => Cell::Link,
        }
    }
}

impl<const D: usize> Action<2, D> for AnyAction {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        match self {
            AnyAction::Ising(action) => action.energy(lattice, config),
            AnyAction::Z2Gauge(action) => action.energy(lattice, config),
        }
    }

    /// The branch here is per proposed flip — the sampler's hot path — but
    /// predictable, since a run never switches model mid-chain.
    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        var: usize,
        proposed: State<2>,
    ) -> f64 {
        match self {
            AnyAction::Ising(action) => action.energy_delta(lattice, config, var, proposed),
            AnyAction::Z2Gauge(action) => action.energy_delta(lattice, config, var, proposed),
        }
    }
}

/// How many variables `deltas_match_from_scratch` probes on a large lattice.
///
/// Each probe costs a whole-lattice energy scan, so checking every variable is
/// quadratic in their count; past the cap the probes are spread evenly across
/// the index range, which still reaches every direction and both parities.
#[cfg(test)]
const MAX_PROBES: usize = 64;

/// Flip variables of `config` in turn and check the incremental energy
/// against the difference of two from-scratch energies.
///
/// The sharpest check that a model carries no assumption about the dimension:
/// `energy` walks cells by index while `energy_delta` walks the incidence
/// tables, so a `D`-specific mistake anywhere in the packing, strides, or
/// neighbor and staple rows shows up as a mismatch on some variable rather
/// than as a plausible wrong number. It lives here because every model's test
/// module runs it over its own action.
#[cfg(test)]
pub(crate) fn deltas_match_from_scratch<const Q: usize, const D: usize, A: Action<Q, D>>(
    shape: [usize; D],
    action: &A,
    cell: Cell,
) {
    let lat = Lattice::new(shape);
    let mut config = Configuration::<Q>::cold(&lat, cell);
    // A non-uniform configuration, so the neighbor and staple sums vary;
    // stepping the state by the variable index spreads all `Q` states over
    // the field.
    for var in (0..config.n_vars()).step_by(3) {
        config.poke(var, State::new(1 + var % (Q - 1)).unwrap());
    }

    // Each probe clones `config`, so the reference energy is computed once.
    let before = action.energy(&lat, &config);
    let n_vars = config.n_vars();
    let step = n_vars.div_ceil(MAX_PROBES).max(1);
    for var in (0..n_vars).step_by(step) {
        // The next state cyclically: a real change at every `Q`, the plain
        // flip at two.
        let proposed = State::new((config.peek(var).index() + 1) % Q).unwrap();
        let delta = action.energy_delta(&lat, &config, var, proposed);

        let mut after = config.clone();
        after.poke(var, proposed);
        // The couplings below are exactly representable and the sums are
        // integer, so the two sides agree bit-for-bit.
        assert_eq!(
            delta,
            action.energy(&lat, &after) - before,
            "{shape:?}: variable {var}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::potts::Potts;
    use super::*;

    #[test]
    fn decode_maps_to_plus_minus_one() {
        assert_eq!(decode(State::new(0).unwrap()), 1);
        assert_eq!(decode(State::new(1).unwrap()), -1);
    }

    #[test]
    fn any_action_forwards_to_the_wrapped_action() {
        // Same lattice, same field, same numbers on both seams: the enum adds
        // a branch and nothing else.
        let lat = Lattice::new([4, 4]);
        let down = State::new(1).unwrap();

        let ising = Ising::new(1.0, 0.5);
        let mut sites = Configuration::<2>::cold(&lat, Cell::Site);
        sites.poke(5, down);
        let any = AnyAction::Ising(ising);
        assert_eq!(any.energy(&lat, &sites), ising.energy(&lat, &sites));
        assert_eq!(
            any.energy_delta(&lat, &sites, 6, down),
            ising.energy_delta(&lat, &sites, 6, down)
        );

        let gauge = Z2Gauge::new(1.0);
        let mut links = Configuration::<2>::cold(&lat, Cell::Link);
        links.poke(5, down);
        let any = AnyAction::Z2Gauge(gauge);
        assert_eq!(any.energy(&lat, &links), gauge.energy(&lat, &links));
        assert_eq!(
            any.energy_delta(&lat, &links, 6, down),
            gauge.energy_delta(&lat, &links, 6, down)
        );
    }

    #[test]
    fn any_action_reports_the_cell_its_field_must_sit_on() {
        assert_eq!(AnyAction::Ising(Ising::new(1.0, 0.0)).cell(), Cell::Site);
        assert_eq!(AnyAction::Z2Gauge(Z2Gauge::new(1.0)).cell(), Cell::Link);
    }

    /// The value-semantic half of the models, swept over dimensions:
    /// [`deltas_match_from_scratch`] covers the *incremental* seam, these cover
    /// what the observables read. Each configuration's answer is fixed by
    /// construction rather than by a reference run, so the assertions are exact
    /// at any dimension.
    fn observables_hold_on_known_configurations<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        let n_sites = lat.n_sites() as f64;

        // Ising: cold is the ground state — every forward bond aligned.
        let ising = Ising::new(1.5, 0.25);
        let cold = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(
            ising.energy(&lat, &cold),
            -1.5 * lat.n_links() as f64 - 0.25 * n_sites,
            "{shape:?}"
        );
        assert_eq!(ising.magnetization(&cold), n_sites, "{shape:?}");

        let correlator = ising.correlator(&lat, &cold);
        assert_eq!(correlator.len(), D, "{shape:?}: one row per axis");
        for (mu, row) in correlator.iter().enumerate() {
            assert_eq!(row.len(), shape[mu] / 2 + 1, "{shape:?}: axis {mu}");
            assert!(row.iter().all(|&c| c == 1.0), "{shape:?}: axis {mu}");
        }

        // Potts at three states: cold is the ground state here too. Its
        // correlator anchors at `1 - 1/Q` rather than at Ising's `1` because
        // the connected form subtracts the agreement floor (docs/potts.md,
        // "Measuring correlations").
        const Q: usize = 3;
        let potts = Potts::symmetric(1.5);
        let uniform = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(
            potts.energy(&lat, &uniform),
            -1.5 * lat.n_links() as f64,
            "{shape:?}"
        );
        assert_eq!(potts.order_parameter(&uniform), 1.0, "{shape:?}");

        let ceiling = 1.0 - 1.0 / Q as f64;
        let agreement = potts.correlator(&lat, &uniform);
        assert_eq!(agreement.len(), D, "{shape:?}: one row per axis");
        for (mu, row) in agreement.iter().enumerate() {
            assert_eq!(row.len(), shape[mu] / 2 + 1, "{shape:?}: axis {mu}");
            assert!(row.iter().all(|&c| c == ceiling), "{shape:?}: axis {mu}");
        }

        if D < 2 {
            return; // no plaquette below two dimensions, so nothing gauge-like
        }

        // Gauge: cold is the ground state — every plaquette at `+1`.
        let gauge = Z2Gauge::new(1.5);
        let cold = Configuration::<2>::cold(&lat, Cell::Link);
        assert_eq!(
            gauge.plaquette_sum(&lat, &cold),
            lat.n_plaquettes() as f64,
            "{shape:?}"
        );
        assert_eq!(
            gauge.energy(&lat, &cold),
            -1.5 * lat.n_plaquettes() as f64,
            "{shape:?}"
        );

        // Every Wilson rectangle and every Polyakov line on a cold field is one.
        let max_side = shape.iter().min().copied().unwrap_or(0) / 2;
        let table = gauge.wilson_rectangles(&lat, &cold, max_side);
        assert_eq!(table.len(), max_side + 1, "{shape:?}");
        for (r, row) in table.iter().enumerate() {
            for (t, &w) in row.iter().enumerate() {
                assert_eq!(w, 1.0, "{shape:?}: W({r},{t})");
                assert_eq!(w, table[t][r], "{shape:?}: table is symmetric");
            }
        }
        for dir in 0..D {
            assert_eq!(
                gauge.polyakov_loop(&lat, &cold, dir),
                1.0,
                "{shape:?}: {dir}"
            );
        }

        // A gauge transformation is a symmetry of the energy and of every
        // closed loop (docs/z2-gauge.md, "Gauge invariance") — the sharpest
        // check that `plaquette_links`, `loop_links`, and `site_links` all
        // agree about the geometry, and it says more the higher the dimension.
        let mut transformed = cold.clone();
        for site in (0..lat.n_sites()).step_by(3) {
            for link in lat.site_links(site) {
                let flipped = State::new(1 - transformed.peek(link).index()).unwrap();
                transformed.poke(link, flipped);
            }
        }
        // Flipping a site's links twice would put the configuration back;
        // without this the assertions below would hold vacuously.
        assert_ne!(
            transformed, cold,
            "{shape:?}: the transformation left the configuration alone"
        );
        assert_eq!(
            gauge.energy(&lat, &transformed),
            gauge.energy(&lat, &cold),
            "{shape:?}: gauge transformation changed the energy"
        );
        assert_eq!(
            gauge.wilson_rectangles(&lat, &transformed, max_side),
            table,
            "{shape:?}: gauge transformation changed the Wilson table"
        );
        for dir in 0..D {
            assert_eq!(
                gauge.polyakov_loop(&lat, &transformed, dir),
                1.0,
                "{shape:?}: gauge transformation changed the Polyakov loop along {dir}"
            );
        }
    }

    /// The observables hold at every dimension up to six — not the ten
    /// [`deltas_match_from_scratch`] reaches, because `wilson_rectangles` needs
    /// room for a rectangle, so every extent here is at least four and `4^10`
    /// sites is far past what belongs in a unit test.
    #[test]
    fn observables_hold_in_every_dimension() {
        observables_hold_on_known_configurations([6]);
        observables_hold_on_known_configurations([4, 6]);
        observables_hold_on_known_configurations([4, 6, 4]);
        observables_hold_on_known_configurations([4, 6, 4, 4]);
        observables_hold_on_known_configurations([4, 4, 6, 4, 4]);
        observables_hold_on_known_configurations([4, 4, 4, 4, 4, 4]);
    }
}
