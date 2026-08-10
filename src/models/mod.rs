//! The built-in models, one submodule each — its action, its observables, its
//! run-config schema, and its samplers, end to end.
//!
//! Everything below this module is meaning-free: a [`State`] is an index, a
//! [`Configuration`] a flat array of indices, a [`Lattice`] geometry alone. A
//! model is the first place that says what the indices *are*, and each
//! submodule owns one model's whole answer. What lives at this level is only
//! what spans models rather than belonging to one: the two-state value
//! semantics [`ising`] and [`gauge`] share (`decode`), the correlator scan the
//! two site models share (`axis_pair_sums`), and [`AnyAction`], the runtime
//! choice among the actions. The seams, the geometry, and the statistics live
//! at the crate root, and nothing at the root reaches back down into this
//! module.

pub mod gauge;
pub mod ising;
pub mod potts;

use crate::action::Action;
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::state::State;
use gauge::Z2Gauge;
use ising::Ising;

/// Map a two-state index to the value it stands for: `0 → +1`, `1 → −1`. The
/// whole of the two-state value semantics, kept inside [`models`](self) for the
/// two models that read it and visible to nothing else.
///
/// It is named for the operation rather than for either model's variable,
/// because both read it and not by coincidence: an Ising spin and a Z2 gauge
/// link are the same variable, an element of the group `{+1, −1}` under
/// multiplication, moved from the sites to the links. One function records that
/// where a copy per model would obscure it.
// TODO: value semantics belong to a type, not a function — the decode is fixed
// by `Q`, not by the model reading it, which is why one function serves both.
// The prior art (Grid, QDP++) carries it as a per-cell value type owning its own
// product. Build that when a second value appears that is not `±1` — a Potts
// index, a `U(1)` phase, an `SU(N)` matrix — since one example cannot decide its
// shape. Until then this stays inside `models` so the move costs one file.
pub(crate) fn decode(state: State<2>) -> i32 {
    1 - 2 * state.index() as i32
}

/// Sum `pair(s_i, s_{i+r})` over every site, for each axis and each displacement
/// `r = 0..=L_mu/2` — the scan both models' correlators are built from.
///
/// The two differ only in what one pair contributes: a product of signs for
/// [`Ising`], an equality test for [`Potts`](potts::Potts). Everything else is
/// shared and easy
/// to get subtly wrong twice — which half of each axis is stored, that
/// `C_r = C_{L_mu - r}` is why the other half can be dropped, and that the
/// accumulation stays in an integer so the final scaling is the only rounding.
/// Stating it once leaves each model with its one-line kernel.
///
/// The counts are returned raw. What to divide by, and whether to subtract a
/// floor afterwards, is the caller's, since the two conventions differ there.
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
/// time.
///
/// The counterpart of [`AnyUpdater`](crate::updater::AnyUpdater), and the same
/// reasoning: the types are fixed at compile time but which one a run uses is a
/// value read from a file, so the two have to meet at a single type. The
/// variants are a closed set, which is what makes the choice recordable.
///
/// It unifies the energy seam and only that. The two models do not measure the
/// same quantities — [`Ising`] has a magnetization, [`Z2Gauge`] a plaquette sum
/// and Wilson loops — so a caller that measures still branches on the model
/// rather than reading through this enum; see each model's own measure
/// function, [`ising::measure`] and [`gauge::gauge_measure`].
///
/// [`Potts`](potts::Potts) is absent, and not by oversight. This enum implements
/// `Action<2, D>` because both its variants do, whereas `Potts` is generic over
/// `Q` and its `Q = 2` case is a different model from the one anyone runs it
/// for. Admitting it would mean carrying `Q` on the enum, which the other two
/// cannot satisfy at anything but two — so the third model waits for the same
/// unified runtime this is already waiting for.
///
/// Nothing consumes it yet, by design. The Ising and gauge runtimes are separate
/// siblings today — [`IsingSampler`](crate::models::ising::sampler::IsingSampler) and
/// [`GaugeSampler`](crate::models::gauge::sampler::GaugeSampler), each holding a concrete
/// model — and this is the seam the eventual *unified* runtime will hold, the way
/// `IsingSampler` already holds an [`AnyUpdater`](crate::updater::AnyUpdater):
/// dispatched from a `ModelKind` discriminant in [`config`](crate::config) that
/// would mirror [`UpdaterKind`](crate::config::UpdaterKind), and built ahead of
/// that consumer only because `AnyUpdater` has already fixed its shape.
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
    /// Choosing the model and choosing where its variables live is one decision,
    /// and a caller building a configuration for an action picked at runtime has
    /// no other way to ask. Without it the `debug_assert!`s in the energies are
    /// the only thing between a mismatched field and wrong physics, and those
    /// are compiled out of a release build.
    pub fn cell(&self) -> Cell {
        match self {
            AnyAction::Ising(_) => Cell::Site,
            AnyAction::Z2Gauge(_) => Cell::Link,
        }
    }
}

impl<const D: usize> Action<2, D> for AnyAction {
    /// Forward to the wrapped action. The match is the whole cost of runtime
    /// dispatch — one branch per model, and on this path only once per measured
    /// configuration.
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        match self {
            AnyAction::Ising(action) => action.energy(lattice, config),
            AnyAction::Z2Gauge(action) => action.energy(lattice, config),
        }
    }

    /// Forward to the wrapped action. This one sits in the sampler's hot path,
    /// so the branch is per proposed flip; it is predictable, since a run never
    /// switches model mid-chain.
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
/// Every variable is checked when a lattice is small enough, because that is
/// the strongest form of the test. Each probe costs a whole-lattice energy
/// scan, so checking all of them is quadratic in the variable count and a
/// ten-dimensional box runs into billions of operations under a debug build.
/// Past the cap the probes are spread evenly across the index range instead,
/// which still reaches every direction and both parities.
#[cfg(test)]
const MAX_PROBES: usize = 64;

/// Flip variables of `config` in turn and check the incremental energy
/// against the difference of two from-scratch energies.
///
/// This is the sharpest check that a model carries no assumption about the
/// dimension: `energy` walks cells by index while `energy_delta` walks the
/// incidence tables, so the two agree only if the packing, the strides, and
/// the neighbor and staple rows all say the same thing about the geometry.
/// A `D`-specific mistake anywhere in that chain shows up as a mismatch on
/// some variable rather than as a plausible wrong number.
///
/// It lives here rather than in any one model's tests because every model's
/// test module runs it over its own action.
#[cfg(test)]
pub(crate) fn deltas_match_from_scratch<const Q: usize, const D: usize, A: Action<Q, D>>(
    shape: [usize; D],
    action: &A,
    cell: Cell,
) {
    let lat = Lattice::new(shape);
    let mut config = Configuration::<Q>::cold(&lat, cell);
    // A non-uniform configuration, so the neighbor and staple sums vary
    // rather than all collapsing onto the ground state's. Stepping the state
    // by the variable index spreads all `Q` of them over the field, which at
    // `Q = 2` is the same alternating pattern this used to lay down.
    for var in (0..config.n_vars()).step_by(3) {
        config.poke(var, State::new(1 + var % (Q - 1)).unwrap());
    }

    // `config` is never mutated below — each probe clones it — so the
    // reference energy is computed once rather than per probe.
    let before = action.energy(&lat, &config);
    let n_vars = config.n_vars();
    let step = n_vars.div_ceil(MAX_PROBES).max(1);
    for var in (0..n_vars).step_by(step) {
        // The next state cyclically, which is a real change at every `Q` and
        // is the plain flip at two.
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
        // Same lattice, same field, same numbers on both seams: the enum adds a
        // branch and nothing else. One lattice serves both variants, since the
        // fields differ only in length — links are twice the sites in 2D.
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

    /// The value-semantic half of the models, swept over dimensions.
    ///
    /// [`deltas_match_from_scratch`] covers the *incremental* seam; these cover
    /// what the observables read. Each is checked on a configuration whose answer
    /// is fixed by construction rather than by a reference run, so the assertions
    /// are exact at any dimension: a cold field has every bond, plaquette, and
    /// loop at its extreme value, and a gauge transformation is a symmetry of the
    /// energy whatever the lattice looks like.
    fn observables_hold_on_known_configurations<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        let n_sites = lat.n_sites() as f64;

        // Ising: cold is the ground state, `E = -j * n_links - h * n_sites`,
        // since every one of the `D * n_sites` forward bonds is aligned.
        let ising = Ising::new(1.5, 0.25);
        let cold = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(
            ising.energy(&lat, &cold),
            -1.5 * lat.n_links() as f64 - 0.25 * n_sites,
            "{shape:?}"
        );
        assert_eq!(ising.magnetization(&cold), n_sites, "{shape:?}");

        // The correlator is one row per axis, each `L_mu / 2 + 1` long, and all
        // ones on a cold field including the `C_0` anchor.
        let correlator = ising.correlator(&lat, &cold);
        assert_eq!(correlator.len(), D, "{shape:?}: one row per axis");
        for (mu, row) in correlator.iter().enumerate() {
            assert_eq!(row.len(), shape[mu] / 2 + 1, "{shape:?}: axis {mu}");
            assert!(row.iter().all(|&c| c == 1.0), "{shape:?}: axis {mu}");
        }

        // Potts at three states: cold is the ground state here too, and there is
        // no field term to add to it. Its correlator anchors at `1 - 1/Q` rather
        // than at Ising's `1`, because the connected form takes off the rate at
        // which independent labels agree.
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

        // Gauge: cold is the ground state, `E = -j * n_plaquettes`.
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

        // A gauge transformation is a symmetry: flipping every link touching a
        // site leaves the energy, the Wilson table, and every Polyakov line
        // alone, because each closed loop crosses the site's links an even
        // number of times. This is the sharpest check that `plaquette_links`,
        // `loop_links`, and `site_links` all agree about the geometry, and the
        // number of links a flip touches is `2 * D`, so it says more the higher
        // the dimension goes.
        let mut transformed = cold.clone();
        for site in (0..lat.n_sites()).step_by(3) {
            for link in lat.site_links(site) {
                let flipped = State::new(1 - transformed.peek(link).index()).unwrap();
                transformed.poke(link, flipped);
            }
        }
        // Without this the rest is vacuous: flipping a site's links twice would
        // put the configuration back and every assertion below would hold
        // trivially.
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

    /// The observables hold on known configurations at every dimension up to six.
    ///
    /// Six rather than the ten [`deltas_match_from_scratch`] reaches, because
    /// `wilson_rectangles` needs room for a rectangle — it caps its sides at half
    /// the shortest extent — so every extent here is at least four, and `4^10`
    /// sites is far past what belongs in a unit test. The narrower sweep is the
    /// one that costs volume; the exact energy check pays only for extents of two
    /// and so goes further.
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
