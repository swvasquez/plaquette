//! The ferromagnetic `q`-state Potts model: one unordered label out of `Q` on
//! every site, coupled by agreement. The physics and conventions are in
//! `docs/potts.md`.
//!
//! [`Potts`] owns the energy and the observables built on the label
//! populations; [`potts_measure`] composes the primaries into a per-config
//! [`PottsSample`], and [`potts_correlator`] wraps the model's scan in the
//! shared [`Correlator`] record. The run-config schema and the CPU/GPU
//! samplers sit in the submodules, re-exported here.

pub mod gpu;
pub mod run_config;
pub mod sampler;

pub use gpu::gpu_chain;
pub use run_config::PottsRunConfig;
pub use sampler::{AnyPottsChain, PottsSampler};

use super::axis_pair_sums;
use crate::action::{Action, BondAction};
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::observables::Correlator;
use crate::state::State;

/// The ferromagnetic `q`-state Potts model: one label out of `Q` on every site,
/// coupled by whether nearest neighbors carry the *same* label, with an optional
/// per-label energy offset,
///
/// ```text
/// H = -j * sum_<ij> delta(s_i, s_j)  -  sum_i h[s_i]
/// ```
///
/// where the first sum runs over each nearest-neighbor bond once and `delta` is
/// `1` when the two labels agree and `0` otherwise. `j > 0` is ferromagnetic:
/// agreeing neighbors lower the energy. Energies come out in the same units as
/// `j` and `h`.
///
/// The labels are *unordered*: the energy only ever asks whether two indices
/// are equal and never turns one into a number, so permuting the `Q` labels
/// the same way at every site leaves it untouched. A nonzero `h` breaks that
/// symmetry, which is the point of the term. At `Q = 2` this is the Ising
/// model up to a factor of two in the coupling, and `h = [H, -H]` reproduces
/// an Ising field of `H`. The offset as a per-label chemical potential, the
/// Ising map, and why both models are kept are all in `docs/potts.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Potts<const Q: usize> {
    /// Nearest-neighbor coupling `J`.
    j: f64,
    /// Per-label energy offset: entry `a` is subtracted from the energy once for
    /// every site carrying label `a`. All zeros is the symmetric model.
    h: [f64; Q],
}

/// The fewest dimensions [`Potts`] is defined in: one, the same as
/// [`Ising::MIN_DIMENSION`](crate::models::ising::Ising::MIN_DIMENSION) and for the same reason.
///
/// A free constant rather than an associated one because `Potts` is generic
/// over `Q`, and reading an associated constant would mean naming some `Q` —
/// which reads as though the floor depended on the state count. It does not.
/// [`Potts::MIN_DIMENSION`] aliases this.
pub const POTTS_MIN_DIMENSION: usize = 1;

/// The fewest states [`Potts`] is defined at: two.
///
/// At one state every bond agrees, the energy is constant, there is no other
/// state to propose, and [`Potts::order_parameter`] divides by `Q - 1 = 0`.
/// None of that fails loudly on its own — the chain runs and reports numbers —
/// which is why the floor is asserted rather than left to the arithmetic.
pub const POTTS_MIN_STATES: usize = 2;

/// What every guard on [`POTTS_MIN_STATES`] says, so the places that check it
/// cannot drift into several wordings.
pub(crate) const TOO_FEW_STATES: &str =
    "the Potts action scores agreement between labels, which needs at least two states";

impl<const Q: usize> Potts<Q> {
    /// The fewest dimensions this model is defined in — [`POTTS_MIN_DIMENSION`].
    pub const MIN_DIMENSION: usize = POTTS_MIN_DIMENSION;

    /// The fewest states this model is defined at — [`POTTS_MIN_STATES`].
    pub const MIN_STATES: usize = POTTS_MIN_STATES;

    /// A Potts action with nearest-neighbor coupling `j` and per-label energy
    /// offsets `h`; use [`symmetric`](Potts::symmetric) for the all-zero `h`.
    pub fn new(j: f64, h: [f64; Q]) -> Self {
        Potts { j, h }
    }

    /// A Potts action with coupling `j` and no per-label offsets — the
    /// symmetric model every exact result quoted in `docs/potts.md` assumes.
    pub fn symmetric(j: f64) -> Self {
        Potts::new(j, [0.0; Q])
    }

    /// The nearest-neighbor coupling this action carries.
    pub fn coupling(&self) -> f64 {
        self.j
    }

    /// The per-label energy offsets this action carries.
    pub fn offsets(&self) -> [f64; Q] {
        self.h
    }

    /// Whether every offset is zero, letting `energy` skip the label scan on
    /// what is by default every run.
    fn is_symmetric(&self) -> bool {
        self.h.iter().all(|&h_a| h_a == 0.0)
    }

    /// How many sites carry each label — the one lattice scan every quantity
    /// below is a reduction of.
    ///
    /// A relabelling permutes this array and nothing else, so any reduction
    /// that ignores the *order* of the entries is automatically invariant —
    /// both order parameters are such a reduction. It is `pub(crate)` so
    /// [`potts_measure`](crate::models::potts::potts_measure) can scan once and
    /// hand the counts to both; the contract checks live here because every
    /// path reaches them through this one.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub(crate) fn label_counts(config: &Configuration<Q>) -> [usize; Q] {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Integer counts: exact, and the divides by N happen in the callers.
        let mut counts = [0usize; Q];
        for state in config.variables() {
            counts[state.index()] += 1;
        }
        counts
    }

    /// [`order_parameter`](Potts::order_parameter) from counts already in hand.
    pub(crate) fn order_from_counts(counts: &[usize; Q], n_vars: usize) -> f64 {
        let most = counts.iter().copied().max().expect("Q >= 2 counts exist");
        let f_max = most as f64 / n_vars as f64;
        (Q as f64 * f_max - 1.0) / (Q as f64 - 1.0)
    }

    /// [`simplex_order_parameter`](Potts::simplex_order_parameter) from counts
    /// already in hand.
    pub(crate) fn simplex_from_counts(counts: &[usize; Q], n_vars: usize) -> f64 {
        // Integer accumulator: `sum_a N_a^2`, exact until the final scaling. The
        // square of a count fits comfortably, since the counts sum to N.
        let squares: u128 = counts.iter().map(|&c| (c as u128) * (c as u128)).sum();

        let n = n_vars as f64;
        let sum_f_squared = squares as f64 / (n * n);
        // Rounding can put the bracket a hair below zero on an exactly uniform
        // split, where it is analytically zero; clamping keeps the square root
        // real rather than returning NaN at the disordered floor.
        ((Q as f64 * sum_f_squared - 1.0) / (Q as f64 - 1.0))
            .max(0.0)
            .sqrt()
    }

    /// The order parameter `m = (Q * f_max - 1) / (Q - 1)`, where `f_max` is
    /// the fraction of sites carrying the most common label — zero at an equal
    /// split, one when a single label holds every site.
    ///
    /// Unsigned and cannot be otherwise: the labels are unordered, so there is
    /// no signed sum to take, and this already *is* the analogue of `<|m|>` —
    /// a downstream reduction cannot recover a signed `<m>` from the series.
    /// One of the two conventions in use;
    /// [`simplex_order_parameter`](Potts::simplex_order_parameter) is the
    /// other, and they differ away from the ends. See `docs/potts.md`.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn order_parameter(&self, config: &Configuration<Q>) -> f64 {
        Self::order_from_counts(&Self::label_counts(config), config.n_vars())
    }

    /// The order parameter in its *vector* form,
    /// `m = sqrt[(Q * sum_a f_a^2 - 1) / (Q - 1)]`, where `f_a` is the
    /// fraction of sites carrying label `a` — the length of the mean simplex
    /// vector, and the convention most published Potts curves are.
    ///
    /// Agrees with [`order_parameter`](Potts::order_parameter) at both ends
    /// and differs in between. The simplex construction, and why its `Q - 1`
    /// components make the Ising `3` in
    /// [`binder_cumulant`](crate::statistics::binder_cumulant) the wrong
    /// normalization for Potts, are in `docs/potts.md`.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn simplex_order_parameter(&self, config: &Configuration<Q>) -> f64 {
        Self::simplex_from_counts(&Self::label_counts(config), config.n_vars())
    }

    /// The per-config connected two-point function
    /// `C_r = (1/N) Σ_i [delta(s_i, s_{i+r}) - 1/Q]`, measured along each lattice
    /// axis under periodic boundaries.
    ///
    /// Returns one row per axis, laid out exactly as [`Ising::correlator`](crate::models::ising::Ising::correlator)'s:
    /// entry `μ`, index `r` covers displacements `r = 0..=L_μ/2`.
    ///
    /// Unlike the Ising correlator this subtracts a floor: independent labels
    /// still agree with probability `1/Q`, and taking that off is what makes
    /// the disordered-phase decay reach zero and a correlation length readable.
    /// The `r = 0` anchor is then `1 - 1/Q` rather than Ising's `1`. This is
    /// the raw per-config estimator only; ensemble averages and fits belong to
    /// statistics. See `docs/potts.md`.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn correlator<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
    ) -> [Vec<f64>; D] {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        let agree = axis_pair_sums(lattice, config, |s_i, s_j| i64::from(s_i == s_j));
        let n = config.n_vars() as f64;
        let floor = 1.0 / Q as f64;
        std::array::from_fn(|mu| agree[mu].iter().map(|&c| c as f64 / n - floor).collect())
    }
}

impl<const Q: usize, const D: usize> Action<Q, D> for Potts<Q> {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<Q>) -> f64 {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Integer accumulator: the count is exact until the final scaling.
        let mut agree: i64 = 0;
        for site in 0..config.n_vars() {
            let s_i = config.peek(site);
            // The neighbor row is ordered +0, −0, +1, −1, ...; taking the
            // forward columns only (every other entry) visits each bond once.
            for &j_site in lattice.site_neighbors(site).iter().step_by(2) {
                agree += i64::from(config.peek(j_site) == s_i);
            }
        }

        // The offset term `-sum_a h[a] * N_a` needs only the label populations;
        // the symmetric model — the default — skips the scan outright.
        let offset: f64 = if self.is_symmetric() {
            0.0
        } else {
            let counts = Self::label_counts(config);
            self.h
                .iter()
                .zip(counts)
                .map(|(&h_a, n_a)| h_a * n_a as f64)
                .sum()
        };

        -self.j * agree as f64 - offset
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
        var: usize,
        proposed: State<Q>,
    ) -> f64 {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Only bonds touching `var` change agreement, so
        // ΔE = -j * (agree_after - agree_before), counting neighbors on the
        // proposed and current labels. Once `Q > 2` the two counts are *not*
        // complements: a neighbor may carry a third label and enter neither.
        let current = config.peek(var);
        if proposed == current {
            return 0.0;
        }

        // Integer accumulator, exact until the final scaling, as in `energy`.
        let mut change: i64 = 0;
        for &j_site in lattice.site_neighbors(var) {
            let s_j = config.peek(j_site);
            change += i64::from(s_j == proposed) - i64::from(s_j == current);
        }

        // Only this site's own offset changes, by the difference between the
        // two labels' entries — why a constant shift is invisible to the chain.
        let offset = self.h[proposed.index()] - self.h[current.index()];

        -self.j * change as f64 - offset
    }
}

impl<const Q: usize> BondAction<Q> for Potts<Q> {
    /// Breaking one agreeing bond costs `j`, since the delta convention scores
    /// an agreeing bond `-j` and a disagreeing one zero.
    fn bond_energy_gap(&self) -> f64 {
        self.j
    }

    fn relabel_invariant(&self) -> bool {
        self.is_symmetric()
    }
}

/// The per-config measurement record for the Potts model, the counterpart of
/// [`Sample`](crate::models::ising::Sample).
///
/// It carries an unsigned order parameter where [`Sample`](crate::models::ising::Sample)
/// carries a signed magnetization — already the analogue of `<|m|>`, not a
/// rename. Both conventions are carried because published curves use either
/// and neither is recoverable from the other; see `docs/potts.md`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PottsSample {
    /// Total energy `H` of the configuration (from [`Action::energy`]).
    pub energy: f64,
    /// The order parameter in its most-populated-label form,
    /// `m = (Q · f_max − 1) / (Q − 1)`, running from `0` in the disordered phase
    /// to `1` when every site carries one label (from
    /// [`Potts::order_parameter`]).
    pub order: f64,
    /// The order parameter in its vector form,
    /// `m = sqrt[(Q · Σ_a f_a² − 1) / (Q − 1)]` (from
    /// [`Potts::simplex_order_parameter`]). Runs between the same two limits
    /// as `order`, and sits above it in between.
    pub simplex_order: f64,
}

/// Measure one `config` of the Potts `model` on `lattice` into a
/// [`PottsSample`].
///
/// `E` comes from the [`Action`] trait, as in [`measure`](crate::models::ising::measure);
/// the two order parameters are reductions of one label scan, so this counts
/// once and reduces twice rather than calling [`Potts::order_parameter`] and
/// [`Potts::simplex_order_parameter`], which each count for themselves.
pub fn potts_measure<const Q: usize, const D: usize>(
    model: &Potts<Q>,
    lattice: &Lattice<D>,
    config: &Configuration<Q>,
) -> PottsSample {
    let counts = Potts::<Q>::label_counts(config);
    let n_vars = config.n_vars();
    PottsSample {
        energy: model.energy(lattice, config),
        order: Potts::<Q>::order_from_counts(&counts, n_vars),
        simplex_order: Potts::<Q>::simplex_from_counts(&counts, n_vars),
    }
}

/// Measure the per-config two-point [`Correlator`]
/// `C_r = ⟨delta(s_i, s_{i+r})⟩ − 1/Q` of a Potts `config` on `lattice`.
///
/// Composes [`Potts::correlator`] exactly as [`correlator`](crate::models::ising::correlator) composes the Ising
/// one; these entries are already *connected*, so nothing is left for a
/// downstream subtraction to do.
pub fn potts_correlator<const Q: usize, const D: usize>(
    model: &Potts<Q>,
    lattice: &Lattice<D>,
    config: &Configuration<Q>,
) -> Correlator<D> {
    Correlator {
        per_axis: model.correlator(lattice, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deltas_match_from_scratch;
    use crate::models::ising::Ising;

    /// The Potts counterpart of the Ising sweep, at three states: with a third
    /// label in play the `after` and `before` counts stop being complements,
    /// so a delta written as though they were passes at two states and fails
    /// here.
    #[test]
    fn potts_deltas_match_from_scratch_in_every_dimension() {
        fn probe<const D: usize>(shape: [usize; D]) {
            deltas_match_from_scratch::<3, D, _>(shape, &Potts::symmetric(1.0), Cell::Site);
        }

        probe([6]);
        probe([4, 4]);
        probe([3, 4, 5]);
        probe([2, 3, 2, 3]);
        probe([2, 2, 3, 2, 2]);
        probe([2, 3, 2, 2, 3, 2]);
        probe([2, 2, 2, 3, 2, 2, 2, 2]);
        probe([2, 2, 2, 2, 3, 2, 2, 2, 2, 2]);
    }

    /// A uniform configuration is the ground state at each of the `Q` labels
    /// in turn — nothing may single out label `0`, the one a cold start picks.
    #[test]
    fn a_uniform_potts_configuration_has_every_bond_agreeing() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6, 4]);
        let action = Potts::symmetric(1.5);
        // `D` forward bonds per site is exactly the link count.
        let n_bonds = lat.n_links() as f64;

        for label in 0..Q {
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            for site in 0..config.n_vars() {
                config.poke(site, State::new(label).unwrap());
            }

            assert_eq!(
                action.energy(&lat, &config),
                -1.5 * n_bonds,
                "label {label}"
            );
            assert_eq!(action.order_parameter(&config), 1.0, "label {label}");
        }
    }

    /// Relabelling every site by one permutation of the `Q` states moves
    /// nothing measured — the sharpest check that the energy compares labels
    /// rather than reading them as numbers.
    #[test]
    fn potts_energy_is_invariant_under_a_global_relabelling() {
        const Q: usize = 4;
        let lat = Lattice::new([4, 4, 4]);
        let action = Potts::symmetric(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(31);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);
        let order_before = action.order_parameter(&config);
        let simplex_before = action.simplex_order_parameter(&config);
        let correlator_before = action.correlator(&lat, &config);

        // A three-cycle: neither the identity nor a mere swap.
        let permutation = [1usize, 2, 0, 3];
        let mut relabelled = config.clone();
        for site in 0..config.n_vars() {
            let to = permutation[config.peek(site).index()];
            relabelled.poke(site, State::new(to).unwrap());
        }
        assert_ne!(
            relabelled, config,
            "the relabelling should change the field"
        );

        assert_eq!(action.energy(&lat, &relabelled), before);
        assert_eq!(action.order_parameter(&relabelled), order_before);
        assert_eq!(action.simplex_order_parameter(&relabelled), simplex_before);
        assert_eq!(action.correlator(&lat, &relabelled), correlator_before);
    }

    /// Potts at coupling `2J` prices every move exactly like zero-field Ising
    /// at `J`, and the whole-lattice energies differ by `-J` per bond — two
    /// independently written actions landing on the same numbers. The map is
    /// in `docs/potts.md`.
    #[test]
    fn potts_at_two_states_prices_moves_like_ising() {
        let lat = Lattice::new([4, 6]);
        let j = 1.5;
        let potts = Potts::symmetric(2.0 * j);
        let ising = Ising::new(j, 0.0);

        let mut rng = crate::rng::RandRng::seed_from_u64(97);
        let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        // The whole-lattice energies differ by the constant the delta drops.
        let offset = -j * lat.n_links() as f64;
        assert_eq!(
            potts.energy(&lat, &config),
            ising.energy(&lat, &config) + offset
        );

        for site in 0..config.n_vars() {
            let proposed = State::new(1 - config.peek(site).index()).unwrap();
            assert_eq!(
                potts.energy_delta(&lat, &config, site, proposed),
                ising.energy_delta(&lat, &config, site, proposed),
                "site {site}"
            );
        }
    }

    /// A uniform configuration agrees at every separation, so every correlator
    /// entry is the connected form's ceiling `1 - 1/Q`.
    #[test]
    fn the_potts_correlator_of_a_uniform_config_is_one_minus_one_over_q() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6]);
        let action = Potts::symmetric(1.0);
        let config = Configuration::<Q>::cold(&lat, Cell::Site);

        let c = action.correlator(&lat, &config);
        assert_eq!(c.len(), 2); // one row per axis
        let ceiling = 1.0 - 1.0 / Q as f64;
        for (mu, row) in c.iter().enumerate() {
            assert_eq!(row.len(), lat.shape()[mu] / 2 + 1);
            assert!(row.iter().all(|&v| v == ceiling), "axis {mu}: {row:?}");
        }
    }

    /// Independent labels agree at the uncorrelated rate `1/Q`, which is what
    /// the connected form subtracts — every entry past `r = 0` sits near zero,
    /// while `r = 0` keeps its `1 - 1/Q` anchor.
    #[test]
    fn the_potts_correlator_of_independent_labels_is_near_zero() {
        const Q: usize = 3;
        let lat = Lattice::new([32, 32]);
        let action = Potts::symmetric(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(20260809);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        let c = action.correlator(&lat, &config);
        for (mu, row) in c.iter().enumerate() {
            assert!(
                (row[0] - (1.0 - 1.0 / Q as f64)).abs() < 1e-12,
                "axis {mu}: C_0 = {}",
                row[0]
            );
            // 1024 sites of a two-valued indicator: the standard error on each
            // entry is about 0.015, so 0.05 is a few of those wide.
            for (r, &value) in row.iter().enumerate().skip(1) {
                assert!(value.abs() < 0.05, "axis {mu}, r = {r}: {value}");
            }
        }
    }

    /// The order parameter runs from 0 at an equal split to 1 on a single
    /// label — what the `(Q f_max - 1) / (Q - 1)` normalization is for.
    #[test]
    fn the_order_parameter_spans_zero_to_one() {
        const Q: usize = 4;
        let lat = Lattice::new([8, 8]); // 64 sites, divisible by Q
        let action = Potts::symmetric(1.0);

        let mut split = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..split.n_vars() {
            split.poke(site, State::new(site % Q).unwrap());
        }
        assert_eq!(action.order_parameter(&split), 0.0);

        let ordered = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(action.order_parameter(&ordered), 1.0);

        // Halfway between: half the sites on one label, the rest spread evenly
        // over the other three, so f_max = 1/2 and m = (4/2 - 1) / 3 = 1/3.
        let mut partial = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 32..partial.n_vars() {
            partial.poke(site, State::new(1 + site % (Q - 1)).unwrap());
        }
        assert!((action.order_parameter(&partial) - 1.0 / 3.0).abs() < 1e-12);
    }

    /// The two order parameters agree at both ends and differ in between: the
    /// half-and-half split gives `1/4` under the most-populated reading and
    /// `1/2` under the vector one, which is the whole reason both are kept.
    #[test]
    fn the_two_order_parameters_agree_at_the_ends_and_differ_between_them() {
        const Q: usize = 3;
        let lat = Lattice::new([6, 6]); // 36 sites, divisible by Q and by 2
        let action = Potts::symmetric(1.0);

        let uniform = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(action.order_parameter(&uniform), 1.0);
        assert_eq!(action.simplex_order_parameter(&uniform), 1.0);

        let mut split = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..split.n_vars() {
            split.poke(site, State::new(site % Q).unwrap());
        }
        assert_eq!(action.order_parameter(&split), 0.0);
        assert_eq!(action.simplex_order_parameter(&split), 0.0);

        let mut half = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..half.n_vars() / 2 {
            half.poke(site, State::new(1).unwrap());
        }
        assert!((action.order_parameter(&half) - 0.25).abs() < 1e-12);
        assert!((action.simplex_order_parameter(&half) - 0.5).abs() < 1e-12);
    }

    /// At two states the simplex is an interval, so the two conventions are
    /// the same function, and both equal `|M| / N` of the corresponding Ising
    /// field — tying both to a quantity that was already tested.
    #[test]
    fn at_two_states_both_order_parameters_are_the_ising_magnitude() {
        let lat = Lattice::new([4, 4]);
        let potts = Potts::symmetric(1.0);
        let ising = Ising::new(1.0, 0.0);
        let n_sites = lat.n_sites() as f64;

        let mut rng = crate::rng::RandRng::seed_from_u64(20260811);
        for _ in 0..32 {
            let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let expected = (ising.magnetization(&config) / n_sites).abs();

            assert!((potts.order_parameter(&config) - expected).abs() < 1e-12);
            assert!((potts.simplex_order_parameter(&config) - expected).abs() < 1e-12);
        }
    }

    /// Each site pays the offset entry matching its own label: at `j = 0` the
    /// whole energy is the offset sum, fixed by the populations.
    #[test]
    fn the_offsets_are_paid_once_per_site_carrying_the_label() {
        const Q: usize = 3;
        let lat = Lattice::new([6, 6]); // 36 sites, three equal populations
        let h = [0.5, -1.5, 0.25];
        let action = Potts::<Q>::new(0.0, h);

        let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..config.n_vars() {
            config.poke(site, State::new(site % Q).unwrap());
        }

        // Twelve sites on each label, and `j = 0` leaves only the offset term.
        let expected = -12.0 * (h[0] + h[1] + h[2]);
        assert_eq!(action.energy(&lat, &config), expected);
    }

    /// A constant added to every offset shifts every configuration's energy by
    /// the same amount, so it cancels out of every `energy_delta` and
    /// Boltzmann ratio — the reason only differences between entries carry
    /// content.
    #[test]
    fn a_constant_added_to_every_offset_shifts_the_energy_and_nothing_else() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6]);
        let shifted = Potts::<Q>::new(1.0, [1.25, 0.75, 2.25]);
        let base = Potts::<Q>::new(1.0, [0.25, -0.25, 1.25]); // each entry 1.0 lower

        let mut rng = crate::rng::RandRng::seed_from_u64(20260812);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        // The whole-lattice energies differ by the shift times the site count.
        let sites = config.n_vars() as f64;
        assert_eq!(
            shifted.energy(&lat, &config),
            base.energy(&lat, &config) - sites
        );

        for site in 0..config.n_vars() {
            for label in 0..Q {
                let proposed = State::new(label).unwrap();
                assert_eq!(
                    shifted.energy_delta(&lat, &config, site, proposed),
                    base.energy_delta(&lat, &config, site, proposed),
                    "site {site} -> {label}"
                );
            }
        }
    }

    /// An offset breaks the relabelling symmetry — the other side of
    /// [`potts_energy_is_invariant_under_a_global_relabelling`].
    #[test]
    fn an_offset_breaks_the_relabelling_symmetry() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 4]);
        let action = Potts::<Q>::new(1.0, [0.75, 0.0, 0.0]);

        let mut rng = crate::rng::RandRng::seed_from_u64(20260813);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        // Cycle every label forward one place.
        let mut relabelled = config.clone();
        for site in 0..config.n_vars() {
            let to = (config.peek(site).index() + 1) % Q;
            relabelled.poke(site, State::new(to).unwrap());
        }

        // The coupling term is still invariant, so any change is the offset's.
        let symmetric = Potts::<Q>::symmetric(1.0);
        assert_eq!(
            symmetric.energy(&lat, &relabelled),
            symmetric.energy(&lat, &config)
        );
        assert_ne!(action.energy(&lat, &relabelled), before);
    }

    /// Offsets `[H, -H]` at coupling `2J` reproduce an Ising field of `H` at
    /// `J`: the two actions must again price every move identically.
    #[test]
    fn the_offsets_reproduce_an_ising_field_at_two_states() {
        let lat = Lattice::new([4, 6]);
        let (j, field) = (1.5, 0.25);
        let potts = Potts::<2>::new(2.0 * j, [field, -field]);
        let ising = Ising::new(j, field);

        let mut rng = crate::rng::RandRng::seed_from_u64(20260814);
        let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        // Only the coupling term's constant separates the whole energies.
        let offset = -j * lat.n_links() as f64;
        assert_eq!(
            potts.energy(&lat, &config),
            ising.energy(&lat, &config) + offset
        );

        for site in 0..config.n_vars() {
            let proposed = State::new(1 - config.peek(site).index()).unwrap();
            assert_eq!(
                potts.energy_delta(&lat, &config, site, proposed),
                ising.energy_delta(&lat, &config, site, proposed),
                "site {site}"
            );
        }
    }

    /// A link field is rejected rather than silently misread, as it is for the
    /// other site model.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Potts labels live on sites")]
    fn the_potts_action_rejects_a_link_field() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<3>::cold(&lat, Cell::Link);
        Potts::symmetric(1.0).energy(&lat, &config);
    }

    /// The state floor holds in release, unlike the cell-kind guards: `Q` is a
    /// compile-time constant, so a real `assert!` folds away to nothing and
    /// there is no reason to make it debug-only.
    #[test]
    #[should_panic(expected = "at least two states")]
    fn the_potts_action_refuses_one_state_in_any_profile() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<1>::cold(&lat, Cell::Site);
        Potts::symmetric(1.0).energy(&lat, &config);
    }

    #[test]
    fn potts_measure_bundles_energy_and_the_order_parameter() {
        // Uniform 4x6 at three states, j = 2: all 48 bonds agree, so
        // E = -2 * 48, and one label holding every site puts both order
        // parameters at their ceiling.
        let lat = Lattice::new([4, 6]);
        let model = Potts::<3>::symmetric(2.0);
        let config = Configuration::<3>::cold(&lat, Cell::Site);

        let sample = potts_measure(&model, &lat, &config);
        assert_eq!(sample.energy, model.energy(&lat, &config));
        assert_eq!(sample.order, model.order_parameter(&config));
        assert_eq!(sample.simplex_order, model.simplex_order_parameter(&config));
        assert_eq!(sample.energy, -96.0);
        assert_eq!(sample.order, 1.0);
        assert_eq!(sample.simplex_order, 1.0);
    }

    #[test]
    fn potts_measure_carries_both_order_parameter_conventions() {
        // A half-and-half split makes the two conventions genuinely disagree.
        let lat = Lattice::new([4, 4]);
        let model = Potts::<3>::symmetric(1.0);
        let mut config = Configuration::<3>::cold(&lat, Cell::Site);
        for site in 0..8 {
            config.poke(site, State::new(1).unwrap());
        }

        let sample = potts_measure(&model, &lat, &config);
        assert!((sample.order - 0.25).abs() < 1e-12);
        assert!((sample.simplex_order - 0.5).abs() < 1e-12);
    }

    #[test]
    fn potts_correlator_wraps_the_model_measurement() {
        // A uniform field puts every entry at the ceiling `1 - 1/Q`.
        let lat = Lattice::new([4, 4]);
        let model = Potts::<3>::symmetric(1.0);
        let config = Configuration::<3>::cold(&lat, Cell::Site);

        let c = potts_correlator(&model, &lat, &config);
        assert_eq!(c.per_axis, model.correlator(&lat, &config));
        assert_eq!(c.per_axis[0].len(), 3); // r = 0..=L/2 on axis 0 (L = 4)
        for row in &c.per_axis {
            assert!(row.iter().all(|&v| (v - 2.0 / 3.0).abs() < 1e-12));
        }
    }
}
