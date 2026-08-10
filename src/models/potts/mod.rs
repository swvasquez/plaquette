//! The ferromagnetic `q`-state Potts model: one unordered label out of `Q` on
//! every site, coupled by agreement, and the observables that respect the
//! relabelling symmetry.
//!
//! [`Potts`] owns the energy — bonds scored by whether the two labels agree,
//! plus an optional per-label offset — and the value-semantic observables built
//! on the label populations: the two order-parameter conventions and the
//! connected correlator. [`potts_measure`] composes the primaries into a
//! per-config [`PottsSample`], and [`potts_correlator`] wraps the model's scan
//! in the shared [`Correlator`] record. The
//! run-config schema and the CPU/GPU samplers sit in the submodules,
//! re-exported here.

pub mod gpu;
pub mod run_config;
pub mod sampler;

pub use gpu::GpuPottsChain;
pub use run_config::PottsRunConfig;
pub use sampler::{AnyPottsChain, PottsSampler};

use super::axis_pair_sums;
use crate::action::Action;
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
/// The offset `h` is one number per label, and a site pays the one matching its
/// own label whatever its neighbors are doing — a chemical potential per label,
/// which is what the second sum literally is once it is read as `-sum_a h[a] *
/// N_a` over the label populations. It is uniform across the lattice: every site
/// carrying a given label gets the same number, so the sum is over the `Q`
/// labels and not over anything geometric. Adding a constant to every entry
/// shifts `H` by a constant and cancels out of every energy difference, so only
/// the differences between entries carry content, and the usual choice is a
/// single entry offset with the rest at zero.
///
/// The labels are *unordered*, and that is the whole character of the model.
/// Nothing distinguishes one from another, so the energy only ever asks whether
/// two indices are equal and never turns one into a number — which is why this
/// is the one action here generic over `Q` rather than fixed at the two states
/// the private `decode` knows. Permuting the `Q` labels the same way at every
/// site leaves the energy untouched, and everything measured has to respect that
/// symmetry; see [`order_parameter`](Potts::order_parameter) for why the Ising
/// magnetization has no counterpart under it.
///
/// At `Q = 2` and zero offset this is the Ising model up to a factor of two in
/// the coupling. Reading the two labels as `±1` gives
/// `delta(s_i, s_j) = (1 + s_i s_j) / 2`, so `Potts` at coupling `2J` differs
/// from zero-field [`Ising`](crate::models::ising::Ising) at `J` only by the constant `-J` per bond — the same
/// energy landscape, and therefore the same `energy_delta` on every move. The
/// offsets extend that: `h = [H, -H]` reproduces an Ising field of `H`, since
/// Ising's `-h * sum_i s_i` gives `+h` to an up spin and `-h` to a down one.
///
/// Both models are kept rather than one wrapping the other: [`Ising`](crate::models::ising::Ising)'s
/// observables are built on reading states as signs, its field is a scalar
/// rather than a vector, and its GPU kernel is a faster specialized path.
///
/// The offsets break the relabelling symmetry, which is the point of having
/// them. At `h = 0` the `Q` ordered ground states are exactly degenerate, so no
/// label-specific average survives a long run; a nonzero entry lifts that
/// degeneracy and picks one out. Everything measured on a symmetric run — the
/// order parameters especially — is built to be invariant under relabelling, and
/// that argument does not survive a nonzero `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Potts<const Q: usize> {
    /// Nearest-neighbor coupling `J`.
    j: f64,
    /// Per-label energy offset: entry `a` is subtracted from the energy once for
    /// every site carrying label `a`. All zeros is the symmetric model.
    h: [f64; Q],
}

/// The fewest dimensions [`Potts`] is defined in.
///
/// One, the same as [`Ising::MIN_DIMENSION`](crate::models::ising::Ising::MIN_DIMENSION) and for the same reason: the energy
/// scores nearest-neighbor bonds, and a ring of labels has them.
///
/// A free constant rather than an associated one, unlike the other two models',
/// because `Potts` is generic over `Q` and reading an associated constant off it
/// would mean naming some `Q` — which reads as though the floor depended on the
/// state count. It does not, and neither does the one below, which *constrains*
/// the state count and so certainly cannot be indexed by it.
/// [`Potts::MIN_DIMENSION`] aliases this so the type still answers the question
/// its siblings answer.
pub const POTTS_MIN_DIMENSION: usize = 1;

/// The fewest states [`Potts`] is defined at.
///
/// Two. One state is not a small Potts model but no model at all: every site
/// carries the same label, every bond agrees, the energy is the same constant
/// for every configuration, and there is no other state to propose.
/// [`Potts::order_parameter`] would divide by `Q - 1 = 0` on top of that. None of
/// this fails loudly on its own — it produces a chain that runs and reports
/// numbers — which is why the floor is stated rather than left to the
/// arithmetic, exactly as [`Z2Gauge::MIN_DIMENSION`](crate::models::gauge::Z2Gauge::MIN_DIMENSION) is.
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
    /// offsets `h`.
    ///
    /// Pass `[0.0; Q]` for the symmetric model, or use
    /// [`symmetric`](Potts::symmetric), which says the same thing without making
    /// the reader count zeros.
    pub fn new(j: f64, h: [f64; Q]) -> Self {
        Potts { j, h }
    }

    /// A Potts action with coupling `j` and no per-label offsets — the symmetric
    /// model, where all `Q` labels are interchangeable and the `Q` ordered ground
    /// states are degenerate.
    ///
    /// This is the standard case and the one every exact result quoted in
    /// `docs/potts.md` assumes, which is why it gets its own constructor rather
    /// than a defaulted argument.
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

    /// Whether every offset is zero, so the `Q` labels are interchangeable.
    ///
    /// Read as a shortcut rather than as a question about the model: the offset
    /// term is then identically zero for every configuration, so `energy` can
    /// skip counting labels altogether. `Q` comparisons in place of a scan of
    /// the lattice, on what is by default every run.
    fn is_symmetric(&self) -> bool {
        self.h.iter().all(|&h_a| h_a == 0.0)
    }

    /// How many sites carry each label — the one lattice scan every quantity
    /// below is a reduction of, and the only thing any of them reads.
    ///
    /// Counting is where the relabelling symmetry is dealt with: a permutation
    /// of the labels permutes this array and nothing else, so any reduction of
    /// it that ignores the *order* of the entries is automatically invariant.
    /// Both order parameters are such a reduction — one takes the maximum, the
    /// other a sum of squares — which is why neither has to argue for its own
    /// invariance separately.
    ///
    /// It is `pub(crate)` so [`potts_measure`](crate::models::potts::potts_measure)
    /// can scan once and hand the counts to both reductions, rather than each
    /// public method walking the lattice for itself. The contract checks live
    /// here rather than in the callers because every path reaches them through
    /// this one.
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

    /// The order parameter `m = (Q * f_max - 1) / (Q - 1)`, where `f_max` is the
    /// fraction of sites carrying the most common label.
    ///
    /// The normalization is the conventional one, chosen so both ends come out
    /// clean: in the disordered phase every label holds about `1/Q` of the
    /// sites, so `f_max -> 1/Q` and `m -> 0`, while a fully ordered
    /// configuration puts every site on one label and gives `m = 1`.
    ///
    /// It is built from the most populated label rather than from a signed sum
    /// because the labels are unordered — there is no `+1` and `−1` to add up,
    /// so [`Ising::magnetization`](crate::models::ising::Ising::magnetization)'s definition has nothing to carry over. What
    /// does survive the relabelling symmetry is the *imbalance* between the
    /// populations, and reading it off the largest population is what makes the
    /// answer invariant: a permutation of the labels permutes the counts and
    /// leaves their maximum where it was.
    ///
    /// The price is that this is unsigned and cannot be otherwise, so a
    /// downstream reduction cannot recover both `<m>` and `<|m|>` from the
    /// series the way it can from [`Ising::magnetization`](crate::models::ising::Ising::magnetization). This already *is*
    /// the analogue of `<|m|>`.
    ///
    /// This is one of the two conventions in use, and
    /// [`simplex_order_parameter`](Potts::simplex_order_parameter) is the other;
    /// they agree at both ends and differ in between, so a comparison against a
    /// published number has to know which one it is.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn order_parameter(&self, config: &Configuration<Q>) -> f64 {
        Self::order_from_counts(&Self::label_counts(config), config.n_vars())
    }

    /// The order parameter in its *vector* form:
    /// `m = sqrt[(Q * sum_a f_a^2 - 1) / (Q - 1)]`, where `f_a` is the fraction
    /// of sites carrying label `a`.
    ///
    /// The other convention in use, and the one most published Potts curves are.
    /// It comes from a different construction: place the `Q` labels at the
    /// vertices of a regular `Q-1` dimensional simplex, so that unit vectors
    /// `e_a` satisfy `e_a . e_b = -1/(Q-1)` for `a != b` and sum to zero, and
    /// take the length of the average `(1/N) sum_i e_{s_i}`. Expanding that
    /// square collapses the vectors out entirely and leaves the populations,
    /// which is why this costs the same scan
    /// [`order_parameter`](Potts::order_parameter) does rather than any geometry.
    ///
    /// It agrees with the other convention at both ends — one at a uniform
    /// configuration, zero at an equal split — and differs between them: half
    /// the sites on one label and half on another gives `1/4` there and `1/2`
    /// here. At `Q = 2` the simplex is an interval and the two coincide
    /// identically, both reducing to `|M| / N` of the corresponding Ising field.
    ///
    /// This is the form whose `Q - 1` components make the Ising `3` in
    /// [`binder_cumulant`](crate::statistics::binder_cumulant) the wrong
    /// normalization for Potts; `docs/potts.md` carries that argument.
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
    /// entry `μ`, index `r` covers displacements `r = 0..=L_μ/2`, the
    /// non-redundant half that translation invariance leaves.
    ///
    /// Unlike the Ising correlator this subtracts a floor, because the raw
    /// agreement fraction does not decay to zero: two labels far enough apart to
    /// be independent still agree with probability `1/Q`, so an uncorrelated
    /// configuration gives `1/Q` at every separation rather than `0`. Taking
    /// that off is what leaves a quantity that falls to zero in the disordered
    /// phase the way `<s_i s_{i+r}>` does, and so what makes a correlation
    /// length readable off the decay. The `r = 0` anchor is then `1 - 1/Q`
    /// rather than Ising's `1`, since a site always agrees with itself.
    ///
    /// This is the raw per-config estimator only. The ensemble average and the
    /// correlation-length fit are reductions over a chain of these arrays and
    /// belong to statistics.
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

        // The offset term is `-sum_a h[a] * N_a`, so it needs the label
        // populations and nothing about the geometry — the same counts the order
        // parameters reduce. The symmetric model skips the scan outright rather
        // than walking the lattice to multiply by zeros, which is worth the test
        // because it is the default and what every exact result assumes.
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

        // Only bonds touching `var` can change their agreement; every other bond
        // reads two untouched labels and cancels. So
        //
        //     ΔE = -j * (agree_after - agree_before),
        //
        // where each term counts how many of the site's 2D neighbors carry the
        // proposed or the current label. The two labels differ, so a neighbor
        // contributes to at most one of the counts.
        let current = config.peek(var);
        if proposed == current {
            return 0.0; // proposed state equals the current one
        }

        // Integer accumulator, exact until the final scaling, as in `energy`.
        let mut change: i64 = 0;
        for &j_site in lattice.site_neighbors(var) {
            let s_j = config.peek(j_site);
            change += i64::from(s_j == proposed) - i64::from(s_j == current);
        }

        // Only this site's own offset changes, and only by the difference
        // between the two labels' entries — which is why a constant added to
        // every entry is invisible to the chain.
        let offset = self.h[proposed.index()] - self.h[current.index()];

        -self.j * change as f64 - offset
    }
}

/// The per-config measurement record for the Potts model, the counterpart of
/// [`Sample`](crate::models::ising::Sample).
///
/// It carries an order parameter where [`Sample`](crate::models::ising::Sample) carries a magnetization, and
/// the difference is not a rename. Potts labels are unordered, so there is no
/// signed sum to take: what the record holds is the population imbalance, which
/// is already the analogue of `<|m|>` rather than the signed quantity both
/// `<m²>` and `<|m|>` can be recovered from downstream.
///
/// Both order parameters are carried rather than one, because the literature
/// uses both and they are not interchangeable away from the two ends. Neither is
/// recoverable from the other, and a run compared against a published curve needs
/// whichever that curve plotted — while the cost of having both is a second
/// reduction over the same label counts, negligible beside the neighbor scan the
/// energy already pays for.
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
    /// `m = sqrt[(Q · Σ_a f_a² − 1) / (Q − 1)]` — the length of the average over
    /// the sites of unit vectors pointing at the vertices of a `Q − 1`
    /// dimensional simplex (from [`Potts::simplex_order_parameter`]). Runs
    /// between the same two limits, and sits above `order` in between.
    pub simplex_order: f64,
}

/// Measure one `config` of the Potts `model` on `lattice` into a
/// [`PottsSample`].
///
/// The primaries come from the seams the model exposes, as [`measure`](crate::models::ising::measure)'s do: `E`
/// from the [`Action`] trait, the two order parameters from the same label
/// counts. Both are reductions of one scan of the lattice, so this counts once
/// and reduces twice rather than calling [`Potts::order_parameter`] and
/// [`Potts::simplex_order_parameter`], which each count for themselves — the
/// convenience those two offer a caller who wants only one is waste to a caller
/// who wants both.
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
/// one. What differs is that these entries are already *connected*: two
/// independent labels agree with probability `1/Q`, and the model takes that
/// floor off, so nothing is left for a downstream subtraction to do.
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

    /// The Potts counterpart, at three states rather than two.
    ///
    /// Three is the point of it: `Q = 2` would run the same arithmetic the Ising
    /// sweep above already covers, while at three a site's neighbors can carry a
    /// label that is neither the current nor the proposed one, so the `after`
    /// and `before` counts stop being complements of each other. A delta written
    /// as though they were would pass at two states and fail here.
    ///
    /// See [`ising_deltas_match_from_scratch_in_every_dimension`] for why the
    /// shapes are shaped this way.
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

    /// A uniform Potts configuration is the ground state: every bond agrees, so
    /// the energy is `-j` times the bond count, whichever label it settled on.
    ///
    /// Running it at each of the `Q` labels in turn is the cheapest form of the
    /// relabelling symmetry — nothing may single out label `0`, which is the one
    /// a cold start happens to pick.
    #[test]
    fn a_uniform_potts_configuration_has_every_bond_agreeing() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6, 4]);
        let action = Potts::symmetric(1.5);
        // A periodic lattice has `D` forward bonds per site, which is exactly
        // the link count.
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

    /// The Potts symmetry, which has no Ising analogue: relabelling every site
    /// by one permutation of the `Q` states leaves the energy alone.
    ///
    /// This is the sharpest check that the energy compares labels rather than
    /// reading them as numbers. A term that decoded an index into a value — the
    /// way both other models do — would move under a permutation that reorders
    /// those values, while the count of agreeing bonds cannot: a permutation is
    /// a bijection, so `s_i == s_j` before and after are the same statement.
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

        // A three-cycle on the first three labels, leaving the fourth fixed, so
        // the permutation is neither the identity nor a mere swap.
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

    /// Potts at two states is Ising at half the coupling: the two price every
    /// move identically.
    ///
    /// Reading the two labels as `±1` gives `delta = (1 + s_i s_j) / 2`, so a
    /// `Potts` bond at coupling `2J` is a zero-field `Ising` bond at `J` plus the
    /// constant `-J`. Constants cancel in a difference, so the two `energy_delta`
    /// values agree exactly, and the whole-lattice energies differ by `-J` times
    /// the bond count. Two independently written actions landing on the same
    /// numbers is what makes this worth asserting rather than deriving.
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
    /// entry is the full `1 - 1/Q` — the connected form's ceiling, and the value
    /// its `r = 0` anchor takes on any configuration at all.
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

    /// Independent labels agree at the uncorrelated rate `1/Q`, which is exactly
    /// what the connected form subtracts — so every entry past `r = 0` sits near
    /// zero, while `r = 0` keeps its `1 - 1/Q` anchor.
    ///
    /// This is the property that makes the subtraction worth doing: without it
    /// the correlator would tend to `1/Q` rather than to zero at large
    /// separation, and no correlation length could be read off the decay.
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

    /// The order parameter runs from 0 to 1: an equal split among the `Q` labels
    /// is the disordered floor, a single label the ordered ceiling.
    ///
    /// The floor is the reason for the `(Q f_max - 1) / (Q - 1)` normalization at
    /// all — the raw most-populated fraction would report `1/Q` there, which is
    /// a different number for every `Q` and no use as a scale.
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

    /// The two order parameters agree at both ends and differ in between.
    ///
    /// The ends are what the normalizations are chosen for, so agreeing there is
    /// no evidence that the two are different quantities — the half-and-half
    /// split is. It gives `1/4` under the most-populated reading, since one
    /// label holds half the sites, and `1/2` under the vector one, since two
    /// simplex vertices at sixty degrees average to half a unit vector. Anyone
    /// comparing a run against a published number has to know which of these it
    /// was, which is the whole reason both are kept.
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

    /// At two states the two conventions are the same function, and both are the
    /// Ising magnitude.
    ///
    /// A `Q - 1` dimensional simplex is an interval at `Q = 2`, so the vector
    /// construction has nowhere to point but along it and collapses onto the
    /// population imbalance the other convention reads directly. Both then equal
    /// `|M| / N` of the Ising field the same labels describe, which ties the two
    /// new quantities to one that was already tested rather than only to each
    /// other. Every configuration on the lattice is checked, since the identity
    /// is exact rather than statistical.
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

    /// The per-label offset does what its definition says: each site pays the
    /// entry matching its own label, with no reference to its neighbors.
    ///
    /// Checked on a configuration whose label populations are fixed by
    /// construction, so the expected energy is arithmetic rather than a
    /// reference run: at `j = 0` the coupling term vanishes and the whole energy
    /// is the offset sum, which the counts give exactly.
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

    /// A constant added to every offset is invisible to the chain.
    ///
    /// The reason only differences carry content: shifting every entry by the
    /// same amount changes the energy of *every* configuration by the same
    /// constant, so it cancels out of every `energy_delta` and out of every
    /// Boltzmann ratio the sampler forms. Both halves are asserted, since a
    /// delta that quietly disagreed with the whole-lattice energy would break
    /// the telescoping identity the sweeps rely on.
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

        // Every move is priced identically, which is what makes the two runs the
        // same chain rather than merely the same distribution.
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

    /// An offset breaks the relabelling symmetry, which is the whole point of
    /// having one.
    ///
    /// The symmetric model's invariance is asserted in
    /// [`potts_energy_is_invariant_under_a_global_relabelling`]; this is the
    /// other side of it. Permuting the labels while leaving `h` alone now moves
    /// the energy, because a site that was paying one entry starts paying
    /// another.
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

    /// The offsets reproduce an Ising field at two states.
    ///
    /// `Potts` at coupling `2J` already matches zero-field `Ising` at `J`; this
    /// extends that to the field. Ising's `-h * sum_i s_i` gives `+h` to an up
    /// spin and `-h` to a down one, so `[H, -H]` is the same term written per
    /// label, and the two actions must again price every move identically.
    #[test]
    fn the_offsets_reproduce_an_ising_field_at_two_states() {
        let lat = Lattice::new([4, 6]);
        let (j, field) = (1.5, 0.25);
        let potts = Potts::<2>::new(2.0 * j, [field, -field]);
        let ising = Ising::new(j, field);

        let mut rng = crate::rng::RandRng::seed_from_u64(20260814);
        let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        // Only the coupling term's constant separates the whole energies; the
        // field terms are equal outright.
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

    /// The state floor holds in release, unlike the cell-kind guards.
    ///
    /// Deliberately not gated on `debug_assertions`, for the reason
    /// [`the_gauge_action_refuses_one_dimension_in_any_profile`] is not: `Q` is a
    /// compile-time constant, so a real `assert!` folds away to nothing and there
    /// is no reason to make it debug-only. At one state every bond agrees for
    /// every configuration, so the chain would report a constant energy and an
    /// order parameter divided by zero — numbers, not a failure.
    #[test]
    #[should_panic(expected = "at least two states")]
    fn the_potts_action_refuses_one_state_in_any_profile() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<1>::cold(&lat, Cell::Site);
        Potts::symmetric(1.0).energy(&lat, &config);
    }

    #[test]
    fn potts_measure_bundles_energy_and_the_order_parameter() {
        // Uniform 4x6 at three states, j = 2: every one of the lattice's
        // `D * N = 48` bonds agrees, so E = -2 * 48, and one label holding every
        // site puts the order parameter at its ceiling.
        let lat = Lattice::new([4, 6]);
        let model = Potts::<3>::symmetric(2.0);
        let config = Configuration::<3>::cold(&lat, Cell::Site);

        let sample = potts_measure(&model, &lat, &config);
        assert_eq!(sample.energy, model.energy(&lat, &config));
        assert_eq!(sample.order, model.order_parameter(&config));
        assert_eq!(sample.simplex_order, model.simplex_order_parameter(&config));
        assert_eq!(sample.energy, -96.0);
        assert_eq!(sample.order, 1.0);
        // A uniform field is where the two conventions coincide.
        assert_eq!(sample.simplex_order, 1.0);
    }

    #[test]
    fn potts_measure_carries_both_order_parameter_conventions() {
        // Half the sites on one label and half on another: the record must show
        // the two conventions genuinely disagreeing, or carrying both would be
        // storing one number twice.
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
        // A uniform field agrees at every separation, so every entry is the
        // connected form's ceiling `1 - 1/Q`, including the `r = 0` anchor.
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
