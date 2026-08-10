//! The Ising model: spins `±1` on the sites, the value semantics they are read
//! through, and the observables built on them.
//!
//! [`Ising`] owns the energy — nearest-neighbor bonds scored as a product of
//! signs, plus a uniform external field — and the value-semantic observables
//! that read the same decode: the magnetization and the two-point correlator.
//! [`measure`] composes the primaries into a per-config [`Sample`], and
//! [`correlator`] wraps the model's scan in the shared
//! [`Correlator`] record. The run-config
//! schema, the CPU/GPU samplers, and the GPU backend sit in the submodules,
//! re-exported here.

pub mod gpu;
pub mod run_config;
pub mod sampler;

pub use gpu::GpuIsingChain;
pub use run_config::IsingRunConfig;
pub use sampler::{AnyIsingChain, IsingSampler};

use super::{axis_pair_sums, decode};
use crate::action::Action;
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::observables::Correlator;
use crate::state::State;

/// The Ising model: spins `s_i = ±1` with nearest-neighbor coupling `j` and a
/// uniform external field `h`,
///
/// ```text
/// H = -j * sum_<ij> s_i s_j  -  h * sum_i s_i
/// ```
///
/// where the first sum runs over each nearest-neighbor bond once. `j > 0` is
/// ferromagnetic (aligned neighbors lower the energy). Energies come out in the
/// same units as `j` and `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ising {
    /// Nearest-neighbor coupling `J`.
    j: f64,
    /// Uniform external field `h` (set to `0.0` for the field-free model).
    h: f64,
}

impl Ising {
    /// The fewest dimensions this model is defined in.
    ///
    /// The energy scores nearest-neighbor bonds, and a line has them — a ring of
    /// spins is a perfectly good, and exactly solvable, Ising model. The peer of
    /// [`Z2Gauge::MIN_DIMENSION`](crate::models::gauge::Z2Gauge::MIN_DIMENSION), which is two for a reason that bites much
    /// harder; this one exists so both models answer the same question in the
    /// same place.
    pub const MIN_DIMENSION: usize = 1;

    /// An Ising action with coupling `j` and external field `h`.
    pub fn new(j: f64, h: f64) -> Self {
        Ising { j, h }
    }

    /// The total magnetization `M = sum_i s_i` of `config` — the raw *signed*
    /// spin sum, not `|M|` and not a density.
    ///
    /// Keeping the sign is what makes both `<m²>` and `<|m|>` recoverable from
    /// the series downstream. It reads the private `decode` map, so it is an
    /// inherent method rather than part of the energy-only [`Action`] trait.
    pub fn magnetization(&self, config: &Configuration<2>) -> f64 {
        // Integer accumulator: the sum is exact until the final cast.
        let spin_sum: i64 = config.variables().iter().map(|&s| decode(s) as i64).sum();
        spin_sum as f64
    }

    /// The per-config two-point correlator `C_r = (1/N) Σ_i s_i · s_{i+r}`,
    /// measured along each lattice axis under periodic boundaries.
    ///
    /// Returns one row per axis: entry `μ`, index `r` is `C_r` for displacement
    /// `r = 0..=L_μ/2` (length `L_μ/2 + 1`). Only that non-redundant half is
    /// stored, since `C_r = C_{L_μ − r}` by translation invariance, and index `0`
    /// keeps the `C_0 = 1` anchor.
    ///
    /// This is the *raw per-config estimator* only. The connected subtraction
    /// (`− ⟨s⟩²`), the ensemble average, and the correlation-length fit are
    /// reductions over a chain of these arrays and belong to statistics.
    pub fn correlator<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
    ) -> [Vec<f64>; D] {
        let sums = axis_pair_sums(lattice, config, |s_i, s_j| {
            (decode(s_i) * decode(s_j)) as i64
        });
        let n = config.n_vars() as f64;
        std::array::from_fn(|mu| sums[mu].iter().map(|&c| c as f64 / n).collect())
    }
}

impl<const D: usize> Action<2, D> for Ising {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        debug_assert!(config.cell() == Cell::Site, "Ising spins live on sites");

        // Integer accumulators: the sums are exact until the final scaling.
        let mut bond_sum: i64 = 0; // sum over each bond once of s_i s_j
        let mut spin_sum: i64 = 0; // sum_i s_i, for the field term

        for site in 0..config.n_vars() {
            let s_i = decode(config.peek(site));
            spin_sum += s_i as i64;

            // The neighbor row is ordered +0, −0, +1, −1, ...; taking the
            // forward columns only (every other entry) visits each bond once.
            for &j_site in lattice.site_neighbors(site).iter().step_by(2) {
                bond_sum += (s_i * decode(config.peek(j_site))) as i64;
            }
        }

        -self.j * bond_sum as f64 - self.h * spin_sum as f64
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        var: usize,
        proposed: State<2>,
    ) -> f64 {
        debug_assert!(config.cell() == Cell::Site, "Ising spins live on sites");

        // ΔE = -(s'_i - s_i) * (J * sum_{j in nbrs(i)} s_j + h). Only bonds
        // touching `var` and its field term change; everything else cancels.
        let ds = (decode(proposed) - decode(config.peek(var))) as i64;
        if ds == 0 {
            return 0.0; // proposed state equals the current one
        }

        // All 2D neighbors (both directions): every incident bond changes.
        let neighbor_sum: i64 = lattice
            .site_neighbors(var)
            .iter()
            .map(|&j_site| decode(config.peek(j_site)) as i64)
            .sum();

        -(ds as f64) * (self.j * neighbor_sum as f64 + self.h)
    }
}

/// The per-config measurement record: the primary quantities of one
/// [`Configuration`], both kept *signed*. A plain value bundle — reduction
/// downstream takes the absolute values and moments it needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Total energy `H` of the configuration (from [`Action::energy`]).
    pub energy: f64,
    /// Total magnetization `M = sum_i s_i` — the raw signed spin sum, not `|M|`
    /// and not a density (from [`Ising::magnetization`]).
    pub magnetization: f64,
}

/// Measure one `config` of the Ising `model` on `lattice` into a [`Sample`].
///
/// The two primaries come from the two seams the model exposes: `E` from the
/// [`Action`] trait, `M` from the inherent [`Ising::magnetization`].
pub fn measure<const D: usize>(
    model: &Ising,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
) -> Sample {
    Sample {
        energy: model.energy(lattice, config),
        magnetization: model.magnetization(config),
    }
}

/// Measure the per-config two-point [`Correlator`] `C_r = (1/N) Σ_i s_i · s_{i+r}`
/// of an Ising `config` on `lattice`.
///
/// Composes [`Ising::correlator`] the same way [`measure`] composes energy and
/// magnetization: the model owns the computation, this layer owns the record.
/// The entries are raw rather than connected — the `− ⟨s⟩²` subtraction is a
/// function of the ensemble mean and so belongs downstream.
pub fn correlator<const D: usize>(
    model: &Ising,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
) -> Correlator<D> {
    Correlator {
        per_axis: model.correlator(lattice, config),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deltas_match_from_scratch;

    #[test]
    fn correlator_of_aligned_config_is_all_ones() {
        // Every spin +1: s_i · s_{i+r} = 1 for every site and displacement, so
        // C_r = 1 on every axis, including the C_0 normalization.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat, Cell::Site);

        let c = action.correlator(&lat, &config);
        assert_eq!(c.len(), 2); // one row per axis
        for row in &c {
            assert_eq!(row.len(), 3); // r = 0..=L/2, i.e. L/2 + 1 = 3 for L = 4
            assert!(row.iter().all(|&v| v == 1.0));
        }
    }

    #[test]
    fn correlator_of_checkerboard_alternates() {
        // Checkerboard (spin = (−1)^(x+y)): a one-step move along either axis
        // flips the spin, so C_r = (−1)^r. The r = 2 shift exercises the periodic
        // wrap for sites near the boundary; L = 4 is even, so the pattern is
        // consistent under those boundaries.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        let down = State::new(1).unwrap();
        for site in 0..lat.n_sites() {
            let x = lat.site_coords(site);
            if (x[0] + x[1]) % 2 == 1 {
                config.poke(site, down);
            }
        }

        let c = action.correlator(&lat, &config);
        let expected = [1.0, -1.0, 1.0]; // (−1)^r for r = 0..=L/2 = 0, 1, 2
        assert_eq!(c[0], expected);
        assert_eq!(c[1], expected);
    }

    #[test]
    fn magnetization_is_the_signed_spin_sum() {
        // 4x4 = 16 sites. Cold is all state 0 (+1), so M = +N.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(action.magnetization(&config), 16.0);

        // 4 up, 12 down -> M = 4 − 12 = −8: the sign is kept, not folded to |M|.
        let down = State::new(1).unwrap();
        for site in 0..12 {
            config.poke(site, down);
        }
        assert_eq!(action.magnetization(&config), -8.0);
    }

    #[test]
    fn cold_start_is_the_ground_state_energy() {
        // All spins +1: every bond contributes +1. A D-dim periodic lattice has
        // D forward bonds per site, so bond_sum = D * N and E = -j * D * N.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let action = Ising::new(1.0, 0.0);
        assert_eq!(action.energy(&lat, &config), -32.0); // -1 * 2 * 16
    }

    #[test]
    fn field_term_tracks_total_magnetization() {
        // Cold (all +1) with j = 0 isolates the field term: E = -h * N.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let action = Ising::new(0.0, 0.5);
        assert_eq!(action.energy(&lat, &config), -8.0); // -0.5 * 16
    }

    #[test]
    fn energy_delta_matches_from_scratch_difference() {
        // j = 1.0 and h = 0.5 are exactly representable and the sums are integer,
        // so the two sides agree bit-for-bit and bare `==` is legitimate here.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();

        // A non-uniform configuration so neighbor sums actually vary.
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        for &s in &[5usize, 6, 10] {
            config.poke(s, down);
        }

        for &site in &[0usize, 5, 6, 9, 10] {
            let proposed = if config.peek(site) == up { down } else { up };
            let before = action.energy(&lat, &config);
            let delta = action.energy_delta(&lat, &config, site, proposed);

            let mut after = config.clone();
            after.poke(site, proposed);
            assert_eq!(delta, action.energy(&lat, &after) - before);
        }
    }

    /// The dimensions both models are checked at, up to ten.
    ///
    /// Ten is a ceiling on what anyone would plausibly run rather than anything
    /// the code knows about — nothing in the library states an upper bound, and
    /// `Lattice<12>` would compile. The point of going this far is that every
    /// count the incidence tables are built from grows with `D`, some of them
    /// quadratically: at ten dimensions a site has twenty neighbors, a link sits
    /// in eighteen plaquettes, and each site anchors forty-five of them.
    ///
    /// The unequal extents matter as much as the dimensions. A cubic shape hides
    /// a transposed stride, since every axis then has the same place value.
    #[test]
    fn ising_deltas_match_from_scratch_in_every_dimension() {
        let action = Ising::new(1.0, 0.5);
        deltas_match_from_scratch([6], &action, Cell::Site);
        deltas_match_from_scratch([4, 4], &action, Cell::Site);
        deltas_match_from_scratch([3, 4, 5], &action, Cell::Site);
        deltas_match_from_scratch([2, 3, 2, 3], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 3, 2, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 3, 2, 2, 3, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 2, 3, 2, 2, 2, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 2, 2, 3, 2, 2, 2, 2, 2], &action, Cell::Site);
    }

    #[test]
    fn energy_delta_is_zero_for_the_current_state() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let same = config.peek(7);
        assert_eq!(action.energy_delta(&lat, &config, 7, same), 0.0);
    }

    // Debug-only, like its gauge twin below: the guard it asserts on is a
    // `debug_assert!`, which release builds compile out.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Ising spins live on sites")]
    fn the_ising_action_rejects_a_link_field() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        Ising::new(1.0, 0.0).energy(&lat, &config);
    }

    #[test]
    fn measure_bundles_energy_and_magnetization() {
        // Cold 4x4, j = 1, h = 0: the action tests already pin E = -2jN = -32 and
        // M = +N = 16, and measure must carry those through unchanged.
        let lat = Lattice::new([4, 4]);
        let model = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat, Cell::Site);

        let sample = measure(&model, &lat, &config);
        assert_eq!(sample.energy, model.energy(&lat, &config));
        assert_eq!(sample.magnetization, model.magnetization(&config));
        assert_eq!(sample.energy, -32.0);
        assert_eq!(sample.magnetization, 16.0);
    }

    #[test]
    fn measure_keeps_the_magnetization_sign() {
        // A down-majority config: M must come back negative (signed sum), not
        // folded to |M|.
        let lat = Lattice::new([4, 4]);
        let model = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        let down = State::new(1).unwrap();
        for site in 0..12 {
            config.poke(site, down);
        }

        let sample = measure(&model, &lat, &config);
        assert_eq!(sample.magnetization, -8.0);
    }

    #[test]
    fn correlator_wraps_the_model_measurement() {
        // Cold gives C_r = 1 everywhere, so the record is all ones.
        let lat = Lattice::new([4, 4]);
        let model = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat, Cell::Site);

        let c = correlator(&model, &lat, &config);
        assert_eq!(c.per_axis, model.correlator(&lat, &config));
        assert_eq!(c.per_axis[0].len(), 3); // r = 0..=L/2 on axis 0 (L = 4)
        for row in &c.per_axis {
            assert!(row.iter().all(|&v| v == 1.0));
        }
    }
}
