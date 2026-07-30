//! Model: the value *semantics* the lower layers withhold, and everything that
//! reads them — the energy, and the config-derived observables.
//!
//! Everything below this module is meaning-free: a [`State<Q>`] is an index, a
//! [`Configuration`] a flat array of indices, a [`Lattice`] geometry alone. The
//! model is the first place that says what the indices *are* — for Ising, that
//! `{0, 1}` map to spins `{+1, −1}`. That decode is the private `spin` map,
//! read by the energy and by observables like [`Ising::magnetization`], and
//! nothing outside this module sees spins.
//!
//! [`Action`] is the energy seam the updater depends on: energies only, never
//! spin values. It owns just the physics parameters, borrowing the lattice and
//! configuration per call, and keeps no running energy. Value-semantic
//! observables stay inherent methods on the concrete model rather than trait
//! methods, so the trait stays energy-only. [`Ising`] is the `Q = 2` case,
//! implemented for any `D`.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::state::State;

/// The energy functional the sampler is built around.
///
/// Generic over the state count `Q` and the lattice dimension `D`, so the
/// updater can name the seam without naming a specific model.
pub trait Action<const Q: usize, const D: usize> {
    /// The energy `H` of `config` on `lattice`, computed from scratch — a full
    /// lattice scan, not the hot path.
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<Q>) -> f64;

    /// The energy change `ΔE = H(after) − H(before)` of poking `site` to
    /// `proposed`, without mutating `config`.
    ///
    /// The sampler's hot path: it reads only the bonds incident to `site`, so it
    /// is `O(1)` in the lattice size rather than a rescan. It equals
    /// `energy(after) − energy(before)` by construction — exactly when the
    /// couplings and sums are integer-valued, up to rounding otherwise — and is
    /// the more accurate side of that comparison.
    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
        site: usize,
        proposed: State<Q>,
    ) -> f64;
}

/// The Ising model: spins `s_i = ±1` with nearest-neighbour coupling `j` and a
/// uniform external field `h`,
///
/// ```text
/// H = -j * sum_<ij> s_i s_j  -  h * sum_i s_i
/// ```
///
/// where the first sum runs over each nearest-neighbour bond once. `j > 0` is
/// ferromagnetic (aligned neighbours lower the energy). Energies come out in the
/// same units as `j` and `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ising {
    /// Nearest-neighbour coupling `J`.
    j: f64,
    /// Uniform external field `h` (set to `0.0` for the field-free model).
    h: f64,
}

impl Ising {
    /// An Ising action with coupling `j` and external field `h`.
    pub fn new(j: f64, h: f64) -> Self {
        Ising { j, h }
    }

    /// The total magnetization `M = sum_i s_i` of `config` — the raw *signed*
    /// spin sum, not `|M|` and not a density.
    ///
    /// Keeping the sign is what makes both `<m²>` and `<|m|>` recoverable from
    /// the series downstream. It reads the private `spin` map, so it is an
    /// inherent method rather than part of the energy-only [`Action`] trait.
    pub fn magnetization(&self, config: &Configuration<2>) -> f64 {
        // Integer accumulator: the sum is exact until the final cast.
        let spin_sum: i64 = config.variables().iter().map(|&s| spin(s) as i64).sum();
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
        let shape = lattice.shape();
        let n = config.n_sites();

        // Integer accumulators, one row per axis indexed by displacement `r`:
        // sum over sites of s_i · s_{i+r}, exact until the final divide by N.
        let mut sums: [Vec<i64>; D] = std::array::from_fn(|mu| vec![0i64; shape[mu] / 2 + 1]);
        for site in 0..n {
            let s_i = spin(config.peek(site));
            for (mu, row) in sums.iter_mut().enumerate() {
                for (r, cell) in row.iter_mut().enumerate() {
                    let j = lattice.shift(site, mu, r);
                    *cell += (s_i * spin(config.peek(j))) as i64;
                }
            }
        }

        std::array::from_fn(|mu| sums[mu].iter().map(|&c| c as f64 / n as f64).collect())
    }
}

/// Map a two-state index to its spin value: `0 → +1`, `1 → −1`. The whole of
/// the Ising value semantics, kept private to this module.
fn spin(state: State<2>) -> i32 {
    1 - 2 * state.index() as i32
}

impl<const D: usize> Action<2, D> for Ising {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        // Integer accumulators: the sums are exact until the final scaling.
        let mut bond_sum: i64 = 0; // sum over each bond once of s_i s_j
        let mut spin_sum: i64 = 0; // sum_i s_i, for the field term

        for site in 0..config.n_sites() {
            let s_i = spin(config.peek(site));
            spin_sum += s_i as i64;

            // The neighbour row is ordered +0, −0, +1, −1, ...; taking the
            // forward columns only (every other entry) visits each bond once.
            for &j_site in lattice.neighbors(site).iter().step_by(2) {
                bond_sum += (s_i * spin(config.peek(j_site))) as i64;
            }
        }

        -self.j * bond_sum as f64 - self.h * spin_sum as f64
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        site: usize,
        proposed: State<2>,
    ) -> f64 {
        // ΔE = -(s'_i - s_i) * (J * sum_{j in nbrs(i)} s_j + h). Only bonds
        // touching `site` and its field term change; everything else cancels.
        let ds = (spin(proposed) - spin(config.peek(site))) as i64;
        if ds == 0 {
            return 0.0; // proposed state equals the current one
        }

        // All 2D neighbours (both directions): every incident bond changes.
        let neighbor_sum: i64 = lattice
            .neighbors(site)
            .iter()
            .map(|&j_site| spin(config.peek(j_site)) as i64)
            .sum();

        -(ds as f64) * (self.j * neighbor_sum as f64 + self.h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spin_mapping_is_plus_minus_one() {
        assert_eq!(spin(State::new(0).unwrap()), 1);
        assert_eq!(spin(State::new(1).unwrap()), -1);
    }

    #[test]
    fn correlator_of_aligned_config_is_all_ones() {
        // Every spin +1: s_i · s_{i+r} = 1 for every site and displacement, so
        // C_r = 1 on every axis, including the C_0 normalization.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat);

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
        let mut config = Configuration::<2>::cold(&lat);
        let down = State::new(1).unwrap();
        for site in 0..lat.n_sites() {
            let x = lat.coords(site);
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
        let mut config = Configuration::<2>::cold(&lat);
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
        let config = Configuration::<2>::cold(&lat);
        let action = Ising::new(1.0, 0.0);
        assert_eq!(action.energy(&lat, &config), -32.0); // -1 * 2 * 16
    }

    #[test]
    fn field_term_tracks_total_magnetization() {
        // Cold (all +1) with j = 0 isolates the field term: E = -h * N.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat);
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

        // A non-uniform configuration so neighbour sums actually vary.
        let mut config = Configuration::<2>::cold(&lat);
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

    #[test]
    fn energy_delta_is_zero_for_the_current_state() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let config = Configuration::<2>::cold(&lat);
        let same = config.peek(7);
        assert_eq!(action.energy_delta(&lat, &config, 7, same), 0.0);
    }
}
