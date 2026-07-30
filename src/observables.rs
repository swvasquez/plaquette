//! Observables: measure a single [`Configuration`] into a per-config record.
//!
//! This is the `measure` a driver calls at each sample point: a pure function of
//! one configuration, with no history, ensemble, RNG, or chain state behind it.
//! There is deliberately no `Observable` trait — the swappable seam is already
//! the driver's injected measurement closure.
//!
//! The record holds only the *primaries*, signed energy and signed
//! magnetization. Absolute values, densities, and moments are ensemble
//! reductions rather than functions of one config, so they belong to
//! [`statistics`](crate::statistics), which recovers both `<m²>` and `<|m|>`
//! from the signed `M` series kept here.
//!
//! A new observable goes here or on the model depending on whether it needs the
//! model's decode from a state index to a physical value. Structural ones
//! (domain walls, cluster sizes) depend only on configuration and geometry and
//! live here; value-semantic ones (magnetization, energy) live on the concrete
//! model beside the private decode they read, and `measure` composes them.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::{Action, Ising};

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

/// The per-config two-point correlator `C_r = (1/N) Σ_i s_i · s_{i+r}`, measured
/// along each lattice axis under periodic boundaries.
///
/// `per_axis[μ][r]` is `C_r` for displacement `r = 0..=L_μ/2` along axis `μ`.
/// This is a *separate* observable from [`Sample`] rather than a field of it: it
/// costs `O(N · L)` and is non-`Copy`, so keeping it out means runs that don't
/// want it don't pay for it.
///
/// The raw per-config estimator only — no connected subtraction, no ensemble
/// average, no correlation-length fit. See [`Ising::correlator`] for the stored
/// half.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlator<const D: usize> {
    /// One row per axis: `per_axis[μ][r]` = `C_r` along axis `μ`,
    /// `r = 0..=L_μ/2`.
    pub per_axis: [Vec<f64>; D],
}

/// Measure the per-config two-point [`Correlator`] of `config` on `lattice`.
///
/// Composes [`Ising::correlator`] the same way [`measure`] composes energy and
/// magnetization: the model owns the computation, this layer owns the record.
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
    use crate::state::State;

    #[test]
    fn measure_bundles_energy_and_magnetization() {
        // Cold 4x4, j = 1, h = 0: the action tests already pin E = -2jN = -32 and
        // M = +N = 16, and measure must carry those through unchanged.
        let lat = Lattice::new([4, 4]);
        let model = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat);

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
        let mut config = Configuration::<2>::cold(&lat);
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
        let config = Configuration::<2>::cold(&lat);

        let c = correlator(&model, &lat, &config);
        assert_eq!(c.per_axis, model.correlator(&lat, &config));
        assert_eq!(c.per_axis[0].len(), 3); // r = 0..=L/2 on axis 0 (L = 4)
        for row in &c.per_axis {
            assert!(row.iter().all(|&v| v == 1.0));
        }
    }
}
