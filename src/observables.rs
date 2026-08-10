//! Observables: measure a single [`Configuration`] into a per-config record.
//!
//! This is the `measure` a driver calls at each sample point: a pure function of
//! one configuration, with no history, ensemble, RNG, or chain state behind it.
//! There is deliberately no `Observable` trait — the swappable seam is already
//! the driver's injected measurement closure.
//!
//! A record holds only the *primaries*: energy and magnetization for [`Ising`],
//! energy and the plaquette sum for [`Z2Gauge`], which has no magnetization to
//! hold, energy and two order parameters for [`Potts`], whose two conventions
//! are not interchangeable and neither recoverable from the other. Absolute
//! values,
//! densities, and moments are ensemble reductions rather than functions of one
//! config, so they belong to [`statistics`](crate::statistics), which recovers
//! both `<m²>` and `<|m|>` from the signed `M` series kept here.
//!
//! Where a quantity has a sign it is kept, which is what leaves that recovery
//! possible. [`PottsSample`] is the exception and cannot be otherwise: its
//! labels are unordered, so its order parameter has no sign to keep and already
//! stands in the place `<|m|>` does.
//!
//! Each model gets its own record and its own composing function rather than one
//! generic set, because the three do not measure the same quantities and a record
//! wide enough for all of them would carry a hole for whichever model is running.
//! [`Correlator`] is the one record shared across models, because there the shape
//! and the reading of it genuinely are the same.
//!
//! A new observable goes here or on the model depending on whether it needs the
//! model's decode from a state index to a physical value. Structural ones
//! (domain walls, cluster sizes) depend only on configuration and geometry and
//! live here; value-semantic ones (magnetization, energy) live on the concrete
//! model beside the private decode they read, and `measure` composes them.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::{Action, Ising, Potts, Z2Gauge};

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

/// The per-config two-point correlator, one row per lattice axis, measured under
/// periodic boundaries.
///
/// `per_axis[μ][r]` is `C_r` for displacement `r = 0..=L_μ/2` along axis `μ`.
/// What one entry *means* comes from the model that filled it — `<s_i s_{i+r}>`
/// for [`Ising`] (see [`correlator`]), the connected agreement
/// `<delta(s_i, s_{i+r})> − 1/Q` for [`Potts`] (see [`potts_correlator`]) — and
/// one record serves both because how it is laid out and read is identical: a
/// row per axis, indexed by displacement, storing only the non-redundant half
/// since `C_r = C_{L_μ − r}` by translation invariance. A second type would
/// differ in nothing but its name.
///
/// This is a *separate* observable from [`Sample`] rather than a field of it: it
/// costs `O(N · L)` and is non-`Copy`, so keeping it out means runs that don't
/// want it don't pay for it.
///
/// The raw per-config estimator only — no ensemble average and no
/// correlation-length fit; both are reductions over a chain of these.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlator<const D: usize> {
    /// One row per axis: `per_axis[μ][r]` = `C_r` along axis `μ`,
    /// `r = 0..=L_μ/2`.
    pub per_axis: [Vec<f64>; D],
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

/// The per-config measurement record for the Potts model, the counterpart of
/// [`Sample`].
///
/// It carries an order parameter where [`Sample`] carries a magnetization, and
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
/// The primaries come from the seams the model exposes, as [`measure`]'s do: `E`
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
/// Composes [`Potts::correlator`] exactly as [`correlator`] composes the Ising
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

/// The per-config measurement record for the gauge model, the counterpart of
/// [`Sample`].
///
/// It has no magnetization, and no field can be added to give it one. The gauge
/// symmetry is local, so for every configuration there is an equally probable
/// one with any single link flipped, and the average of a lone link variable is
/// therefore exactly zero rather than merely small — Elitzur's theorem. A field
/// summing the links would produce a series fluctuating around zero forever,
/// which is why the confinement transition is read off closed loops instead
/// (see [`WilsonRectangles`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeSample {
    /// Total energy `H` of the configuration (from [`Action::energy`]).
    pub energy: f64,
    /// The signed plaquette sum `sum_□ σ_□` (from [`Z2Gauge::plaquette_sum`]).
    ///
    /// Redundant with `energy`, which is `-j` times it, and kept anyway: the
    /// mean plaquette is what the literature reports and what the exact
    /// two-dimensional result `tanh(β)` is stated for, while `energy` is what
    /// [`specific_heat`](crate::statistics::specific_heat) consumes. Neither
    /// caller should have to undo the other's scaling, and at `j = 0` only this
    /// one survives.
    pub plaquette_sum: f64,
}

/// Measure one `config` of the gauge `model` on `lattice` into a
/// [`GaugeSample`].
pub fn gauge_measure<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
) -> GaugeSample {
    GaugeSample {
        energy: model.energy(lattice, config),
        plaquette_sum: model.plaquette_sum(lattice, config),
    }
}

/// The per-config table of rectangular Wilson loops: `per_size[r][t]` is the
/// product of the link variables around an `r`-by-`t` rectangle, averaged over
/// every placement of that rectangle on the lattice.
///
/// Named for rectangles rather than for loops in general, because that is what
/// it enumerates. A Wilson loop is defined for *any* closed path, and
/// [`Z2Gauge::loop_product`] is that definition; this measures one particular
/// family of paths, the one whose average answers the confinement question.
///
/// This is to [`GaugeSample`] what [`Correlator`] is to [`Sample`] — the
/// separate, expensive one. It costs the rectangle count times the lattice
/// volume times the perimeter and it allocates, so runs that don't want it don't
/// pay for it.
///
/// The raw per-config estimator only. What the table is *for* is the comparison
/// downstream: if the chain average falls off like the exponential of `r * t`
/// the theory confines, and if it falls off with the perimeter instead it does
/// not. That fit belongs to statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct WilsonRectangles {
    /// `per_size[r][t]` for both sides running `0..=max_side`, symmetric in the
    /// two indices, with row and column `0` the trivial `1.0`.
    pub per_size: Vec<Vec<f64>>,
}

/// Measure the per-config [`WilsonRectangles`] table of `config` on `lattice`, with
/// both side lengths running up to `max_side`.
///
/// `max_side` is capped at half the smallest extent, past which a rectangle
/// wraps the torus and stops measuring what it is meant to.
pub fn wilson_rectangles<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
    max_side: usize,
) -> WilsonRectangles {
    WilsonRectangles {
        per_size: model.wilson_rectangles(lattice, config, max_side),
    }
}

/// Measure the per-config Polyakov loop of `config` along `dir` — the product
/// down a line wrapping that direction, averaged over the lines and kept signed.
///
/// One number rather than a record, so there is nothing to bundle; it is a free
/// function here rather than a field on [`GaugeSample`] because it costs a walk
/// of the whole lattice and only a run studying deconfinement wants it. Which
/// direction wraps is the caller's to say, since nothing in the geometry marks
/// one out as time.
pub fn polyakov_loop<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
    dir: usize,
) -> f64 {
    model.polyakov_loop(lattice, config, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::Cell;
    use crate::state::State;

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

    #[test]
    fn gauge_measure_bundles_energy_and_plaquette_sum() {
        // Cold 4x4x4 with j = 2: every plaquette product is +1, so the sum is
        // the plaquette count (3 planes * 64 sites) and E = -j times it.
        let lat = Lattice::new([4, 4, 4]);
        let model = Z2Gauge::new(2.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let sample = gauge_measure(&model, &lat, &config);
        assert_eq!(sample.plaquette_sum, 192.0);
        assert_eq!(sample.energy, -384.0);
        assert_eq!(sample.energy, -2.0 * sample.plaquette_sum);
    }

    #[test]
    fn the_plaquette_sum_survives_a_zero_coupling() {
        // The reason both fields are kept: at j = 0 the energy is identically
        // zero and carries nothing, while the plaquette sum still measures the
        // configuration.
        let lat = Lattice::new([4, 4]);
        let model = Z2Gauge::new(0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        config.poke(0, State::new(1).unwrap());

        let sample = gauge_measure(&model, &lat, &config);
        assert_eq!(sample.energy, 0.0);
        // Link 0 sits on two of the sixteen plaquettes, flipping both to -1.
        assert_eq!(sample.plaquette_sum, 12.0);
    }

    #[test]
    fn wilson_rectangles_wraps_the_model_measurement() {
        // Cold gives 1.0 at every size, and the record carries the model's own
        // table through unchanged.
        let lat = Lattice::new([6, 6]);
        let model = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let w = wilson_rectangles(&model, &lat, &config, 3);
        assert_eq!(w.per_size, model.wilson_rectangles(&lat, &config, 3));
        assert_eq!(w.per_size.len(), 4); // sides 0..=3
        assert!(w.per_size.iter().all(|row| row.iter().all(|&v| v == 1.0)));
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

    #[test]
    fn polyakov_loop_wraps_the_model_measurement() {
        // The one observables free function that lacked a wrapper test: it must
        // carry the model's own Polyakov value through unchanged. A cold link
        // field has every loop equal to 1, the torus-winding Polyakov line
        // included.
        let lat = Lattice::new([4, 4, 4]);
        let model = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        for dir in 0..3 {
            assert_eq!(
                polyakov_loop(&model, &lat, &config, dir),
                model.polyakov_loop(&lat, &config, dir)
            );
        }
        assert_eq!(polyakov_loop(&model, &lat, &config, 0), 1.0);
    }
}
