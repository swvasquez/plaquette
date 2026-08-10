//! Observables: the model-neutral records a measurement fills.
//!
//! A measurement is a pure function of one [`Configuration`](crate::configuration::Configuration):
//! no history, ensemble, RNG, or chain state behind it. There is deliberately
//! no `Observable` trait — the swappable seam is already the driver's injected
//! measurement closure — and each model owns its record and its composing
//! function (see [`models`](crate::models)), because the three do not measure
//! the same quantities and a record wide enough for all of them would carry a
//! hole for whichever model is running.
//!
//! What lives here is only what is genuinely shared across models:
//! [`Correlator`], whose shape and reading are identical whichever model fills
//! it. Absolute values, densities, and moments are ensemble reductions rather
//! than functions of one config, so they belong to
//! [`statistics`](crate::statistics).

/// The per-config two-point correlator, one row per lattice axis, measured under
/// periodic boundaries.
///
/// `per_axis[μ][r]` is `C_r` for displacement `r = 0..=L_μ/2` along axis `μ`.
/// What one entry *means* comes from the model that filled it —
/// `<s_i s_{i+r}>` for [`Ising`](crate::models::ising::Ising) (see
/// [`correlator`](crate::models::ising::correlator)), the connected agreement
/// `<delta(s_i, s_{i+r})> − 1/Q` for [`Potts`](crate::models::potts::Potts)
/// (see [`potts_correlator`](crate::models::potts::potts_correlator)) — and one
/// record serves both because how it is laid out and read is identical: a row
/// per axis, indexed by displacement, storing only the non-redundant half since
/// `C_r = C_{L_μ − r}` by translation invariance. A second type would differ in
/// nothing but its name.
///
/// This is a *separate* observable from the per-config records rather than a
/// field of one: it costs `O(N · L)` and is non-`Copy`, so keeping it out means
/// runs that don't want it don't pay for it.
///
/// The raw per-config estimator only — no ensemble average and no
/// correlation-length fit; both are reductions over a chain of these.
#[derive(Debug, Clone, PartialEq)]
pub struct Correlator<const D: usize> {
    /// One row per axis: `per_axis[μ][r]` = `C_r` along axis `μ`,
    /// `r = 0..=L_μ/2`.
    pub per_axis: [Vec<f64>; D],
}
