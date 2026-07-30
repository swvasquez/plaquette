//! Statistics: turn a measured *series* into estimates with honest error bars.
//!
//! This is the ensemble layer, the complement of
//! [`observables`](crate::observables). An observable is a function of one
//! configuration; everything here is a function of the whole series, so its
//! input is always a slice of numbers the consumer already collected, never a
//! configuration.
//!
//! There are two shapes. [`reduce`] is the *primary* reduction: every scalar
//! series goes through the identical call and comes back as an [`Estimate`],
//! whose `mean`, `stderr`, `tau_int`, and `n_eff` are facets of one
//! automatic-windowing computation. The *derived* reduction is a blocked
//! jackknife over per-block moment sums, returning a [`Derived`] that carries no
//! `tau_int`/`n_eff` — a fluctuation is a function of moments, not of a time
//! series, so an autocorrelation time is meaningless for it.
//!
//! The three fluctuation quantities ([`specific_heat`], [`susceptibility`],
//! [`binder_cumulant`]) build their moment feature columns and hand them to
//! `jackknife_features`, which forms each leave-one-block-out estimate as
//! `total − block_b` in O(1) — moments are additive over blocks, so the whole
//! jackknife is one O(N) pass rather than a re-scan per block. [`creutz_ratio`]
//! is derived for the same reason without being a fluctuation: it is a nonlinear
//! function of four separate ensemble means, so its error has to be propagated
//! through that function rather than averaged, which is exactly what the
//! jackknife does.

/// The reduction of one scalar series: a mean and its autocorrelation-corrected
/// error, with the diagnostics that say whether that error can be trusted.
///
/// `tau_int` is the integrated autocorrelation time and `n_eff = N / (2·tau_int)`
/// the effective sample count. A visible `n_eff` far below `N` is what tells a
/// user their error bars are untrustworthy — so these travel *attached to every
/// mean*, not as an opt-in extra.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    /// Sample mean of the series.
    pub mean: f64,
    /// Standard error on the mean, inflated by autocorrelation:
    /// `sqrt(2·tau_int·C(0)/N)`.
    pub stderr: f64,
    /// Integrated autocorrelation time, clamped to `≥ 0.5`.
    pub tau_int: f64,
    /// Effective number of independent samples, `N / (2·tau_int)`.
    pub n_eff: f64,
}

/// A derived (fluctuation) quantity and its jackknife error. Distinct from
/// [`Estimate`] because it carries no `tau_int`/`n_eff`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Derived {
    /// Full-series plug-in value of the estimator. Always computed from all the
    /// samples, so it is unaffected by how the blocking below turns out.
    pub value: f64,
    /// Jackknife standard error over blocks, or `NaN` when there were too few
    /// blocks to measure a scatter at all. Never zero-by-default: a zero here
    /// would read as perfect certainty.
    pub stderr: f64,
    /// How many blocks the error rests on — the jackknife's counterpart to
    /// [`Estimate::n_eff`], and the number that says whether `stderr` can be
    /// trusted. Fewer than 2 means `stderr` is `NaN`.
    pub n_blocks: usize,
}

/// The effective sample size below which an autocorrelation estimate cannot be
/// believed.
///
/// The usual lattice requirement is `N > 50·tau_int`, which is exactly
/// `n_eff > 25`. Below it `tau_int` stops being a measurement and becomes the
/// ceiling of what a run that length can perceive, since the windowing in
/// [`reduce`] caps it near `N/5` however correlated the data really is.
/// Everything downstream — `n_eff`, the jackknife block length, every `stderr` —
/// then comes out too optimistic with nothing in its value to show it.
pub const MIN_EFFECTIVE_SAMPLES: f64 = 25.0;

impl Estimate {
    /// Whether the run was long enough to measure its own autocorrelation. False
    /// means `tau_int` has saturated and `stderr` is understated; the fix is a
    /// longer run or a better updater, not a different estimator.
    pub fn is_reliable(&self) -> bool {
        self.n_eff >= MIN_EFFECTIVE_SAMPLES
    }
}

impl Derived {
    /// Whether enough independent blocks went into `stderr` to believe it. The
    /// block count is this layer's effective sample size, so it is held to the
    /// same [`MIN_EFFECTIVE_SAMPLES`] threshold as [`Estimate::is_reliable`].
    pub fn is_reliable(&self) -> bool {
        self.n_blocks as f64 >= MIN_EFFECTIVE_SAMPLES
    }
}

// --- base operations: general reductions over a slice, with no physics in them ---

/// Reduce a scalar series to its mean and autocorrelation-corrected error.
///
/// The automatic-windowing procedure (Sokal / Madras–Sokal): build the
/// normalized autocorrelation curve `rho(t)`, accumulate the running window sum
/// `tau(W) = 0.5 + Σ_{t=1}^{W} rho(t)`, and truncate at the smallest `W ≥ 1`
/// with `W ≥ c·tau(W)` (`c = 5`), cutting off the noisy tail. The autocovariance
/// uses the biased `/N` normalization the windowing assumes, which damps that
/// tail.
///
/// A constant series returns zero error with `tau_int = 0.5`; so does a series
/// shorter than two points, which cannot show autocorrelation at all. `tau_int`
/// is clamped to `≥ 0.5` throughout.
pub fn reduce(series: &[f64]) -> Estimate {
    let n = series.len();
    if n < 2 {
        let mean = if n == 1 { series[0] } else { 0.0 };
        return Estimate {
            mean,
            stderr: 0.0,
            tau_int: 0.5,
            n_eff: n as f64,
        };
    }

    let nf = n as f64;
    let mean = sample_mean(series);

    // Autocovariance at lag 0: the variance, /N normalized.
    let c0 = series.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / nf;
    if c0 == 0.0 {
        // Constant series: no scale to normalize by, no fluctuation to correct.
        return Estimate {
            mean,
            stderr: 0.0,
            tau_int: 0.5,
            n_eff: nf,
        };
    }

    // Accumulate the window sum lag by lag, stopping as soon as the window
    // closes (W ≥ c·tau(W)). If it never closes up to the largest lag N−1, keep
    // the widest window's value.
    let c = 5.0;
    let mut tau = 0.5;
    let mut tau_int = 0.5;
    for t in 1..n {
        let ct = series[..n - t]
            .iter()
            .zip(&series[t..])
            .map(|(&a, &b)| (a - mean) * (b - mean))
            .sum::<f64>()
            / nf;
        tau += ct / c0;
        tau_int = tau; // widest-window fallback if the window never closes
        if t as f64 >= c * tau {
            break;
        }
    }

    if tau_int < 0.5 {
        tau_int = 0.5;
    }

    let n_eff = nf / (2.0 * tau_int);
    let stderr = (2.0 * tau_int * c0 / nf).sqrt();
    Estimate {
        mean,
        stderr,
        tau_int,
        n_eff,
    }
}

/// Estimate a derived quantity and its error by a *blocked* jackknife, over
/// per-block feature sums rather than re-scanned slices.
///
/// Every fluctuation here is a function of raw power-moments of one series
/// (`⟨x²⟩ − ⟨x⟩²`, `1 − ⟨x⁴⟩/(3⟨x²⟩²)`), and moments are additive over blocks —
/// so leave-one-block-out is `total − block_b`, an O(1) update per block after a
/// single O(N) accumulation. The old slice-based form rebuilt an O(N) leave-out
/// copy per block, making a run O(N · n_blocks); at 400k samples that was minutes.
/// This is O(N · n_features) with `n_features ≤ 3`.
///
/// `features` are aligned columns of length N — the caller supplies whatever
/// moments (and centering) its estimator needs. `estimator` receives the vector
/// of *leave-out means*, one per column, and returns the derived value for that
/// leave-out set. The blocked jackknife itself is otherwise unchanged: `n_b =
/// N / block_len` whole blocks, remainder dropped, `value` the full-series
/// plug-in, error `sqrt((n_b−1)/n_b · Σ_b (theta_(b) − theta_bar)²)`, and the
/// same refusal — `NaN` with `n_blocks` when there are fewer than two blocks.
///
/// No general slice-based form is kept: every caller is a moment estimator, and
/// leaving a known-O(N·n_b) path around invites misuse. A genuinely non-moment
/// estimator (one that isn't additive over blocks) would justify bringing it back.
///
/// # Contract: the caller owns the block length
///
/// `block_len` is a physics decision — it must be at least the correlation time
/// of *every* column the estimator consumes, since a combination of moments
/// decorrelates only as fast as its slowest integrand. Use [`block_len_for`] to
/// take the max over them. Blocks longer than needed widen the error (wasteful
/// but honest); blocks shorter than the correlation time understate it, the
/// dangerous direction.
fn jackknife_features(
    features: &[&[f64]],
    block_len: usize,
    estimator: impl Fn(&[f64]) -> f64,
) -> Derived {
    let n_features = features.len();
    let n = features[0].len();

    // The full-series plug-in uses all N samples (the remainder included), so the
    // reported value is exactly what the pre-refactor slice form returned.
    let total_n: Vec<f64> = features.iter().map(|col| col.iter().sum::<f64>()).collect();
    let value = {
        let means: Vec<f64> = total_n.iter().map(|s| s / n as f64).collect();
        estimator(&means)
    };

    let block_len = block_len.max(1);
    let n_b = n / block_len;
    if n_b < 2 {
        return Derived {
            value,
            stderr: f64::NAN,
            n_blocks: n_b,
        };
    }

    // One O(N) pass per column: the per-block partial sums over the whole-block
    // prefix, and their total. The remainder past `used` is dropped, matching the
    // old form's block handling exactly.
    let used = block_len * n_b;
    let mut partials = vec![vec![0.0; n_features]; n_b];
    let mut total_used = vec![0.0; n_features];
    for (j, col) in features.iter().enumerate() {
        for (b, partial) in partials.iter_mut().enumerate() {
            let lo = b * block_len;
            let sum: f64 = col[lo..lo + block_len].iter().sum();
            partial[j] = sum;
            total_used[j] += sum;
        }
    }

    // Leave-one-block-out is total − block_b, in O(1) per column per block.
    let leave_n = (used - block_len) as f64;
    let mut means = vec![0.0; n_features];
    let mut thetas = Vec::with_capacity(n_b);
    for partial in &partials {
        for j in 0..n_features {
            means[j] = (total_used[j] - partial[j]) / leave_n;
        }
        thetas.push(estimator(&means));
    }

    let theta_bar = sample_mean(&thetas);
    let factor = (n_b as f64 - 1.0) / n_b as f64;
    let spread = thetas.iter().map(|&t| (t - theta_bar).powi(2)).sum::<f64>();
    Derived {
        value,
        stderr: (factor * spread).sqrt(),
        n_blocks: n_b,
    }
}

// --- instantiations: one physics formula each, over its moment features ---

/// Block length for an estimator, sized by the slowest series it consumes.
///
/// A fluctuation depends on several moments at once (e.g. `⟨E⟩` and `⟨E²⟩`), and
/// the combination decorrelates only as fast as its slowest integrand — near
/// `T_c` the higher moment lags the lower one. So take the max `tau_int` over all
/// of them and block by `ceil(2·tau)`. This can only *lengthen* blocks versus any
/// single series, so it never understates the error.
///
/// This is the pragmatic bound. The tight version is the linearized (Gamma-method)
/// autocorrelation of the derived quantity — project each sample onto the
/// estimator's gradient and take that series' `tau_int` — which can be shorter
/// than the max but never understates it. Not implemented here; this is the safe,
/// much simpler improvement.
fn block_len_for(series: &[&[f64]]) -> usize {
    let tau = series.iter().map(|s| reduce(s).tau_int).fold(0.5, f64::max);
    ((2.0 * tau).ceil() as usize).max(1)
}

/// Specific heat `C = beta²·(⟨E²⟩ − ⟨E⟩²)/N` from the energy series.
///
/// `energies` is the series of *total* energies `E = Sample::energy`; `n_sites`
/// is `N`. Its integrands are `E` and `E²`, so the blocks are sized by whichever
/// of the two decorrelates more slowly.
///
/// The features are centered by the global mean `Ē`: with `f0 = E − Ē` and
/// `f1 = (E − Ē)²`, the estimator `β²·(⟨f1⟩ − ⟨f0⟩²)/N` is exactly `β²·Var(E)/N`
/// for any center, but centering keeps every leave-out `⟨f0⟩ ≈ 0`, so the
/// variance subtraction never differences two large nearly-equal numbers — the
/// same stable center-then-square numerics the pre-refactor form used.
pub fn specific_heat(energies: &[f64], beta: f64, n_sites: f64) -> Derived {
    let mean_e = sample_mean(energies);
    let f0: Vec<f64> = energies.iter().map(|e| e - mean_e).collect();
    let f1: Vec<f64> = f0.iter().map(|d| d * d).collect();
    // Block length is sized from the raw integrands E and E² (tau_int of the
    // centered square differs), while the jackknife runs on the centered features.
    let e_squared: Vec<f64> = energies.iter().map(|e| e * e).collect();
    let block_len = block_len_for(&[energies, &e_squared]);
    jackknife_features(&[&f0, &f1], block_len, move |m| {
        beta * beta * (m[1] - m[0] * m[0]).max(0.0) / n_sites
    })
}

/// Susceptibility `chi = beta·N·(⟨m²⟩ − ⟨|m|⟩²)` from the magnetization series.
///
/// `magnetizations` is the series of *signed total* `M = Sample::magnetization`.
/// Since `m² = |m|²`, the bracket is exactly the variance of `|m|` and is
/// computed as one — which also means this estimator never sees the sign.
///
/// The series is reduced to `|m| = |M|/N`, so the sign-flip time (orders of
/// magnitude longer at low temperature, and irrelevant to an even estimator)
/// never enters. Its integrands are `|m|` and `m²`, so the blocks are sized by
/// the slower of those, and the features are centered exactly as in
/// [`specific_heat`] for the same stability.
pub fn susceptibility(magnetizations: &[f64], beta: f64, n_sites: f64) -> Derived {
    let abs_m: Vec<f64> = magnetizations.iter().map(|m| (m / n_sites).abs()).collect();
    let mean_abs = sample_mean(&abs_m);
    let f0: Vec<f64> = abs_m.iter().map(|a| a - mean_abs).collect();
    let f1: Vec<f64> = f0.iter().map(|d| d * d).collect();
    // Raw integrands |m| and m² size the blocks; centered features run the jackknife.
    let m_squared: Vec<f64> = abs_m.iter().map(|a| a * a).collect();
    let block_len = block_len_for(&[&abs_m, &m_squared]);
    jackknife_features(&[&f0, &f1], block_len, move |m| {
        beta * n_sites * (m[1] - m[0] * m[0]).max(0.0)
    })
}

/// Binder cumulant `U_4 = 1 − ⟨m⁴⟩/(3·⟨m²⟩²)` from the magnetization series.
///
/// `U_4` is scale-invariant — the `1/N` factors in `m = M/N` cancel in the ratio
/// — so no division by `n_sites` is needed and none is taken.
///
/// Both moments are even in `M`, so as with [`susceptibility`] the series is
/// reduced to `|M|`. Its integrands are the raw moments `m²` and `m⁴`, so those
/// are the features directly (no centering — a ratio of two similar-magnitude
/// positives is well-conditioned), and the blocks are sized by the slower of them.
pub fn binder_cumulant(magnetizations: &[f64]) -> Derived {
    let m_squared: Vec<f64> = magnetizations.iter().map(|m| m * m).collect();
    let m_fourth: Vec<f64> = m_squared.iter().map(|s| s * s).collect();
    let block_len = block_len_for(&[&m_squared, &m_fourth]);
    jackknife_features(&[&m_squared, &m_fourth], block_len, |m| {
        1.0 - m[1] / (3.0 * m[0] * m[0])
    })
}

/// Creutz ratio `chi(R,T)` — the string tension read off four adjacent Wilson
/// loops, one series per loop size.
///
/// The logarithm of a Wilson loop average splits into an area term, a perimeter
/// self-energy term, and a constant,
/// `−log⟨W(R,T)⟩ = sigma·R·T + mu·(R+T) + c`, and only the first of the three is
/// the string tension. The 2×2 combination
///
/// ```text
/// chi(R,T) = −log[ ⟨W(R,T)⟩·⟨W(R−1,T−1)⟩ / ( ⟨W(R−1,T)⟩·⟨W(R,T−1)⟩ ) ]
/// ```
///
/// cancels the other two exactly — the perimeters sum to `(R+T) + (R+T−2) −
/// (R+T−1) − (R+T−1) = 0` and the constants cancel in pairs — while the areas
/// leave `R·T + (R−1)(T−1) − (R−1)T − R(T−1) = 1`, so what is left is `sigma`
/// plus whatever short-distance corrections the loops are still small enough to
/// carry. Those corrections die out as the loops grow, which is why `chi` is
/// reported per `(R,T)`: reading a plateau off the table is the caller's
/// judgement, not this function's.
///
/// The four arguments are the series for `(R,T)`, `(R−1,T−1)`, `(R−1,T)` and
/// `(R,T−1)` in that order, sample-aligned and equally long — they come from the
/// same chain, and the jackknife leaves out the same block of *configurations*
/// from all four at once, which is what keeps their shared fluctuations
/// correlated in the error rather than adding in quadrature. A trivial side
/// (`R−1 = 0`) is the constant `1.0` series, so `chi(1,1)` reduces to
/// `−log⟨W(1,1)⟩`, the mean plaquette.
///
/// This is not a mean of anything measured per configuration but a nonlinear
/// function of four means, so the four raw series are the features directly. No
/// centering: Wilson averages are positive and of similar magnitude in the
/// confined phase, so the ratio is well-conditioned, as in [`binder_cumulant`].
/// When the ratio is not a positive finite number — a near-deconfined or simply
/// too-noisy run, where a loop average can wander to zero or through it — the
/// logarithm has no value to give and the estimator returns `NaN` rather than a
/// clamped stand-in. That is the honest report that `chi` could not be resolved
/// at this loop size, and it propagates: a single refusing leave-out block makes
/// `stderr` `NaN` too.
pub fn creutz_ratio(w_rt: &[f64], w_r1t1: &[f64], w_r1t: &[f64], w_rt1: &[f64]) -> Derived {
    debug_assert!(
        [w_r1t1, w_r1t, w_rt1].iter().all(|s| s.len() == w_rt.len()),
        "the four Wilson series must come from the same chain, sample for sample"
    );

    let block_len = block_len_for(&[w_rt, w_r1t1, w_r1t, w_rt1]);
    jackknife_features(&[w_rt, w_r1t1, w_r1t, w_rt1], block_len, |m| {
        let ratio = m[0] * m[1] / (m[2] * m[3]);
        if ratio > 0.0 && ratio.is_finite() {
            -ratio.ln()
        } else {
            f64::NAN
        }
    })
}

/// Plain arithmetic mean of a series.
fn sample_mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean of a per-sample transform: apply `f` to each sample, *then* average.
    /// Test-side reference for the direct moment formulas.
    fn sample_mean_map(xs: &[f64], f: impl Fn(f64) -> f64) -> f64 {
        xs.iter().map(|&x| f(x)).sum::<f64>() / xs.len() as f64
    }

    /// The plain blocked-mean jackknife, as a single-feature call — the identity
    /// estimator over one column. Stands in for the removed slice-based
    /// `jackknife_blocks` in the tests that exercise the blocking mechanism itself.
    fn jack_mean(series: &[f64], block_len: usize) -> Derived {
        jackknife_features(&[series], block_len, |m| m[0])
    }

    /// Deterministic pseudo-random base for the correlation tests: splitmix64
    /// mapped to `[-1, 1)`. Close enough to independent that a bare (`k = 1`)
    /// series reduces to `tau_int ≈ 0.5`.
    fn base_value(i: usize) -> f64 {
        let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    /// A length-`total` series where each base value is repeated `k` times. The
    /// length is fixed as `k` grows, so a rising `tau_int` shows directly as a
    /// falling `n_eff`.
    fn run_correlated(total: usize, k: usize) -> Vec<f64> {
        (0..total).map(|i| base_value(i / k)).collect()
    }

    #[test]
    fn reduce_on_independent_series_gives_half_and_full_n() {
        // A strictly alternating series has no positive autocorrelation, so the
        // window closes at once and tau_int clamps to its floor of 0.5, leaving
        // every sample independent: n_eff == N.
        let n = 512;
        let series: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        let est = reduce(&series);
        assert!(
            (est.tau_int - 0.5).abs() < 1e-9,
            "tau_int = {}",
            est.tau_int
        );
        assert!((est.n_eff - n as f64).abs() < 1e-9, "n_eff = {}", est.n_eff);
    }

    #[test]
    fn reduce_on_constant_series_has_zero_error() {
        let series = vec![3.0; 128];
        let est = reduce(&series);
        assert_eq!(est.stderr, 0.0);
        assert_eq!(est.tau_int, 0.5);
        assert_eq!(est.n_eff, 128.0);
        assert_eq!(est.mean, 3.0);
    }

    #[test]
    fn reduce_tau_rises_and_n_eff_falls_with_run_length() {
        // Longer runs of repeated values mean slower decorrelation: tau_int must
        // rise and n_eff must fall as k grows. Assert the ordering, not values.
        let total = 1024;
        let e1 = reduce(&run_correlated(total, 1));
        let e4 = reduce(&run_correlated(total, 4));
        let e16 = reduce(&run_correlated(total, 16));

        assert!(
            e1.tau_int < e4.tau_int && e4.tau_int < e16.tau_int,
            "tau_int not increasing: {} {} {}",
            e1.tau_int,
            e4.tau_int,
            e16.tau_int
        );
        assert!(
            e1.n_eff > e4.n_eff && e4.n_eff > e16.n_eff,
            "n_eff not decreasing: {} {} {}",
            e1.n_eff,
            e4.n_eff,
            e16.n_eff
        );
    }

    #[test]
    fn jackknife_value_matches_direct_plug_in() {
        // The central value must be the full-series estimator, exactly — checked
        // against the specific-heat formula computed directly.
        let energies: Vec<f64> = (0..200).map(|i| base_value(i) * 10.0).collect();
        let (beta, n_sites) = (0.4, 64.0);

        let mean_e = sample_mean(&energies);
        let mean_e2 = sample_mean_map(&energies, |x| x * x);
        let direct = beta * beta * (mean_e2 - mean_e * mean_e) / n_sites;

        let d = specific_heat(&energies, beta, n_sites);
        assert!(
            (d.value - direct).abs() < 1e-9,
            "value = {}, direct = {}",
            d.value,
            direct
        );
        assert!(d.stderr > 0.0, "stderr = {}", d.stderr);
    }

    #[test]
    fn jackknife_refuses_rather_than_understating() {
        // With too little data to form two blocks there is no scatter to measure.
        // The value must still come back, but the error must be NaN — not zero,
        // which would read as perfect certainty.
        let one = [42.0];
        let d = jack_mean(&one, block_len_for(&[&one]));
        assert!(d.n_blocks < 2, "n_blocks = {}", d.n_blocks);
        assert!(d.stderr.is_nan(), "stderr = {}", d.stderr);
        assert_eq!(d.value, 42.0);
    }

    #[test]
    fn jackknife_reports_the_block_count_it_used() {
        // An uncorrelated series blocks at length 1, so every sample is its own
        // block and the count matches the series length.
        let series: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let d = jack_mean(&series, block_len_for(&[&series]));
        assert_eq!(d.n_blocks, 200);
        assert!(d.stderr > 0.0);
    }

    #[test]
    fn block_len_is_driven_by_the_slowest_series() {
        // One series churns every sample (tau ~ 0.5), the other moves in long
        // runs (large tau). The block length must follow the slow one — its
        // 2*tau — not the fast one, so an estimator over both stays honest.
        let fast: Vec<f64> = (0..1024)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let slow: Vec<f64> = (0..1024).map(|i| base_value(i / 16)).collect();

        let slow_tau = reduce(&slow).tau_int;
        let expected = ((2.0 * slow_tau).ceil() as usize).max(1);
        assert!(
            slow_tau > 3.0,
            "slow series not correlated enough: {slow_tau}"
        );
        assert_eq!(block_len_for(&[&fast, &slow]), expected);
        assert_eq!(block_len_for(&[&slow, &fast]), expected); // order-independent
    }

    #[test]
    fn saturated_autocorrelation_is_flagged_unreliable() {
        // A series that flips once in 1000 samples never decorrelated within the
        // run, so its true tau_int is unmeasurable and the windowing reports a
        // ceiling near N/5. The flag is what says so.
        let one_flip: Vec<f64> = (0..1000)
            .map(|i| if i < 500 { 1.0 } else { -1.0 })
            .collect();
        let est = reduce(&one_flip);
        assert!(est.n_eff < MIN_EFFECTIVE_SAMPLES, "n_eff = {}", est.n_eff);
        assert!(!est.is_reliable());
        assert!(!jack_mean(&one_flip, block_len_for(&[&one_flip])).is_reliable());

        // A series of the same length that genuinely decorrelates passes.
        let independent: Vec<f64> = (0..1000)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let good = reduce(&independent);
        assert!(good.is_reliable(), "n_eff = {}", good.n_eff);
        assert!(jack_mean(&independent, block_len_for(&[&independent])).is_reliable());
    }

    #[test]
    fn block_count_collapses_on_correlated_data() {
        // Both series are 200 samples long, but the correlated one supports only
        // a handful of independent blocks. Its error bar rests on those few, and
        // without `n_blocks` exposed you could not tell.
        let independent: Vec<f64> = (0..200)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let one_flip: Vec<f64> = (0..200).map(|i| if i < 100 { 1.0 } else { -1.0 }).collect();

        let a = jack_mean(&independent, block_len_for(&[&independent]));
        let b = jack_mean(&one_flip, block_len_for(&[&one_flip]));
        assert_eq!(a.n_blocks, 200);
        assert!(b.n_blocks < 10, "n_blocks = {}", b.n_blocks);
    }

    #[test]
    fn fluctuations_block_on_magnitude_not_sign() {
        // Two timescales deliberately separated: the sign flips only every 500
        // samples, while the magnitude churns every sample. Both fluctuation
        // estimators use even moments only, so they are blind to the sign — the
        // block length must follow the magnitude's short correlation, not the
        // sign's long one.
        let n_sites = 64.0;
        let signed: Vec<f64> = (0..2000)
            .map(|i| {
                let sign = if (i / 500) % 2 == 0 { 1.0 } else { -1.0 };
                sign * n_sites * (0.5 + 0.1 * base_value(i))
            })
            .collect();
        let magnitude: Vec<f64> = signed.iter().map(|m| m.abs()).collect();

        // The two series really do have wildly different correlation times.
        let (s, m) = (reduce(&signed), reduce(&magnitude));
        assert!(
            s.tau_int > 10.0 * m.tau_int,
            "timescales not separated: signed {} vs magnitude {}",
            s.tau_int,
            m.tau_int
        );

        // Blocking on the signed series would allow only this many blocks.
        let sign_driven = signed.len() / ((2.0 * s.tau_int).ceil() as usize);

        let chi = susceptibility(&signed, 0.4, n_sites);
        let u4 = binder_cumulant(&signed);
        for (name, d) in [("chi", chi), ("U_4", u4)] {
            assert!(
                d.n_blocks > 10 * sign_driven,
                "{name} blocked on the sign: {} blocks vs {} sign-driven",
                d.n_blocks,
                sign_driven
            );
            assert!(d.is_reliable(), "{name} n_blocks = {}", d.n_blocks);
        }

        // The central values are plug-ins over the full series and never touched
        // the blocking, so they must equal the direct formulas on signed data.
        let mean_m2 = sample_mean_map(&signed, |x| (x / n_sites).powi(2));
        let mean_abs = sample_mean_map(&signed, |x| (x / n_sites).abs());
        let chi_direct = 0.4 * n_sites * (mean_m2 - mean_abs * mean_abs);
        assert!(
            (chi.value - chi_direct).abs() < 1e-9,
            "{} vs {}",
            chi.value,
            chi_direct
        );

        let m2 = sample_mean_map(&signed, |x| x.powi(2));
        let m4 = sample_mean_map(&signed, |x| x.powi(4));
        let u4_direct = 1.0 - m4 / (3.0 * m2.powi(2));
        assert!(
            (u4.value - u4_direct).abs() < 1e-12,
            "{} vs {}",
            u4.value,
            u4_direct
        );
    }

    #[test]
    fn block_length_change_leaves_central_values_untouched() {
        // The block length only sizes the error; `value` is a full-series plug-in.
        // On a fixed correlated series, all three must equal the direct formulas
        // regardless of how the blocks come out.
        let (beta, n_sites) = (0.4407, 64.0);
        let signed: Vec<f64> = (0..3000)
            .map(|i| {
                let sign = if (i / 500) % 2 == 0 { 1.0 } else { -1.0 };
                sign * n_sites * (0.6 + 0.15 * base_value(i / 3))
            })
            .collect();

        let e: Vec<f64> = signed
            .iter()
            .map(|m| -1.9 * n_sites + 0.3 * m.abs())
            .collect();
        let mean_e = sample_mean(&e);
        let c_direct = beta * beta * sample_mean_map(&e, |x| (x - mean_e).powi(2)) / n_sites;
        assert!((specific_heat(&e, beta, n_sites).value - c_direct).abs() < 1e-9);

        let m2 = sample_mean_map(&signed, |x| (x / n_sites).powi(2));
        let abs = sample_mean_map(&signed, |x| (x / n_sites).abs());
        let chi_direct = beta * n_sites * (m2 - abs * abs);
        assert!((susceptibility(&signed, beta, n_sites).value - chi_direct).abs() < 1e-9);

        let mm2 = sample_mean_map(&signed, |x| x.powi(2));
        let mm4 = sample_mean_map(&signed, |x| x.powi(4));
        let u4_direct = 1.0 - mm4 / (3.0 * mm2.powi(2));
        assert!((binder_cumulant(&signed).value - u4_direct).abs() < 1e-12);
    }

    /// The exact `(E series, signed M series)` used to capture the reference
    /// numbers below, so the equivalence test runs on the same data.
    fn reference_series() -> (Vec<f64>, Vec<f64>, f64, f64) {
        let (beta, n_sites) = (0.4407, 64.0);
        let signed: Vec<f64> = (0..3000)
            .map(|i| {
                let sign = if (i / 500) % 2 == 0 { 1.0 } else { -1.0 };
                sign * n_sites * (0.6 + 0.15 * base_value(i / 3))
            })
            .collect();
        let e: Vec<f64> = signed
            .iter()
            .map(|m| -1.9 * n_sites + 0.3 * m.abs())
            .collect();
        (e, signed, beta, n_sites)
    }

    #[test]
    fn feature_jackknife_reproduces_pre_refactor_numbers() {
        // Acceptance test for the O(N) refactor: value AND stderr must not move.
        // The right-hand numbers were captured from the pre-refactor slice-based
        // implementation on this exact series. Relative tolerance 1e-9 absorbs the
        // ~1e-13 shift from centering by a single global mean instead of each
        // leave-out's own mean.
        let (e, signed, beta, n_sites) = reference_series();
        let close = |got: f64, want: f64| {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1e-300),
                "got {got:.15e}, want {want:.15e}"
            );
        };

        let c = specific_heat(&e, beta, n_sites);
        close(c.value, 8.24400144960567e-3);
        close(c.stderr, 2.377913272032474e-4);
        assert_eq!(c.n_blocks, 1000);

        let chi = susceptibility(&signed, beta, n_sites);
        close(chi.value, 2.078511824523012e-1);
        close(chi.stderr, 5.437844369399705e-3);
        assert_eq!(chi.n_blocks, 750);

        let u4 = binder_cumulant(&signed);
        close(u4.value, 6.403580348646347e-1);
        close(u4.stderr, 7.451725532088574e-4);
        assert_eq!(u4.n_blocks, 750);
    }

    #[test]
    fn fluctuation_refuses_on_too_short_series() {
        // Fewer than two blocks: the value still returns, but the error is NaN and
        // n_blocks records how few there were. A one-sample series is the clean
        // trigger — the windowing otherwise floors the block count above 1.
        let d = susceptibility(&[48.0], 0.4407, 64.0);
        assert!(d.n_blocks < 2, "n_blocks = {}", d.n_blocks);
        assert!(d.stderr.is_nan(), "stderr = {}", d.stderr);
        assert!(d.value.is_finite());
    }

    #[test]
    fn variance_stays_stable_on_a_near_constant_series() {
        // |m| ≈ 0.97 with ~1e-3 jitter: the variance is ~1e-7, small enough that a
        // naive ⟨x²⟩−⟨x⟩² would cancel catastrophically. The centered features must
        // still land on the reference variance and never go negative.
        let (beta, n_sites) = (0.4407, 1.0);
        let mags: Vec<f64> = (0..2000).map(|i| 0.97 + 1e-3 * base_value(i)).collect();

        let mean = sample_mean(&mags);
        let var_ref = sample_mean_map(&mags, |x| (x - mean).powi(2)); // center-then-square
        let chi_ref = beta * n_sites * var_ref;

        let chi = susceptibility(&mags, beta, n_sites);
        assert!(chi.value > 0.0, "value = {}", chi.value);
        assert!(
            (chi.value - chi_ref).abs() <= 1e-9 * chi_ref,
            "value {:.15e} vs reference {:.15e}",
            chi.value,
            chi_ref
        );
    }

    #[test]
    fn binder_ordered_limit_is_two_thirds() {
        // A magnetization series with no fluctuation (a nonzero constant) is the
        // ordered phase: <m^4> = <m^2>^2, so U_4 -> 1 - 1/3 = 2/3.
        let mags = vec![2.0; 100];
        let d = binder_cumulant(&mags);
        assert!((d.value - 2.0 / 3.0).abs() < 1e-9, "U_4 = {}", d.value);
    }

    #[test]
    fn creutz_reproduces_the_exact_two_dimensional_string_tension() {
        // In two dimensions <W(R,T)> = tanh(beta)^(R*T) exactly, and the Creutz
        // combination of areas collapses to 1 for every R,T — so chi must come
        // back as -log(tanh beta) at *any* loop size, not just asymptotically.
        // Constant series have no jackknife scatter, so the error must vanish too.
        // beta = atanh(1/2), written through tanh(beta) directly: a power of two
        // keeps every loop average exactly constant in floating point, so the
        // jackknife really does see zero scatter. A constant the sample mean
        // cannot reproduce to the last bit reads as a perfectly correlated series
        // instead, which would swallow the whole run into one block.
        let tanh_beta: f64 = 0.5;
        let w = |r: u32, t: u32| vec![tanh_beta.powi((r * t) as i32); 400];
        let expected = -tanh_beta.ln();

        for (r, t) in [(2, 2), (3, 2), (5, 4)] {
            let d = creutz_ratio(&w(r, t), &w(r - 1, t - 1), &w(r - 1, t), &w(r, t - 1));
            assert!(
                (d.value - expected).abs() < 1e-12,
                "chi({r},{t}) = {}, want {expected}",
                d.value
            );
            assert!(d.stderr < 1e-12, "stderr = {}", d.stderr);
            assert_eq!(d.n_blocks, 400);
        }
    }

    #[test]
    fn creutz_at_the_smallest_loop_is_minus_log_the_mean_plaquette() {
        // R = T = 1 puts three of the four loops on the trivial `1.0` anchor that
        // `wilson_rectangles` keeps in row and column zero, so the whole ratio is
        // the (1,1) loop alone and chi(1,1) = -log<W(1,1)>.
        let w11: Vec<f64> = (0..600).map(|i| 0.72 + 0.05 * base_value(i)).collect();
        let ones = vec![1.0; 600];

        let d = creutz_ratio(&w11, &ones, &ones, &ones);
        let expected = -sample_mean(&w11).ln();
        assert!(
            (d.value - expected).abs() < 1e-12,
            "chi(1,1) = {}, want {expected}",
            d.value
        );
        // A fluctuating loop must carry a real error bar, unlike the constant
        // series above.
        assert!(d.stderr > 0.0, "stderr = {}", d.stderr);
        assert!(d.is_reliable(), "n_blocks = {}", d.n_blocks);
    }

    #[test]
    fn creutz_refuses_a_non_positive_ratio() {
        // A loop average that has wandered negative — a too-noisy or
        // near-deconfined run — leaves the logarithm nothing to take. The refusal
        // is NaN rather than a clamp, and it reaches the error bar as well, since
        // every leave-out block inherits the same sign.
        let negative: Vec<f64> = (0..300).map(|i| -0.3 + 0.01 * base_value(i)).collect();
        let ones = vec![1.0; 300];

        let d = creutz_ratio(&negative, &ones, &ones, &ones);
        assert!(d.value.is_nan(), "value = {}", d.value);
        assert!(d.stderr.is_nan(), "stderr = {}", d.stderr);

        // A zero in the denominator is the same refusal: the ratio is not finite.
        let zeros = vec![0.0; 300];
        let d = creutz_ratio(&ones, &ones, &zeros, &ones);
        assert!(d.value.is_nan(), "value = {}", d.value);
    }

    #[test]
    fn creutz_refuses_on_too_short_series() {
        // The inherited too-few-blocks refusal: one sample cannot show a scatter,
        // so the ratio still comes back but the error is NaN.
        let one = [0.5];
        let unit = [1.0];
        let d = creutz_ratio(&one, &unit, &unit, &unit);
        assert!(d.n_blocks < 2, "n_blocks = {}", d.n_blocks);
        assert!(d.stderr.is_nan(), "stderr = {}", d.stderr);
        assert!(
            (d.value - -0.5f64.ln()).abs() < 1e-12,
            "value = {}",
            d.value
        );
    }

    #[test]
    fn creutz_blocks_on_the_slowest_of_its_four_series() {
        // The combination decorrelates only as fast as its slowest ingredient, and
        // the slow one here is a single corner of the 2x2 — blocking on any of the
        // other three would leave far more blocks than the error can support.
        let fast: Vec<f64> = (0..2048).map(|i| 0.8 + 0.02 * base_value(i)).collect();
        let slow: Vec<f64> = (0..2048).map(|i| 0.6 + 0.02 * base_value(i / 16)).collect();

        let expected = ((2.0 * reduce(&slow).tau_int).ceil() as usize).max(1);
        let d = creutz_ratio(&fast, &fast, &slow, &fast);
        assert_eq!(d.n_blocks, 2048 / expected);
        assert!(d.n_blocks < 2048 / 4, "n_blocks = {}", d.n_blocks);
    }

    #[test]
    fn binder_disordered_limit_is_zero() {
        // Constructed so <m^4> = 3<m^2>^2 exactly: one third of the samples are
        // nonzero, two thirds are zero, which makes <m^4>/<m^2>^2 = 1/f = 3 and
        // U_4 = 0 — the disordered anchor.
        let mut mags = vec![0.0; 300];
        for m in mags.iter_mut().take(100) {
            *m = 1.0;
        }
        let d = binder_cumulant(&mags);
        assert!(d.value.abs() < 1e-9, "U_4 = {}", d.value);
    }
}
