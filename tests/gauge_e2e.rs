//! End-to-end sampled physics for the 3D Z2 gauge theory.
//!
//! This file is the full-stack 3D validation for the gauge model. The geometry,
//! action, and observables are each covered by unit tests in isolation, and
//! `model.rs` checks the *exact* 2D area law where Z2 is solvable — but none of
//! those drives a whole 3D chain through the public runtime and asserts that the
//! physics comes out, which is the gap this closes. It runs `GaugeRunConfig` ->
//! `GaugeSampler` at a strong and a weak coupling and confirms the defining
//! behavior: confinement at strong coupling, weakening toward the deconfined
//! phase as the coupling grows past the transition near `beta_c ~ 0.76`.
//!
//! It reads two things off the sampled stream. The mean plaquette rises from near
//! its strong-coupling value toward one as `beta` grows, the coarsest signal that
//! the field is ordering. The Creutz ratio, computed from the Wilson-loop table
//! through `creutz_ratio`, is the string tension: large in the confined phase,
//! small in the deconfined one. Asserting the *contrast* between a strong and a
//! weak coupling — rather than a precise value — keeps the test robust to the
//! statistics of a short run while still exercising the whole stack, sampler and
//! observables and reduction together.
//!
//! Beyond that baseline the file covers the other two link schedules the same way
//! `ising_e2e.rs` covers its three: the tests differ only in the config's
//! `updater` field, so one run is measured through CPU Metropolis, the CPU link
//! checkerboard, and the GPU link checkerboard in turn. The checkerboards are
//! checked by *agreement* rather than against a threshold, since reordering the
//! moves is supposed to change nothing but the autocorrelation. Lattice extents
//! are kept even (`[6, 6, 6]`) because the GPU schedule requires them, and the
//! shared helper reuses that shape so the three backends stay comparable.

use plaquette::{
    Derived, Estimate, GaugeRunConfig, GaugeSampler, creutz_ratio, gauge_measure, reduce,
    wilson_rectangles,
};

/// Largest Wilson side to collect. Two is enough for the 2x2 Creutz ratio, and
/// stays within half the extent of the 6^3 box.
const MAX_SIDE: usize = 2;

/// Samples per run. A few hundred is enough for the coarse contrasts asserted
/// here without making the test slow.
const N_SAMPLES: usize = 300;

/// What one driven run reports back.
struct Measured {
    /// Chain-mean plaquette, `<sigma_plaq>`, with its autocorrelation-inflated
    /// standard error — the error is what lets one run be compared against
    /// another on statistical terms rather than against a hand-picked tolerance.
    plaquette: Estimate,
    /// Creutz ratio at the smallest loop, `chi(1,1) = -log<W(1,1)>`.
    chi_11: Derived,
    /// Creutz ratio of the 2x2 block, `chi(2,2)`.
    chi_22: Derived,
}

/// Drive a 6^3 Z2 gauge chain at inverse temperature `beta` under `updater` and
/// read the plaquette and two Creutz ratios off the stream.
fn run(beta: f64, seed: u64, updater: &str) -> Measured {
    let toml = format!(
        "shape = [6, 6, 6]\n\
         j = 1.0\n\
         beta = {beta}\n\
         updater = \"{updater}\"\n\
         thermalize = 200\n\
         sweeps_between = 3\n\
         n_samples = {N_SAMPLES}\n\
         seed = {seed}\n\
         start = \"cold\"\n"
    );
    let config = GaugeRunConfig::parse(&toml).expect("hand-written config is valid");
    let mut sampler = GaugeSampler::new(&config);

    // Geometry and model come off the sampler as owned values, read once so the
    // stream can borrow the sampler by itself.
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_plaquettes = lattice.n_plaquettes() as f64;

    let mut plaquette = Vec::with_capacity(N_SAMPLES);
    // One Wilson series per size the Creutz ratios need. The table is symmetric,
    // so W(1,2) and W(2,1) coincide, but both are collected for clarity.
    let (mut w11, mut w22, mut w12, mut w21) = (
        Vec::with_capacity(N_SAMPLES),
        Vec::with_capacity(N_SAMPLES),
        Vec::with_capacity(N_SAMPLES),
        Vec::with_capacity(N_SAMPLES),
    );

    for config in sampler.samples().take(N_SAMPLES) {
        plaquette.push(gauge_measure(&model, &lattice, &config).plaquette_sum / n_plaquettes);
        let table = wilson_rectangles(&model, &lattice, &config, MAX_SIDE).per_size;
        w11.push(table[1][1]);
        w22.push(table[2][2]);
        w12.push(table[1][2]);
        w21.push(table[2][1]);
    }

    // chi(1,1) has three trivial sides, whose loop is the constant 1.0.
    let ones = vec![1.0; N_SAMPLES];
    Measured {
        plaquette: reduce(&plaquette),
        chi_11: creutz_ratio(&w11, &ones, &ones, &ones),
        chi_22: creutz_ratio(&w22, &w11, &w12, &w21),
    }
}

/// Confinement is strong at small `beta` and weakens as the coupling grows past
/// the transition: the mean plaquette rises toward one and the string tension
/// (Creutz ratio) falls.
#[test]
fn confinement_weakens_as_the_coupling_grows() {
    let strong = run(0.2, 20260728, "metropolis");
    let weak = run(1.0, 20260729, "metropolis");

    eprintln!(
        "strong (beta=0.2): plaq={:.4} chi11={:.4} chi22={:.4}",
        strong.plaquette.mean, strong.chi_11.value, strong.chi_22.value
    );
    eprintln!(
        "weak   (beta=1.0): plaq={:.4} chi11={:.4} chi22={:.4}",
        weak.plaquette.mean, weak.chi_11.value, weak.chi_22.value
    );

    // Leading strong-coupling result, pinned quantitatively: <plaquette> -> tanh(beta)
    // as beta -> 0, and at beta = 0.2 the correction is well under a percent.
    let tanh_strong = 0.2_f64.tanh();
    assert!(
        (strong.plaquette.mean - tanh_strong).abs() < 0.02,
        "strong plaquette {} vs tanh(0.2) {}",
        strong.plaquette.mean,
        tanh_strong
    );
    // Weak coupling sits deep in the ordered phase: the plaquette is near one.
    assert!(
        weak.plaquette.mean > 0.9,
        "weak plaquette {}",
        weak.plaquette.mean
    );

    // The string tension read off the smallest loop, chi(1,1) = -log<plaquette>,
    // is large in the confined phase and vanishes as the field orders.
    assert!(
        strong.chi_11.value > 1.0,
        "strong chi(1,1) {}",
        strong.chi_11.value
    );
    assert!(
        weak.chi_11.value < 0.1,
        "weak chi(1,1) {}",
        weak.chi_11.value
    );

    // The genuine 2x2 Creutz ratio (four real Wilson series) resolves to a small
    // value in the deconfined phase. At strong coupling the 2x2 loop average sits
    // below the noise of a short run, so creutz_ratio honestly returns NaN there
    // rather than a fabricated tension — the reason only the weak side is asserted.
    assert!(
        weak.chi_22.value.is_finite() && weak.chi_22.value < 0.1,
        "weak chi(2,2) {}",
        weak.chi_22.value
    );
}

/// The coupling the agreement tests run at: inside the confined phase but clear
/// of the transition near `beta_c ~ 0.76`, so the plaquette is pinned to neither
/// 0 nor 1 and the chains still decorrelate over a short run. Near `beta_c` the
/// autocorrelation is long enough that `reduce` under-resolves `tau_int` at this
/// sample count, which would make the error bars the comparison rests on
/// optimistic.
const AGREEMENT_BETA: f64 = 0.5;

/// One line of a `Measured`, for the record a failing comparison is read from.
fn report(label: &str, m: &Measured) {
    eprintln!(
        "{label:<24} (beta={AGREEMENT_BETA}): plaq={:.4}({:.4}) tau={:.2} chi11={:.4}({:.4})",
        m.plaquette.mean, m.plaquette.stderr, m.plaquette.tau_int, m.chi_11.value, m.chi_11.stderr
    );
}

/// Assert that a checkerboard run measures the same physics as the random-order
/// Metropolis run it reorders.
///
/// Agreement is asserted statistically rather than against a hand-picked
/// tolerance: the two runs are independent, so the difference of their means has
/// standard error `sqrt(se_a^2 + se_b^2)`, and a correct pair of schedules
/// exceeds four of those about as often as a fair coin lands heads sixteen times
/// running. The errors come from `reduce`, which inflates by the integrated
/// autocorrelation time — which matters here because reordering the moves is
/// precisely what changes `tau_int`, and a naive `stderr` would understate the
/// slower chain. The comparison cannot be bit-for-bit against any backend:
/// Metropolis draws a link index per step, the CPU checkerboard draws none, and
/// the GPU keys a counter on `(seed, link, sweep)`, so all three run on
/// different streams.
fn assert_matches(reference: &Measured, other: &Measured, label: &str) {
    let plaquette_sigma =
        (reference.plaquette.stderr.powi(2) + other.plaquette.stderr.powi(2)).sqrt();
    let plaquette_gap = (reference.plaquette.mean - other.plaquette.mean).abs();
    assert!(
        plaquette_gap < 4.0 * plaquette_sigma,
        "{label}: mean plaquette differs by {plaquette_gap} = {:.1} sigma: \
         metropolis {}, {label} {}",
        plaquette_gap / plaquette_sigma,
        reference.plaquette.mean,
        other.plaquette.mean
    );

    // The same test on a derived quantity, whose error is a jackknife over
    // blocks rather than a mean's standard error. chi(1,1) is a function of the
    // plaquette alone, so this is not independent evidence — it checks that the
    // agreement survives the reduction machinery, not just the raw average.
    let chi_sigma = (reference.chi_11.stderr.powi(2) + other.chi_11.stderr.powi(2)).sqrt();
    let chi_gap = (reference.chi_11.value - other.chi_11.value).abs();
    assert!(
        chi_sigma.is_finite(),
        "{label}: jackknife error should resolve at beta = {AGREEMENT_BETA}"
    );
    assert!(
        chi_gap < 4.0 * chi_sigma,
        "{label}: chi(1,1) differs by {chi_gap} = {:.1} sigma: metropolis {}, {label} {}",
        chi_gap / chi_sigma,
        reference.chi_11.value,
        other.chi_11.value
    );
}

/// Whether a GPU test can run here, mirroring the crate's internal guard.
///
/// An integration test compiles against the published API, so it cannot reach
/// the crate-internal helper and repeats the rule instead: skip when no adapter
/// is present, but fail when `PLAQUETTE_REQUIRE_GPU` says one is expected — CI
/// sets it so a driver that fails to load is a failure, not a silent pass.
fn gpu_available() -> bool {
    if plaquette::Gpu::new().is_some() {
        return true;
    }
    assert!(
        std::env::var_os("PLAQUETTE_REQUIRE_GPU").is_none(),
        "PLAQUETTE_REQUIRE_GPU is set but no GPU adapter is available"
    );
    eprintln!("no GPU adapter available; skipping GPU end-to-end test");
    false
}

/// CPU link checkerboard: reordering the moves into color passes measures the
/// same physics, through the whole public stack rather than at the sweep level.
#[test]
fn cpu_link_checkerboard_matches_metropolis() {
    let metropolis = run(AGREEMENT_BETA, 20260801, "metropolis");
    let checkerboard = run(AGREEMENT_BETA, 20260802, "link_checkerboard");
    report("metropolis", &metropolis);
    report("link_checkerboard", &checkerboard);

    assert_matches(&metropolis, &checkerboard, "link_checkerboard");
}

/// GPU link checkerboard: the device backend measures it too. Skips rather than
/// fails when no GPU adapter is present, so the suite stays green on a headless
/// runner — the same guard the inline GPU tests use.
///
/// This is the arm that makes the shader's correctness a claim about the whole
/// stack: the config names the backend, `GaugeSampler` builds the device chain,
/// and the same statistical comparison the CPU arm uses decides it.
#[test]
fn gpu_link_checkerboard_matches_metropolis() {
    if !gpu_available() {
        return;
    }
    let metropolis = run(AGREEMENT_BETA, 20260803, "metropolis");
    let checkerboard = run(AGREEMENT_BETA, 20260804, "gpu_link_checkerboard");
    report("metropolis", &metropolis);
    report("gpu_link_checkerboard", &checkerboard);

    assert_matches(&metropolis, &checkerboard, "gpu_link_checkerboard");
}
