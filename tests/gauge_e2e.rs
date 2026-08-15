//! End-to-end sampled physics for the Z2 gauge theory, across dimensions and
//! across all three update backends.
//!
//! This file is the full-stack validation for the gauge model. The geometry, the
//! action, and the observables are each covered by unit tests in isolation, but
//! none of those drives a whole chain through the public runtime and asserts that
//! the physics comes out, which is the gap this closes. It runs `GaugeRunConfig`
//! -> `GaugeSampler` and reads two things off the sampled stream: the mean
//! plaquette, the coarsest signal that the field is ordering, and the Creutz
//! ratio computed from the Wilson-loop table, which is the string tension — large
//! in the confined phase, small in the deconfined one.
//!
//! The dimension comes from the file's `shape`, so the same path serves every
//! dimension the theory is defined in, and what can be asserted differs by
//! dimension rather than by code. Two dimensions is *solvable*: fixing the gauge
//! leaves one free variable per plaquette, so the plaquettes are independent,
//! each averages to `tanh(beta)`, and a rectangle — being the product of the
//! plaquettes it encloses — averages to `tanh(beta)` raised to its area. That
//! makes it the one place a sampled chain can be checked against a closed form
//! rather than against a contrast, and a sign or staple error shows up
//! immediately instead of after a long run near a transition.
//!
//! Three and four dimensions have no closed form, so they get the contrast: the
//! plaquette rises toward one and the string tension falls as the coupling grows
//! past `beta_c ~ 0.7614` in three dimensions and the self-dual
//! `beta_c = ln(1 + sqrt(2)) / 2 ~ 0.4407` in four. Four needs more care than
//! three, because its transition is *first order*: near `beta_c` a short run sits
//! in whichever branch it started in, so both couplings are kept well clear of it
//! and only the contrast is asserted.
//!
//! Beyond that the file covers the other two link schedules the same way
//! `ising_e2e.rs` covers its three, and repeats the comparison at two, four, and
//! six dimensions — none of which the GPU kernel was originally written for. The
//! checkerboards are checked by *agreement* rather than against a threshold,
//! since reordering the moves is supposed to change nothing but the
//! autocorrelation. Six is the ceiling here, chosen as comfortably inside what
//! anyone would plausibly run rather than as anything the library states; the
//! exact `energy` against `energy_delta` check in `model.rs` goes to ten.
//! Lattice extents are kept even because the GPU schedule requires them.

use plaquette::models::gauge::{GaugeRunConfig, GaugeSampler, gauge_measure, wilson_rectangles};
use plaquette::{Derived, Estimate, creutz_ratio, reduce};

/// Largest Wilson side to collect. Two is enough for the 2x2 Creutz ratio, and
/// stays within half the extent of every box that asks for a table.
const MAX_SIDE: usize = 2;

/// Samples per run. A few hundred is enough for the coarse contrasts asserted
/// here without making the test slow.
const N_SAMPLES: usize = 300;

/// The run parameters every test below holds fixed, as TOML.
///
/// One builder rather than a literal per helper, so a decorrelation stride or a
/// start that drifted between them could not silently make two runs
/// incomparable.
fn run_toml<const D: usize>(shape: [usize; D], beta: f64, seed: u64, updater: &str) -> String {
    format!(
        "shape = {shape:?}\n\
         j = 1.0\n\
         beta = {beta}\n\
         updater = \"{updater}\"\n\
         thermalize = 200\n\
         sweeps_between = 3\n\
         n_samples = {N_SAMPLES}\n\
         seed = {seed}\n\
         start = \"cold\"\n"
    )
}

/// What every driven run reports back.
struct Measured {
    /// Chain-mean plaquette, `<sigma_plaq>`, with its autocorrelation-inflated
    /// standard error — the error is what lets one run be compared against
    /// another on statistical terms rather than against a hand-picked tolerance.
    plaquette: Estimate,
    /// Creutz ratio at the smallest loop, `chi(1,1) = -log<W(1,1)>`.
    ///
    /// Derived from the plaquette series rather than from a Wilson table, since
    /// `W(1,1)` *is* the mean plaquette — a 1x1 rectangle is a plaquette. It
    /// still exercises the blocked jackknife the comparisons rest on, and it
    /// costs nothing beyond the average already in hand.
    chi_11: Derived,
}

/// The Wilson table, for the two tests that read more of it than `W(1,1)`.
struct Loops {
    w11: Estimate,
    w12: Estimate,
    w22: Estimate,
    /// Creutz ratio of the 2x2 block, the only one needing four real series.
    chi_22: Derived,
}

/// Drive a `D`-dimensional Z2 gauge chain of `shape` and read the plaquette off
/// the stream.
///
/// Deliberately no Wilson table. Collecting one costs the rectangle count times
/// the volume times the perimeter per sample, and the count grows as `D(D - 1)`
/// — at four dimensions that is three quarters of the run. Most tests here need
/// only the plaquette and the `chi(1,1)` that follows from it, so
/// [`run_with_loops`] is the exception rather than the shared path.
fn run<const D: usize>(shape: [usize; D], beta: f64, seed: u64, updater: &str) -> Measured {
    let toml = run_toml(shape, beta, seed, updater);
    let config = GaugeRunConfig::parse(&toml).expect("hand-written config is valid");
    let mut sampler = GaugeSampler::<D>::new(&config);

    // Geometry and model come off the sampler as owned values, read once so the
    // stream can borrow the sampler by itself.
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_plaquettes = lattice.n_plaquettes() as f64;

    let plaquette: Vec<f64> = sampler
        .samples()
        .take(N_SAMPLES)
        .map(|c| gauge_measure(&model, &lattice, &c).plaquette_sum / n_plaquettes)
        .collect();

    // chi(1,1) has three trivial sides, whose loop is the constant 1.0.
    let ones = vec![1.0; N_SAMPLES];
    Measured {
        chi_11: creutz_ratio(&plaquette, &ones, &ones, &ones),
        plaquette: reduce(&plaquette),
    }
}

/// [`run`], plus the Wilson table up to [`MAX_SIDE`].
///
/// Only the exact two-dimensional area law and the three-dimensional `chi(2,2)`
/// contrast read it; everything else would pay for a table it never touches.
/// The box must have room for the rectangles, so every extent is at least twice
/// `MAX_SIDE`.
fn run_with_loops<const D: usize>(
    shape: [usize; D],
    beta: f64,
    seed: u64,
    updater: &str,
) -> (Measured, Loops) {
    let toml = run_toml(shape, beta, seed, updater);
    let config = GaugeRunConfig::parse(&toml).expect("hand-written config is valid");
    let mut sampler = GaugeSampler::<D>::new(&config);
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

    let ones = vec![1.0; N_SAMPLES];
    (
        Measured {
            chi_11: creutz_ratio(&plaquette, &ones, &ones, &ones),
            plaquette: reduce(&plaquette),
        },
        Loops {
            w11: reduce(&w11),
            w12: reduce(&w12),
            w22: reduce(&w22),
            chi_22: creutz_ratio(&w22, &w11, &w12, &w21),
        },
    )
}

/// Two dimensions, against the exact area law.
///
/// The one gauge assertion in the suite that compares a sampled chain to a closed
/// form: every rectangle must average to `tanh(beta)` raised to its enclosed
/// area. It is checked through the public runtime rather than at the action level
/// — where `model.rs` already checks it on a bare `Chain` — because that is what
/// makes it a statement about the config, the sampler, and the observables
/// together.
///
/// It is also the cheapest place a staple or sign error surfaces. The three sizes
/// span areas one, two, and four, so a wrong exponent shows up as a mismatch that
/// grows with the loop rather than as a uniform offset a tolerance could absorb.
#[test]
fn two_dimensional_wilson_loops_follow_the_exact_area_law() {
    let beta = 0.5;
    let (_, loops) = run_with_loops([8, 8], beta, 20260805, "metropolis");
    let t = beta.tanh();

    for (label, estimate, area) in [
        ("W(1,1)", &loops.w11, 1),
        ("W(1,2)", &loops.w12, 2),
        ("W(2,2)", &loops.w22, 4),
    ] {
        let exact = t.powi(area);
        assert!(
            (estimate.mean - exact).abs() < 0.02,
            "{label} = {:.4}, exact tanh({beta})^{area} = {exact:.4}",
            estimate.mean
        );
    }
}

/// Confinement is strong at small `beta` and weakens as the coupling grows past
/// the transition: the mean plaquette rises toward one and the string tension
/// (Creutz ratio) falls.
#[test]
fn confinement_weakens_as_the_coupling_grows() {
    let (strong, _) = run_with_loops([6, 6, 6], 0.2, 20260728, "metropolis");
    let (weak, weak_loops) = run_with_loops([6, 6, 6], 1.0, 20260729, "metropolis");

    eprintln!(
        "strong (beta=0.2): plaq={:.4} chi11={:.4}",
        strong.plaquette.mean, strong.chi_11.value
    );
    eprintln!(
        "weak   (beta=1.0): plaq={:.4} chi11={:.4} chi22={:.4}",
        weak.plaquette.mean, weak.chi_11.value, weak_loops.chi_22.value
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
        weak_loops.chi_22.value.is_finite() && weak_loops.chi_22.value < 0.1,
        "weak chi(2,2) {}",
        weak_loops.chi_22.value
    );
}

/// Four dimensions: the same contrast, across the self-dual `beta_c ~ 0.4407`.
///
/// Both couplings are kept well clear of `beta_c` because the four-dimensional
/// transition is first order. Near it the two phases coexist, a short run stays
/// in whichever branch its start put it in, and the measured plaquette would
/// depend on `start` rather than on `beta` — a genuinely flaky assertion that
/// would say nothing about the code. Far from it there is one phase and the
/// contrast is unambiguous.
///
/// Four is also where a link sits in `2(D - 1) = 6` plaquettes and its staple row
/// is at its widest here, so it is the strongest check that nothing in the staple
/// table or its stride assumed three dimensions.
#[test]
fn four_dimensional_confinement_weakens_as_the_coupling_grows() {
    let strong = run([4, 4, 4, 4], 0.2, 20260806, "metropolis");
    let weak = run([4, 4, 4, 4], 0.9, 20260807, "metropolis");

    eprintln!(
        "4D strong (beta=0.2): plaq={:.4} chi11={:.4}",
        strong.plaquette.mean, strong.chi_11.value
    );
    eprintln!(
        "4D weak   (beta=0.9): plaq={:.4} chi11={:.4}",
        weak.plaquette.mean, weak.chi_11.value
    );

    assert!(
        strong.plaquette.mean < 0.4,
        "4D strong coupling should be disordered: plaquette {}",
        strong.plaquette.mean
    );
    assert!(
        weak.plaquette.mean > 0.9,
        "4D weak coupling should be ordered: plaquette {}",
        weak.plaquette.mean
    );
    assert!(
        strong.chi_11.value > weak.chi_11.value,
        "4D string tension should fall with the coupling: {} then {}",
        strong.chi_11.value,
        weak.chi_11.value
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
    let metropolis = run([6, 6, 6], AGREEMENT_BETA, 20260801, "metropolis");
    let checkerboard = run([6, 6, 6], AGREEMENT_BETA, 20260802, "link_checkerboard");
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
    let metropolis = run([6, 6, 6], AGREEMENT_BETA, 20260803, "metropolis");
    let checkerboard = run([6, 6, 6], AGREEMENT_BETA, 20260804, "gpu_link_checkerboard");
    report("metropolis", &metropolis);
    report("gpu_link_checkerboard", &checkerboard);

    assert_matches(&metropolis, &checkerboard, "gpu_link_checkerboard");
}

/// CPU heat bath: drawing the link from its conditional rather than proposing a
/// flip measures the same physics, through the whole public stack.
///
/// The comparison is the statistical one `assert_matches` makes, and it has to
/// be: the heat bath consumes one uniform per link where Metropolis skips the
/// draw on a downhill move, so the two run on different streams however they are
/// seeded. What it is sensitive to is the exponent. A sign error or a stray
/// factor of two in the conditional would move the mean plaquette far outside
/// four combined standard errors while leaving the chain looking perfectly
/// healthy.
#[test]
fn cpu_heat_bath_matches_metropolis() {
    let metropolis = run([6, 6, 6], AGREEMENT_BETA, 20260805, "metropolis");
    let heat_bath = run([6, 6, 6], AGREEMENT_BETA, 20260806, "heat_bath");
    report("metropolis", &metropolis);
    report("heat_bath", &heat_bath);

    assert_matches(&metropolis, &heat_bath, "heat_bath");
}

/// GPU link heat bath: the device kernel measures it too, on the same coloring
/// the Metropolis kernel uses.
///
/// This is the arm that says the heat bath shader is right about the staple sum
/// as well as about the conditional, and that the coloring carries over: a heat
/// bath thread conditions on a frozen neighborhood, so a color that was not
/// actually independent would show up here as a shifted mean plaquette rather
/// than as a crash.
#[test]
fn gpu_link_checkerboard_heat_bath_matches_metropolis() {
    if !gpu_available() {
        return;
    }
    // Compared against its sequential reference — the same coloring and the same
    // kernel, one link at a time — rather than against Metropolis, so a coloring
    // and a kernel that were each wrong in compensating ways could not pass.
    let reference = run(
        [6, 6, 6],
        AGREEMENT_BETA,
        20260807,
        "link_checkerboard_heat_bath",
    );
    let gpu = run(
        [6, 6, 6],
        AGREEMENT_BETA,
        20260808,
        "gpu_link_checkerboard_heat_bath",
    );
    report("link_checkerboard_heat_bath", &reference);
    report("gpu_link_checkerboard_heat_bath", &gpu);

    assert_matches(&reference, &gpu, "gpu_link_checkerboard_heat_bath");
}

/// The backends agree at a dimension the GPU kernel was not written for.
///
/// This is the arm that would catch a mismatch between the dimension the host
/// uploads and the one the kernel assumes. The kernel takes `D` as a WGSL
/// `override` and derives its staple stride and group count from it, and it
/// launches `2 * D` color passes per sweep — none of which the three-dimensional
/// tests above would notice going wrong, since three is what the shader used to
/// be written for. This and its four-dimensional twin bracket it from both sides.
#[test]
fn the_backends_agree_in_two_dimensions() {
    let metropolis = run([8, 8], AGREEMENT_BETA, 20260810, "metropolis");
    let checkerboard = run([8, 8], AGREEMENT_BETA, 20260811, "link_checkerboard");
    report("2D metropolis", &metropolis);
    report("2D link_checkerboard", &checkerboard);
    assert_matches(&metropolis, &checkerboard, "link_checkerboard");

    if !gpu_available() {
        return;
    }
    let gpu = run([8, 8], AGREEMENT_BETA, 20260812, "gpu_link_checkerboard");
    report("2D gpu_link_checkerboard", &gpu);
    assert_matches(&metropolis, &gpu, "gpu_link_checkerboard");
}

/// The other side of the same bracket — see
/// [`the_backends_agree_in_two_dimensions`]. Four dimensions has the widest
/// staple row and the most color passes per sweep, so it is where a stride the
/// host and the kernel disagree about would read past the end of a link's row.
#[test]
fn the_backends_agree_in_four_dimensions() {
    let shape = [4, 4, 4, 4];
    let metropolis = run(shape, AGREEMENT_BETA, 20260813, "metropolis");
    let checkerboard = run(shape, AGREEMENT_BETA, 20260814, "link_checkerboard");
    report("4D metropolis", &metropolis);
    report("4D link_checkerboard", &checkerboard);
    assert_matches(&metropolis, &checkerboard, "link_checkerboard");

    if !gpu_available() {
        return;
    }
    let gpu = run(shape, AGREEMENT_BETA, 20260815, "gpu_link_checkerboard");
    report("4D gpu_link_checkerboard", &gpu);
    assert_matches(&metropolis, &gpu, "gpu_link_checkerboard");
}

/// The three backends still agree at six dimensions, and the plaquette still
/// lands on its strong-coupling value there.
///
/// The ceiling the suite checks. Six dimensions is where a link sits in
/// `2(D - 1) = 10` plaquettes and each site anchors `C(6, 2) = 15` of them, so
/// the staple row is more than twice the width the shader was originally written
/// for and the sweep is twelve color passes rather than six. The kernel derives
/// both from `D`, which reaches it as a WGSL `override`, so a disagreement
/// between what the host uploads and what the kernel assumes surfaces here.
///
/// The box is deliberately lopsided and mostly minimal, `[2, 2, 2, 2, 4, 4]`:
/// equal extents would hide a transposed stride, and the volume has to stay
/// small because a six-dimensional lattice grows as `L^D`.
///
/// `beta = 0.2` is well inside the strong-coupling phase, where the leading
/// result `<plaquette> -> tanh(beta)` holds in any dimension. That gives the test
/// an absolute anchor as well as the backend comparison, so all three agreeing on
/// a wrong number would still fail.
#[test]
fn the_backends_agree_in_six_dimensions() {
    let shape = [2, 2, 2, 2, 4, 4];
    let beta = 0.2;

    let metropolis = run(shape, beta, 20260820, "metropolis");
    let checkerboard = run(shape, beta, 20260821, "link_checkerboard");
    eprintln!(
        "6D: metropolis {:.4}({:.4}), checkerboard {:.4}({:.4}), tanh({beta}) = {:.4}",
        metropolis.plaquette.mean,
        metropolis.plaquette.stderr,
        checkerboard.plaquette.mean,
        checkerboard.plaquette.stderr,
        beta.tanh()
    );

    assert!(
        (metropolis.plaquette.mean - beta.tanh()).abs() < 0.02,
        "6D strong-coupling plaquette {} vs tanh({beta}) {}",
        metropolis.plaquette.mean,
        beta.tanh()
    );
    let sigma =
        (metropolis.plaquette.stderr.powi(2) + checkerboard.plaquette.stderr.powi(2)).sqrt();
    assert!(
        (metropolis.plaquette.mean - checkerboard.plaquette.mean).abs() < 4.0 * sigma,
        "CPU schedules disagree in 6D: {} vs {}",
        metropolis.plaquette.mean,
        checkerboard.plaquette.mean
    );

    if !gpu_available() {
        return;
    }
    let gpu = run(shape, beta, 20260822, "gpu_link_checkerboard");
    eprintln!(
        "6D: gpu {:.4}({:.4})",
        gpu.plaquette.mean, gpu.plaquette.stderr
    );
    let sigma = (metropolis.plaquette.stderr.powi(2) + gpu.plaquette.stderr.powi(2)).sqrt();
    assert!(
        (metropolis.plaquette.mean - gpu.plaquette.mean).abs() < 4.0 * sigma,
        "GPU disagrees in 6D: {} vs {}",
        metropolis.plaquette.mean,
        gpu.plaquette.mean
    );
}

/// One dimension is refused, and the message says why.
///
/// The floor is a statement about the action rather than about the loader: below
/// two dimensions the lattice has no direction pair, so it has no plaquettes and
/// the energy would come out identically zero for every configuration. That is a
/// run producing nothing rather than one failing, which is exactly the kind of
/// thing a schema has to catch.
#[test]
fn a_one_dimensional_gauge_run_is_refused() {
    let toml = "shape = [8]\n\
                j = 1.0\n\
                beta = 0.5\n\
                thermalize = 1\n\
                sweeps_between = 1\n\
                n_samples = 1\n\
                seed = 1\n";
    let err = GaugeRunConfig::parse(toml).expect_err("one dimension has no plaquettes");
    let message = err.to_string();
    assert!(message.contains("plaquettes"), "{message}");
    assert!(message.contains("at least 2 dimensions"), "{message}");
}
