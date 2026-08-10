//! End-to-end sampled-physics tests for the Ising model, driving the full public
//! runtime stack across every update backend and across dimensions.
//!
//! These live in `tests/` rather than beside the code because they are meant to
//! exercise `plaquette` the way a consumer does: through its published API alone,
//! from a TOML string down to a measured observable. Where the crate's inline
//! tests check each piece in isolation, these check that the pieces compose into
//! correct physics — parse a run, build a thermalized [`IsingSampler`], stream
//! configurations off it, and measure each one with [`plaquette::measure`].
//!
//! Three of the tests fix the dimension at two and vary the backend instead. At
//! `beta = 1.0` the two-dimensional model sits well inside the ordered phase
//! (`beta_c ≈ 0.4407`), so from any start the chain equilibrates to a
//! near-aligned configuration. Onsager's spontaneous magnetization there is
//! `|m| ≈ 0.999`, which leaves a conservative `> 0.5` threshold on the mean of
//! `|M| / N` both robust and seed-deterministic. Those three differ only in the
//! config's `updater` field, and the extents are kept even (`[16, 16]`) because
//! the GPU checkerboard requires them.
//!
//! The rest vary the dimension instead, taking it from the shape they are
//! handed the way a driver takes it from its own `const D`. One dimension is
//! checked against the closed form rather than against a threshold: the
//! one-dimensional model's two-point function is known exactly at finite `N`,
//! not merely in the thermodynamic limit, so a sampled chain can be compared to
//! the answer at the size actually simulated. Three and four dimensions have no
//! closed form, so they get the ordering contrast the two-dimensional tests use,
//! anchored on their own critical couplings, and three and six repeat the
//! backend comparison — six being the ceiling here, chosen as comfortably inside
//! what anyone would plausibly run rather than as anything the library states.
//! The exact `energy` against `energy_delta` check in `model.rs` goes to ten.

use plaquette::models::ising::{IsingRunConfig, IsingSampler, correlator, measure};

/// The number of samples each ordering test measures after warmup. A few hundred
/// is plenty for the mean of `|m|` to settle deep in the ordered phase, and keeps
/// each test fast.
const N_SAMPLES: usize = 200;

/// Build a run's TOML from the fields the tests actually vary, holding the rest
/// fixed so runs stay comparable and deterministic.
///
/// The hot start is deliberate: ordering from a disordered start is the stronger
/// check, since it shows the chain *reaches* the ordered phase rather than
/// merely staying in it.
fn run_toml<const D: usize>(
    shape: [usize; D],
    beta: f64,
    updater: &str,
    n_samples: usize,
) -> String {
    format!(
        "shape = {shape:?}\n\
         j = 1.0\n\
         beta = {beta}\n\
         updater = \"{updater}\"\n\
         thermalize = 500\n\
         sweeps_between = 2\n\
         n_samples = {n_samples}\n\
         seed = 20260728\n\
         start = \"hot\"\n"
    )
}

/// Drive the full public stack at dimension `D` and return the mean of `|M| / N`
/// over the sampled configs.
///
/// The absolute value is taken per config because the ordered phase settles into
/// either the up or the down well arbitrarily, so the signed mean would cancel
/// while `|m|` reports the ordering the assertions are after.
fn mean_abs_magnetization<const D: usize>(shape: [usize; D], beta: f64, updater: &str) -> f64 {
    let text = run_toml(shape, beta, updater, N_SAMPLES);
    let config = IsingRunConfig::parse(&text).expect("run config should parse");
    let mut sampler = IsingSampler::<D>::new(&config);

    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_sites = lattice.n_sites() as f64;

    let sum: f64 = sampler
        .samples()
        .take(N_SAMPLES)
        .map(|c| (measure(&model, &lattice, &c).magnetization / n_sites).abs())
        .sum();

    sum / N_SAMPLES as f64
}

/// CPU Metropolis: a single-site Metropolis run orders at low temperature.
#[test]
fn cpu_metropolis_orders_at_low_temperature() {
    let mean_abs_m = mean_abs_magnetization([16, 16], 1.0, "metropolis");
    assert!(
        mean_abs_m > 0.5,
        "low-T Metropolis run should order: mean |m| = {mean_abs_m}"
    );
}

/// CPU checkerboard: the two-color sweep produces the same ordering. Same run,
/// same threshold — only the update rule differs.
#[test]
fn cpu_checkerboard_orders_at_low_temperature() {
    let mean_abs_m = mean_abs_magnetization([16, 16], 1.0, "site_checkerboard");
    assert!(
        mean_abs_m > 0.5,
        "low-T checkerboard run should order: mean |m| = {mean_abs_m}"
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

/// GPU checkerboard: the device backend orders too. Skips rather than fails when
/// no GPU adapter is present, so the suite stays green on a headless runner —
/// the same guard the inline GPU tests use.
#[test]
fn gpu_checkerboard_orders_at_low_temperature() {
    if !gpu_available() {
        return;
    }
    let mean_abs_m = mean_abs_magnetization([16, 16], 1.0, "gpu_site_checkerboard");
    assert!(
        mean_abs_m > 0.5,
        "low-T GPU checkerboard run should order: mean |m| = {mean_abs_m}"
    );
}

/// One dimension, against the exact finite-`N` correlator.
///
/// This is the only test in the suite comparing a sampled chain to a closed
/// form rather than to a threshold or to another run, and one dimension is where
/// that is possible: the transfer matrix gives
/// `<s_0 s_r> = (t^r + t^(N-r)) / (1 + t^N)` with `t = tanh(beta j)`, exactly,
/// at the `N` actually simulated. So there is no thermodynamic-limit fudge in
/// the comparison and no tolerance chosen to cover finite-size drift — what
/// remains is sampling error alone.
///
/// It also exercises the parts of the stack a `D = 2` run cannot: `correlator`
/// returns one row per axis, so at `D = 1` there is a single row, and a chain
/// with `2 * D = 2` neighbors per site is the narrowest neighbor table the
/// updaters ever walk.
#[test]
fn one_dimensional_correlator_matches_the_exact_solution() {
    const N: usize = 16;
    const BETA: f64 = 0.4;
    const SAMPLES: usize = 4000;

    let text = run_toml([N], BETA, "metropolis", SAMPLES);
    let config = IsingRunConfig::parse(&text).expect("run config should parse");
    let mut sampler = IsingSampler::<1>::new(&config);
    let (lattice, model) = (sampler.lattice(), sampler.model());

    // Accumulate the per-config correlator along the one axis there is.
    let mut sums = [0.0f64; N / 2 + 1];
    for c in sampler.samples().take(SAMPLES) {
        for (slot, value) in sums
            .iter_mut()
            .zip(&correlator(&model, &lattice, &c).per_axis[0])
        {
            *slot += value;
        }
    }

    let t = (BETA * config.j).tanh();
    for (r, &sum) in sums.iter().enumerate() {
        let measured = sum / SAMPLES as f64;
        let exact = (t.powi(r as i32) + t.powi((N - r) as i32)) / (1.0 + t.powi(N as i32));
        assert!(
            (measured - exact).abs() < 0.02,
            "C_{r}: measured {measured:.4} vs exact {exact:.4}"
        );
    }
}

/// Three dimensions: the same ordering contrast the two-dimensional tests make,
/// across the known `beta_c ≈ 0.2216546`.
///
/// A contrast rather than a value, because three dimensions has no closed form —
/// what is asserted is that the chain is ordered well below the transition and
/// disordered well above it, which no dimension-confused geometry would produce
/// by accident. Both couplings sit far enough from `beta_c` that critical
/// slowing down does not make a short run flaky.
#[test]
fn three_dimensional_runs_order_below_the_transition_and_not_above() {
    let ordered = mean_abs_magnetization([8, 8, 8], 0.35, "metropolis");
    let disordered = mean_abs_magnetization([8, 8, 8], 0.12, "metropolis");
    assert!(
        ordered > 0.7,
        "beta = 0.35 is well below beta_c = 0.2217, so the run should order: {ordered}"
    );
    assert!(
        disordered < 0.2,
        "beta = 0.12 is well above beta_c, so the run should not order: {disordered}"
    );
}

/// Four dimensions, the same way, across `beta_c ≈ 0.14969`.
///
/// Four is where a site has eight neighbors and the neighbor table is at its
/// widest here, so it is the strongest check that nothing in the geometry or the
/// coloring assumed a smaller row.
#[test]
fn four_dimensional_runs_order_below_the_transition_and_not_above() {
    let ordered = mean_abs_magnetization([6, 6, 6, 6], 0.25, "metropolis");
    let disordered = mean_abs_magnetization([6, 6, 6, 6], 0.08, "metropolis");
    assert!(
        ordered > 0.7,
        "beta = 0.25 is well below beta_c = 0.1497, so the run should order: {ordered}"
    );
    assert!(
        disordered < 0.2,
        "beta = 0.08 is well above beta_c, so the run should not order: {disordered}"
    );
}

/// The three backends agree at a dimension none of them was written for.
///
/// The point is the GPU kernel. Its dimension arrives as a WGSL `override` and
/// its neighbor row width derives from that, so a mismatch between what the host
/// uploads and what the kernel assumes would show up here and nowhere else —
/// the two-dimensional tests above would keep passing, since two is what the
/// shader used to be written for. Agreement is asserted between the backends
/// rather than against a number, since reordering the moves is supposed to
/// change nothing but the autocorrelation.
#[test]
fn the_backends_agree_in_three_dimensions() {
    let shape = [8, 8, 8];
    let beta = 0.28; // ordered, but not so deep that |m| saturates and hides a bug

    let metropolis = mean_abs_magnetization(shape, beta, "metropolis");
    let checkerboard = mean_abs_magnetization(shape, beta, "site_checkerboard");
    assert!(
        (metropolis - checkerboard).abs() < 0.05,
        "CPU schedules disagree in 3D: metropolis {metropolis:.4} vs checkerboard {checkerboard:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization(shape, beta, "gpu_site_checkerboard");
    assert!(
        (metropolis - gpu).abs() < 0.05,
        "GPU disagrees in 3D: metropolis {metropolis:.4} vs gpu {gpu:.4}"
    );
}

/// The three backends still agree at six dimensions.
///
/// The ceiling the suite checks. Nothing in the library states an upper bound —
/// `IsingSampler::<9>` would compile — so ten is a guess at what anyone would
/// plausibly run and six is comfortably inside it. What makes the dimension
/// worth reaching is that a site now has twelve neighbors, so the neighbor row
/// the kernel walks is three times the width the shader was originally written
/// for, and the `D` it walks it with arrives as a WGSL `override`.
///
/// The box is deliberately lopsided and mostly minimal, `[2, 2, 2, 2, 4, 4]`.
/// Equal extents would hide a transposed stride, and the volume has to stay
/// small because a six-dimensional lattice grows as `L^6`; at `beta = 0.12` the
/// model sits above its critical coupling of roughly `0.09` and orders to about
/// `|m| = 0.74`, far enough from both 0 and 1 that a backend computing something
/// slightly wrong would show.
#[test]
fn the_backends_agree_in_six_dimensions() {
    let shape = [2, 2, 2, 2, 4, 4];
    let beta = 0.12;

    let metropolis = mean_abs_magnetization(shape, beta, "metropolis");
    let checkerboard = mean_abs_magnetization(shape, beta, "site_checkerboard");
    assert!(
        metropolis > 0.4 && metropolis < 0.95,
        "6D run should order without saturating: {metropolis}"
    );
    assert!(
        (metropolis - checkerboard).abs() < 0.1,
        "CPU schedules disagree in 6D: metropolis {metropolis:.4} vs checkerboard {checkerboard:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization(shape, beta, "gpu_site_checkerboard");
    assert!(
        (metropolis - gpu).abs() < 0.1,
        "GPU disagrees in 6D: metropolis {metropolis:.4} vs gpu {gpu:.4}"
    );
}

/// A config for a different dimension is a message, not a panic.
///
/// The dimension is the one run parameter a driver names in its own source
/// rather than reading from the file, so a file and a program can disagree about
/// it. `check_dimension` is what turns the disagreement into something a driver
/// can print and exit on, and it names both numbers so the reader knows which
/// end to change.
#[test]
fn a_config_for_another_dimension_is_reported() {
    let text = run_toml([8, 8, 8], 0.3, "metropolis", 10);
    let config = IsingRunConfig::parse(&text).expect("run config should parse");

    assert!(config.check_dimension::<3>().is_ok());

    let message = config
        .check_dimension::<2>()
        .expect_err("a three-axis shape is not a two-dimensional run")
        .to_string();
    assert!(message.contains("built for 2 dimensions"), "{message}");
    assert!(message.contains("names 3"), "{message}");
}
