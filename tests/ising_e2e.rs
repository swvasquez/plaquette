//! End-to-end sampled-physics tests for the Ising model, driving the full public
//! runtime stack across all three update backends.
//!
//! These live in `tests/` rather than beside the code because they are meant to
//! exercise `plaquette` the way a consumer does: through its published API alone,
//! from a TOML string down to a measured observable. Where the crate's inline
//! tests check each piece in isolation, these check that the pieces compose into
//! correct physics — parse a run, build a thermalized [`IsingSampler`], stream
//! configurations off it, and measure each one with [`plaquette::measure`].
//!
//! The physics asserted is the same low-temperature ordering the chain's own
//! `low_temperature_chain_trends_toward_alignment` test uses, and for the same
//! reason. At `beta = 1.0` the two-dimensional Ising model sits well inside the
//! ordered phase (`beta_c ≈ 0.44`), so from any start the chain equilibrates to a
//! near-aligned configuration. Onsager's spontaneous magnetization there is
//! `|m| ≈ 0.999`, which leaves a conservative `> 0.5` threshold on the mean of
//! `|M| / N` both robust and seed-deterministic — a genuine physics check that
//! comfortably clears the bar without riding on the exact value.
//!
//! The three tests differ only in the config's `updater` field, so the same
//! run is measured through the CPU Metropolis, CPU checkerboard, and GPU
//! checkerboard backends. Lattice extents are kept even (`[16, 16]`) because the
//! GPU checkerboard requires them, and the shared helper reuses that shape so the
//! three tests stay directly comparable.

use plaquette::{IsingRunConfig, IsingSampler, measure};

/// The number of samples each run measures after warmup. A few hundred is plenty
/// for the mean of `|m|` to settle deep in the ordered phase, and keeps each test
/// fast.
const N_SAMPLES: usize = 200;

/// Build the run's TOML for a given `updater`, holding everything else fixed.
///
/// The only field that varies across the three tests is `updater`, so pinning the
/// rest here — even `[16, 16]` extents, `beta = 1.0`, a hot start, and a fixed
/// seed — is what makes the backends directly comparable and each run
/// deterministic. The hot start is deliberate: ordering from a disordered start is
/// the stronger check, since it shows the chain *reaches* the ordered phase rather
/// than merely staying in it.
fn run_toml(updater: &str) -> String {
    format!(
        "shape = [16, 16]\n\
         j = 1.0\n\
         beta = 1.0\n\
         updater = \"{updater}\"\n\
         thermalize = 200\n\
         sweeps_between = 1\n\
         n_samples = {N_SAMPLES}\n\
         seed = 20260728\n\
         start = \"hot\"\n"
    )
}

/// Drive the full public stack for `updater` and return the mean of `|M| / N`
/// over the sampled configs.
///
/// This is the whole end-to-end path in one place: parse the TOML, build a
/// thermalized sampler, read the geometry off it once, then stream `N_SAMPLES`
/// configurations and fold each into the running mean of the sign-folded
/// magnetization density. The absolute value is taken per config because the
/// ordered phase settles into either the up or the down well arbitrarily, so the
/// signed mean would cancel while `|m|` reports the ordering the assertion is
/// after.
fn mean_abs_magnetization(updater: &str) -> f64 {
    let config = IsingRunConfig::parse(&run_toml(updater)).expect("run config should parse");
    let mut sampler = IsingSampler::new(&config);

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
    let mean_abs_m = mean_abs_magnetization("metropolis");
    assert!(
        mean_abs_m > 0.5,
        "low-T Metropolis run should order: mean |m| = {mean_abs_m}"
    );
}

/// CPU checkerboard: the two-color sweep produces the same ordering. Same run,
/// same threshold — only the update rule differs.
#[test]
fn cpu_checkerboard_orders_at_low_temperature() {
    let mean_abs_m = mean_abs_magnetization("checkerboard");
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
    let mean_abs_m = mean_abs_magnetization("gpu_checkerboard");
    assert!(
        mean_abs_m > 0.5,
        "low-T GPU checkerboard run should order: mean |m| = {mean_abs_m}"
    );
}
