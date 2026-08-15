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
/// The `updater`/`schedule`/`backend` lines for a shorthand name, so a test
/// names a run the way its report labels it while the file speaks the composed
/// config vocabulary.
fn driver_lines(name: &str) -> String {
    let (rule, schedule, backend) = match name {
        "metropolis" => ("metropolis", None, None),
        "heat_bath" => ("heat_bath", None, None),
        "checkerboard" => ("metropolis", Some("checkerboard"), None),
        "checkerboard_heat_bath" => ("heat_bath", Some("checkerboard"), None),
        "gpu_checkerboard" => ("metropolis", Some("checkerboard"), Some("gpu")),
        "gpu_checkerboard_heat_bath" => ("heat_bath", Some("checkerboard"), Some("gpu")),
        "swendsen_wang" => ("swendsen_wang", None, None),
        "gpu_swendsen_wang" => ("swendsen_wang", None, Some("gpu")),
        "wolff" => ("wolff", None, None),
        "gpu_wolff" => ("wolff", None, Some("gpu")),
        other => panic!("unknown updater shorthand {other}"),
    };
    let mut lines = format!("updater = \"{rule}\"\n");
    if let Some(schedule) = schedule {
        lines.push_str(&format!("schedule = \"{schedule}\"\n"));
    }
    if let Some(backend) = backend {
        lines.push_str(&format!("backend = \"{backend}\"\n"));
    }
    lines
}

/// The general TOML builder, with the pacing named by the caller.
///
/// Almost every run here uses the shared pacing [`run_toml`] fixes, so the two
/// runs of a comparison cannot drift apart. The exception is Wolff: its sweep
/// is one cluster move rather than a pass over the lattice (see
/// `docs/wolff.md`), so a Wolff run needs its warmup and stride several times
/// larger to do the same amount of updating.
fn run_toml_paced<const D: usize>(
    shape: [usize; D],
    beta: f64,
    updater: &str,
    n_samples: usize,
    thermalize: usize,
    sweeps_between: usize,
) -> String {
    format!(
        "shape = {shape:?}\n\
         j = 1.0\n\
         beta = {beta}\n\
         {driver}\
         thermalize = {thermalize}\n\
         sweeps_between = {sweeps_between}\n\
         n_samples = {n_samples}\n\
         seed = 20260728\n\
         start = \"hot\"\n",
        driver = driver_lines(updater),
    )
}

fn run_toml<const D: usize>(
    shape: [usize; D],
    beta: f64,
    updater: &str,
    n_samples: usize,
) -> String {
    run_toml_paced(shape, beta, updater, n_samples, 500, 2)
}

/// Drive the full public stack over `text` and return the mean of `|M| / N`
/// over the sampled configs.
///
/// The absolute value is taken per config because the ordered phase settles into
/// either the up or the down well arbitrarily, so the signed mean would cancel
/// while `|m|` reports the ordering the assertions are after.
fn mean_abs_magnetization_from<const D: usize>(text: &str) -> f64 {
    let config = IsingRunConfig::parse(text).expect("run config should parse");
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

/// [`mean_abs_magnetization_from`] under the shared comparison pacing.
fn mean_abs_magnetization<const D: usize>(shape: [usize; D], beta: f64, updater: &str) -> f64 {
    mean_abs_magnetization_from::<D>(&run_toml(shape, beta, updater, N_SAMPLES))
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
    let mean_abs_m = mean_abs_magnetization([16, 16], 1.0, "checkerboard");
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
    let mean_abs_m = mean_abs_magnetization([16, 16], 1.0, "gpu_checkerboard");
    assert!(
        mean_abs_m > 0.5,
        "low-T GPU checkerboard run should order: mean |m| = {mean_abs_m}"
    );
}

/// The heat bath agrees with Metropolis on both backends, through the whole
/// public stack.
///
/// `beta = 0.5` is the same window
/// [`the_cluster_update_agrees_with_metropolis`] uses and for the same reason:
/// `|m|` there is about 0.91 and still moving, so an effective coupling that
/// came out half or double this would land at 0.25 or 1.0 and miss the window
/// by more than ten times its width. That is the failure a heat bath is most
/// likely to have — the conditional is one rearrangement away from a sign error
/// or a factor of two in the exponent, and a chain carrying either still runs,
/// still orders, and still looks healthy.
///
/// The GPU arm is what makes the shader's correctness a claim about the stack
/// rather than about a kernel: the config names the heat bath rule under the
/// checkerboard schedule on the gpu backend, `IsingSampler` builds the device
/// chain with `Kernel::HeatBath`, and the same
/// comparison decides it. The three runs cannot be compared bit-for-bit —
/// Metropolis draws a site index per step, the CPU heat bath draws one uniform
/// per variable, and the GPU keys a counter on `(seed, site, sweep)` — so the
/// agreement is distributional.
#[test]
fn the_heat_bath_agrees_with_metropolis() {
    let shape = [16, 16];
    let beta = 0.5;

    let metropolis = mean_abs_magnetization(shape, beta, "metropolis");
    let heat_bath = mean_abs_magnetization(shape, beta, "heat_bath");
    assert!(
        (0.7..0.99).contains(&metropolis),
        "the reference should be ordered without saturating: {metropolis:.4}"
    );
    assert!(
        (metropolis - heat_bath).abs() < 0.05,
        "heat bath disagrees on the CPU: metropolis {metropolis:.4} vs heat bath {heat_bath:.4}"
    );

    // The device kernel's sequential reference: the same coloring, the same
    // kernel, run one site at a time. Comparing the GPU against this rather than
    // only against Metropolis is what makes the check specific to the shader —
    // agreement with a differently-scheduled run leaves open that the coloring
    // and the kernel are each wrong in compensating ways.
    let cpu_checkerboard = mean_abs_magnetization(shape, beta, "checkerboard_heat_bath");
    assert!(
        (metropolis - cpu_checkerboard).abs() < 0.05,
        "the checkerboard heat bath disagrees: metropolis {metropolis:.4} \
         vs checkerboard {cpu_checkerboard:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization(shape, beta, "gpu_checkerboard_heat_bath");
    assert!(
        (cpu_checkerboard - gpu).abs() < 0.05,
        "the GPU heat bath disagrees with its sequential reference: \
         checkerboard {cpu_checkerboard:.4} vs gpu {gpu:.4}"
    );
}

/// The heat bath runs a model the cluster update refuses.
///
/// An external field breaks the relabeling symmetry Swendsen–Wang needs, so a
/// cluster run of this config fails to load at all. The heat bath carries the
/// field in `ΔE` and needs no symmetry, and both backends have to agree with
/// Metropolis on where the field drives the magnetization. On the GPU that also
/// exercises the one term of the kernel the field-free tests leave at zero.
#[test]
fn the_heat_bath_runs_with_an_external_field() {
    let shape = [16, 16];
    let beta = 0.3;
    let h = 0.3;

    let with_field = |updater: &str| {
        // `run_toml` leaves `h` at its default of zero, so the field is added
        // here rather than by substitution.
        let text = format!("{}h = {h}\n", run_toml(shape, beta, updater, N_SAMPLES));
        let config = IsingRunConfig::parse(&text).expect("run config should parse");
        let mut sampler = IsingSampler::<2>::new(&config);
        let lattice = sampler.lattice();
        let model = sampler.model();
        let n_sites = lattice.n_sites() as f64;
        sampler
            .samples()
            .take(N_SAMPLES)
            .map(|c| measure(&model, &lattice, &c).magnetization / n_sites)
            .sum::<f64>()
            / N_SAMPLES as f64
    };

    // The signed mean, not |m|: the field picks the well, so the sign is the
    // part a mishandled offset term would get wrong.
    let metropolis = with_field("metropolis");
    let heat_bath = with_field("heat_bath");
    assert!(
        (0.4..0.95).contains(&metropolis),
        "the field should magnetize the reference without saturating it: {metropolis:.4}"
    );
    assert!(
        (metropolis - heat_bath).abs() < 0.05,
        "heat bath disagrees under a field: metropolis {metropolis:.4} vs heat bath {heat_bath:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = with_field("gpu_checkerboard_heat_bath");
    assert!(
        (metropolis - gpu).abs() < 0.05,
        "GPU heat bath disagrees under a field: metropolis {metropolis:.4} vs gpu {gpu:.4}"
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
///
/// The local update and both cluster updates are held to the same closed form,
/// and for the cluster ones this is the sharpest validation on this model. What
/// it catches is the bond gap. Ising's `±1` convention makes a broken bond cost
/// `2J` where Potts's delta convention makes it cost `J`, so the bond
/// probability is `1 - exp(-2 beta J)` here — and a chain run at the other one
/// would sample a perfectly healthy Ising model at *half* the coupling. At
/// `beta = 0.4` that moves `t` from `0.380` to `0.197` and the near-neighbor
/// correlator by nearly a fifth, which is an order of magnitude outside the
/// tolerance below. One dimension is also where a cluster update is at its most
/// exposed: with two neighbors per site a cluster is an interval, so the bond
/// probability sets its length distribution directly rather than through a
/// percolation threshold.
#[test]
fn one_dimensional_correlator_matches_the_exact_solution() {
    const N: usize = 16;
    const BETA: f64 = 0.4;
    const SAMPLES: usize = 4000;

    // Wolff runs under its own pacing — its sweep is one cluster, so the
    // shared two-sweep stride would leave consecutive samples correlated
    // enough to crowd the tolerance below.
    for (updater, thermalize, sweeps_between) in [
        ("metropolis", 500, 2),
        ("swendsen_wang", 500, 2),
        ("wolff", 500, 8),
    ] {
        let text = run_toml_paced([N], BETA, updater, SAMPLES, thermalize, sweeps_between);
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
                "{updater}: C_{r} measured {measured:.4} vs exact {exact:.4}"
            );
        }
    }
}

/// The cluster update agrees with the local one on both the CPU and the GPU.
///
/// `beta = 0.5` sits just inside the ordered phase, where `|m|` is about 0.91 and
/// still moving — which is what makes this decisive about the bond gap. Ising
/// scores a broken bond at twice what the Potts delta convention charges, and a
/// cluster run that used the wrong one would sample at an effective coupling of
/// half or double this. Half lands at `beta = 0.25`, well into the disordered
/// phase; double lands at `beta = 1.0`, where `|m|` is pinned near one. Either
/// misses the window below by more than ten times its width.
///
/// The GPU arm also says the device cluster chain is genuinely model-free. It is
/// the same `GpuClusterChain` the Potts runs use, reaching this model through
/// `BondAction` alone, so what is being checked here is that the seam carries the
/// right number rather than that a second kernel was written correctly.
#[test]
fn the_cluster_update_agrees_with_metropolis() {
    let shape = [16, 16];
    let beta = 0.5;

    let metropolis = mean_abs_magnetization(shape, beta, "metropolis");
    let cluster = mean_abs_magnetization(shape, beta, "swendsen_wang");
    assert!(
        (0.7..0.99).contains(&metropolis),
        "the reference should be ordered without saturating: {metropolis:.4}"
    );
    assert!(
        (metropolis - cluster).abs() < 0.05,
        "cluster disagrees on the CPU: metropolis {metropolis:.4} vs cluster {cluster:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization(shape, beta, "gpu_swendsen_wang");
    assert!(
        (metropolis - gpu).abs() < 0.05,
        "cluster disagrees on the GPU: metropolis {metropolis:.4} vs gpu {gpu:.4}"
    );
}

/// The Wolff update agrees with the local one on both the CPU and the GPU.
///
/// The same window and the same reasoning as
/// [`the_cluster_update_agrees_with_metropolis`]: at `beta = 0.5` a run at
/// half or double the effective coupling misses it by more than ten times its
/// width, and the bond gap's factor of two (Ising scores a broken bond at
/// `2J`) is the likeliest such error. What is new here is the extent: one
/// Wolff sweep is one seeded cluster rather than a lattice pass, so the run is
/// paced with a longer warmup and a wider stride to do comparable updating —
/// which also makes this the end-to-end statement that the config-level sweep
/// accounting of `docs/wolff.md` (W6) is a real difference and not a
/// documentation nicety.
///
/// The GPU arm exercises the other construction entirely: the device grows no
/// cluster but filters one out of the full decomposition by its seed site's
/// root, so CPU–GPU agreement here is two independent realizations of the same
/// move agreeing, not shared code agreeing with itself.
#[test]
fn the_wolff_update_agrees_with_metropolis() {
    let shape = [16, 16];
    let beta = 0.5;
    let (thermalize, sweeps_between) = (3000, 10);

    let metropolis = mean_abs_magnetization(shape, beta, "metropolis");
    let wolff = mean_abs_magnetization_from::<2>(&run_toml_paced(
        shape,
        beta,
        "wolff",
        N_SAMPLES,
        thermalize,
        sweeps_between,
    ));
    assert!(
        (0.7..0.99).contains(&metropolis),
        "the reference should be ordered without saturating: {metropolis:.4}"
    );
    assert!(
        (metropolis - wolff).abs() < 0.05,
        "wolff disagrees on the CPU: metropolis {metropolis:.4} vs wolff {wolff:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization_from::<2>(&run_toml_paced(
        shape,
        beta,
        "gpu_wolff",
        N_SAMPLES,
        thermalize,
        sweeps_between,
    ));
    assert!(
        (metropolis - gpu).abs() < 0.05,
        "wolff disagrees on the GPU: metropolis {metropolis:.4} vs gpu {gpu:.4}"
    );
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
    let checkerboard = mean_abs_magnetization(shape, beta, "checkerboard");
    assert!(
        (metropolis - checkerboard).abs() < 0.05,
        "CPU schedules disagree in 3D: metropolis {metropolis:.4} vs checkerboard {checkerboard:.4}"
    );

    if !gpu_available() {
        return;
    }
    let gpu = mean_abs_magnetization(shape, beta, "gpu_checkerboard");
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
    let checkerboard = mean_abs_magnetization(shape, beta, "checkerboard");
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
    let gpu = mean_abs_magnetization(shape, beta, "gpu_checkerboard");
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
