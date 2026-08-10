//! End-to-end sampled physics for the `q`-state Potts model, across dimensions
//! and across all three update backends.
//!
//! This file is the full-stack validation for the Potts model, and it takes its
//! structure from both of the files beside it: the ordering contrast from
//! `ising_e2e.rs`, and the run-the-same-physics-through-every-backend comparison
//! from `gauge_e2e.rs`. Everything goes through the published API alone —
//! `PottsRunConfig` to `PottsSampler` to `potts_measure` — because the pieces are
//! each covered in isolation by the crate's own unit tests and what is untested
//! until here is that they compose.
//!
//! Two dimensions is where the physics is known exactly: the square lattice is
//! self-dual, and the fixed point of the duality puts the transition at
//! `beta_c = ln(1 + sqrt(q))`, about `1.0050` at `q = 3`, and the internal energy
//! there at `-J (1 + 1/sqrt(q))`. Those are anchors rather than contrasts — the
//! ordering and disorder tests are placed on either side of the first, and
//! [`the_critical_energy_sits_just_below_the_exact_duality_value`] reads a chain
//! sitting on it against the second — and it is why the shipped example runs
//! there too.
//! The transition is continuous for `q <= 4`, so a run near it suffers critical
//! slowing down but not the phase coexistence that keeps the four-dimensional
//! gauge tests well clear of their own transition.
//!
//! Three dimensions has no closed form, so it gets the contrast instead, and it
//! earns its place by being the only thing that shows the genericity over `D` is
//! real: the action, the schedules, and the GPU kernel all derive their loop
//! bounds from the dimension rather than assuming the two the rest of the file
//! runs at.
//!
//! The sharpest single check here is neither: at `q = 2` the Potts model *is* the
//! Ising model at half the coupling, so two independently written actions driven
//! through two independently written samplers must produce the same run. See
//! [`potts_at_two_states_reproduces_an_ising_run`].
//!
//! Lattice extents are kept even throughout, because the GPU schedule requires
//! them.

use plaquette::potts_config::PottsRunConfig;
use plaquette::{
    Estimate, IsingRunConfig, IsingSampler, PottsSampler, measure, potts_measure, reduce,
};

/// Samples per run. A few hundred is enough for the contrasts asserted here
/// without making the suite slow.
const N_SAMPLES: usize = 300;

/// The exact square-lattice critical coupling `beta_c = ln(1 + sqrt(q))` at
/// `j = 1`, which is about `1.0050` for the `q = 3` most of this file runs.
fn beta_c(q: usize) -> f64 {
    (1.0 + (q as f64).sqrt()).ln()
}

/// The run parameters the *comparison* tests hold fixed, as TOML.
///
/// One builder for all of them, so a decorrelation stride or a start that
/// drifted between two of those tests could not silently make their runs
/// incomparable — which matters because several of them assert that two runs
/// agree. The two tests that measure against a closed form build their own TOML
/// instead, since each needs phasing this does not offer: much longer for the
/// critical run, much shorter for the decoupled one. Nothing compares those two
/// against these, so there is nothing for them to drift out of step with.
///
/// The hot start is deliberate throughout: ordering *from* a disordered start
/// shows the chain reaches the ordered phase rather than merely staying in it,
/// and it also means no run begins on the label a cold start would single out.
fn run_toml<const D: usize>(
    shape: [usize; D],
    j: f64,
    beta: f64,
    seed: u64,
    updater: &str,
) -> String {
    format!(
        "shape = {shape:?}\n\
         j = {j}\n\
         beta = {beta}\n\
         updater = \"{updater}\"\n\
         thermalize = 500\n\
         sweeps_between = 3\n\
         n_samples = {N_SAMPLES}\n\
         seed = {seed}\n\
         start = \"hot\"\n"
    )
}

/// What every driven run reports back.
struct Measured {
    /// Chain-mean energy per site, with its autocorrelation-inflated standard
    /// error — the error is what lets one run be compared against another on
    /// statistical terms rather than against a hand-picked tolerance.
    energy: Estimate,
    /// Chain-mean order parameter, likewise. Unlike the Ising magnetization it
    /// is already non-negative, so nothing is folded before reducing.
    order: Estimate,
}

/// Drive a `Q`-state, `D`-dimensional chain described by `toml` and read the
/// energy density and order parameter off the stream.
///
/// The shared half of every test in this file: parse, build a thermalized
/// sampler, stream `samples` configurations, reduce both series. What differs
/// between the tests is the TOML they hand in and nothing else, so this is
/// written once — a second copy would be a second place to fix if what is
/// measured ever changes, and the whole point of several of these tests is that
/// two runs are measured identically.
///
/// Both constants are named at the call site because neither can come from the
/// file: the config's `shape` fixes the dimension it *describes*, but the
/// sampler's `D` is what the program was built for, and nothing in the schema
/// mentions `Q` at all.
fn drive<const Q: usize, const D: usize>(toml: &str, samples: usize) -> Measured {
    let config = PottsRunConfig::parse(toml).expect("hand-written config is valid");
    let mut sampler = PottsSampler::<Q, D>::new(&config);

    // Geometry and model come off the sampler as owned values, read once so the
    // stream can borrow the sampler by itself.
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_sites = lattice.n_sites() as f64;

    let mut energy = Vec::with_capacity(samples);
    let mut order = Vec::with_capacity(samples);
    for config in sampler.samples().take(samples) {
        let sample = potts_measure(&model, &lattice, &config);
        energy.push(sample.energy / n_sites);
        order.push(sample.order);
    }

    Measured {
        energy: reduce(&energy),
        order: reduce(&order),
    }
}

/// Drive a `D`-dimensional `Q`-state Potts chain of `shape` under the comparison
/// phasing — see [`run_toml`].
fn run<const Q: usize, const D: usize>(
    shape: [usize; D],
    j: f64,
    beta: f64,
    seed: u64,
    updater: &str,
) -> Measured {
    drive::<Q, D>(&run_toml(shape, j, beta, seed, updater), N_SAMPLES)
}

/// One line of a `Measured`, for the record a failing comparison is read from.
fn report(label: &str, m: &Measured) {
    eprintln!(
        "{label:<28} E/N={:.4}({:.4}) m={:.4}({:.4}) tau={:.2}",
        m.energy.mean, m.energy.stderr, m.order.mean, m.order.stderr, m.order.tau_int
    );
}

/// Assert that a run under one schedule measures the same physics as the
/// random-order Metropolis run it reorders.
///
/// Agreement is asserted statistically rather than against a hand-picked
/// tolerance: the two runs are independent, so the difference of their means has
/// standard error `sqrt(se_a^2 + se_b^2)`, and a correct pair of schedules
/// exceeds four of those about as often as a fair coin lands heads sixteen times
/// running. The errors come from `reduce`, which inflates by the integrated
/// autocorrelation time — which matters here because reordering the moves is
/// precisely what changes `tau_int`, and a naive `stderr` would understate the
/// slower chain. The comparison cannot be bit-for-bit against any backend:
/// Metropolis draws a site index per step, the CPU checkerboard draws none, and
/// the GPU keys a counter on `(seed, site, sweep)` and maps its proposal draw
/// onto the alternatives differently again, so all three run on different
/// streams.
fn assert_matches(reference: &Measured, other: &Measured, label: &str) {
    for (what, a, b) in [
        ("energy density", &reference.energy, &other.energy),
        ("order parameter", &reference.order, &other.order),
    ] {
        let sigma = (a.stderr.powi(2) + b.stderr.powi(2)).sqrt();
        let gap = (a.mean - b.mean).abs();
        assert!(
            gap < 4.0 * sigma,
            "{label}: {what} differs by {gap} = {:.1} sigma: metropolis {}, {label} {}",
            gap / sigma,
            a.mean,
            b.mean
        );
    }
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

/// Well below the transition the chain orders, on every backend.
///
/// `beta = 1.5` against `beta_c ~ 1.0050` is deep enough that the ordered phase
/// is unambiguous from any start, which leaves a conservative `> 0.5` threshold
/// on the mean order parameter both robust and seed-deterministic — the same
/// threshold and the same reasoning `ising_e2e.rs` uses at its own low
/// temperature. All three backends are run here rather than in three separate
/// tests, since the assertion and the run are identical and only the config's
/// `updater` field differs.
#[test]
fn ordering_below_the_transition_holds_on_every_backend() {
    let shape = [16, 16];
    let beta = 1.5;
    assert!(beta > beta_c(3), "this test must sit in the ordered phase");

    for (seed, updater) in [(20260901, "metropolis"), (20260902, "site_checkerboard")] {
        let measured = run::<3, 2>(shape, 1.0, beta, seed, updater);
        report(updater, &measured);
        assert!(
            measured.order.mean > 0.5,
            "{updater}: a run at beta = {beta} should order, got m = {}",
            measured.order.mean
        );
    }

    if !gpu_available() {
        return;
    }
    let gpu = run::<3, 2>(shape, 1.0, beta, 20260903, "gpu_site_checkerboard");
    report("gpu_site_checkerboard", &gpu);
    assert!(
        gpu.order.mean > 0.5,
        "gpu_site_checkerboard: a run at beta = {beta} should order, got m = {}",
        gpu.order.mean
    );
}

/// Well above the transition the chain does not order — the complement of the
/// test above, and the check that the model is not trivially ordered by a bug.
///
/// `beta = 0.4` is far enough below `beta_c ~ 1.0050` that the labels are nearly
/// independent, and the order parameter then sits near its finite-size floor
/// rather than at zero: `f_max` is the largest of three fluctuating fractions,
/// so it sits a little above `1/3` even with no correlation at all, and `m`
/// inherits that. On 256 sites the floor is a few hundredths, so a `< 0.25`
/// threshold is well clear of it while still being nowhere near the `> 0.5` the
/// ordered phase clears.
#[test]
fn disorder_above_the_transition_holds_on_every_backend() {
    let shape = [16, 16];
    let beta = 0.4;
    assert!(
        beta < beta_c(3),
        "this test must sit in the disordered phase"
    );

    for (seed, updater) in [(20260904, "metropolis"), (20260905, "site_checkerboard")] {
        let measured = run::<3, 2>(shape, 1.0, beta, seed, updater);
        report(updater, &measured);
        assert!(
            measured.order.mean < 0.25,
            "{updater}: a run at beta = {beta} should not order, got m = {}",
            measured.order.mean
        );
    }

    if !gpu_available() {
        return;
    }
    let gpu = run::<3, 2>(shape, 1.0, beta, 20260906, "gpu_site_checkerboard");
    report("gpu_site_checkerboard", &gpu);
    assert!(
        gpu.order.mean < 0.25,
        "gpu_site_checkerboard: a run at beta = {beta} should not order, got m = {}",
        gpu.order.mean
    );
}

/// The coupling the agreement tests run at: inside the ordered phase but close
/// enough to `beta_c ~ 1.0050` that the order parameter has not saturated, so a
/// backend computing something slightly wrong still has room to show it. Far
/// enough from it that `reduce` can resolve `tau_int` at this sample count,
/// which the error bars the comparison rests on depend on.
const AGREEMENT_BETA: f64 = 1.1;

/// The three backends agree in two dimensions.
///
/// Reordering the moves into color passes, and running a color at once on a
/// device, is supposed to change nothing but the autocorrelation — so agreement
/// is what is asserted, statistically, rather than a threshold either could pass
/// while computing the wrong thing.
#[test]
fn the_backends_agree_in_two_dimensions() {
    let shape = [16, 16];
    let metropolis = run::<3, 2>(shape, 1.0, AGREEMENT_BETA, 20260910, "metropolis");
    let checkerboard = run::<3, 2>(shape, 1.0, AGREEMENT_BETA, 20260911, "site_checkerboard");
    report("2D metropolis", &metropolis);
    report("2D site_checkerboard", &checkerboard);
    assert_matches(&metropolis, &checkerboard, "site_checkerboard");

    if !gpu_available() {
        return;
    }
    let gpu = run::<3, 2>(
        shape,
        1.0,
        AGREEMENT_BETA,
        20260912,
        "gpu_site_checkerboard",
    );
    report("2D gpu_site_checkerboard", &gpu);
    assert_matches(&metropolis, &gpu, "gpu_site_checkerboard");
}

/// Three dimensions: the ordering contrast, and the same backend agreement.
///
/// This is the arm that says the genericity over `D` is real rather than
/// incidental. The action walks a neighbor row `2D` wide, the GPU kernel derives
/// that width from a `D` arriving as a WGSL `override`, and nothing in the
/// two-dimensional tests above would notice either going wrong. A contrast
/// rather than a value, because three dimensions has no closed form: the two
/// couplings straddle a transition near `beta_c ~ 0.55`, and both sit far enough
/// from it that a short run is not deciding between coexisting phases — the
/// three-dimensional transition is first order even at `q = 3`, unlike the
/// two-dimensional one.
#[test]
fn three_dimensional_runs_order_below_the_transition_and_not_above() {
    let shape = [6, 6, 6];

    let ordered = run::<3, 3>(shape, 1.0, 0.8, 20260920, "metropolis");
    let disordered = run::<3, 3>(shape, 1.0, 0.3, 20260921, "metropolis");
    report("3D beta=0.8", &ordered);
    report("3D beta=0.3", &disordered);
    assert!(
        ordered.order.mean > 0.7,
        "beta = 0.8 is below beta_c ~ 0.55, so the run should order: {}",
        ordered.order.mean
    );
    assert!(
        disordered.order.mean < 0.25,
        "beta = 0.3 is above beta_c, so the run should not order: {}",
        disordered.order.mean
    );

    // Agreement at a coupling on the ordered side but off the saturation
    // ceiling, for the same reason `AGREEMENT_BETA` sits where it does in two
    // dimensions.
    let metropolis = run::<3, 3>(shape, 1.0, 0.65, 20260922, "metropolis");
    let checkerboard = run::<3, 3>(shape, 1.0, 0.65, 20260923, "site_checkerboard");
    report("3D metropolis", &metropolis);
    report("3D site_checkerboard", &checkerboard);
    assert_matches(&metropolis, &checkerboard, "site_checkerboard");

    if !gpu_available() {
        return;
    }
    let gpu = run::<3, 3>(shape, 1.0, 0.65, 20260924, "gpu_site_checkerboard");
    report("3D gpu_site_checkerboard", &gpu);
    assert_matches(&metropolis, &gpu, "gpu_site_checkerboard");
}

/// At two states, a Potts run at coupling `2J` reproduces a zero-field Ising run
/// at `J` — configuration for configuration.
///
/// This is the strongest single check that the new model is right, and it is
/// exact rather than statistical. Reading the two labels as `±1` gives
/// `delta(s_i, s_j) = (1 + s_i s_j) / 2`, so the two energies differ by the
/// constant `-J` per bond and every `energy_delta` is *identical*. Both sides
/// are integer arithmetic scaled once, so that identity holds bit-for-bit; both
/// samplers seed one generator, draw the same hot start from it, and then run
/// the same `Metropolis` schedule, whose two-state proposal consumes nothing.
/// Equal deltas therefore mean equal accept decisions drawn from streams that
/// never diverge, and the two chains stay in lockstep for the whole run.
///
/// So this exercises the whole runtime — two config schemas, two samplers, two
/// actions, two observables — against a correspondence that leaves no room for a
/// tolerance to hide in. The mean energies are asserted separately, since that
/// is the quantity a reader can check against the arithmetic by hand.
#[test]
fn potts_at_two_states_reproduces_an_ising_run() {
    const J: f64 = 1.0;
    let shape = [16, 16];
    let beta = 0.5; // ordered for Ising (beta_c ~ 0.4407), and so for Potts at 2J
    let seed = 20260930;

    let potts_toml = run_toml(shape, 2.0 * J, beta, seed, "metropolis");
    let potts_config = PottsRunConfig::parse(&potts_toml).expect("hand-written config is valid");
    let mut potts = PottsSampler::<2, 2>::new(&potts_config);

    let ising_toml = run_toml(shape, J, beta, seed, "metropolis");
    let ising_config = IsingRunConfig::parse(&ising_toml).expect("hand-written config is valid");
    let mut ising = IsingSampler::<2>::new(&ising_config);

    let lattice = potts.lattice();
    let (potts_model, ising_model) = (potts.model(), ising.model());
    // The constant the two energies differ by: `-J` per bond, and a periodic
    // lattice has one forward bond per site per axis, which is its link count.
    let offset = -J * lattice.n_links() as f64;

    let mut potts_energy = Vec::with_capacity(N_SAMPLES);
    let mut ising_energy = Vec::with_capacity(N_SAMPLES);
    for (from_potts, from_ising) in potts
        .samples()
        .take(N_SAMPLES)
        .zip(ising.samples().take(N_SAMPLES))
    {
        assert_eq!(
            from_potts, from_ising,
            "the two chains should not have diverged"
        );
        let e_potts = potts_measure(&potts_model, &lattice, &from_potts).energy;
        let e_ising = measure(&ising_model, &lattice, &from_ising).energy;
        assert_eq!(
            e_potts,
            e_ising + offset,
            "per-sample energy correspondence"
        );
        potts_energy.push(e_potts);
        ising_energy.push(e_ising);
    }

    let potts_mean = reduce(&potts_energy).mean;
    let ising_mean = reduce(&ising_energy).mean;
    eprintln!(
        "q = 2 correspondence: potts <E> = {potts_mean:.4}, \
         ising <E> = {ising_mean:.4}, offset = {offset}"
    );
    assert!(
        (potts_mean - (ising_mean + offset)).abs() < 1e-9,
        "mean energies should agree once the constant offset is removed"
    );
}

/// Samples the critical-energy run takes, and the phasing it takes them with.
///
/// Far longer than [`N_SAMPLES`], and not by preference. A local update
/// decorrelates slowly at a continuous transition, which is exactly where this
/// run has to sit, so the chain needs both a long warmup and a wide gap before
/// `reduce` can resolve its own autocorrelation. At the lattice size below that
/// still leaves an integrated autocorrelation time of six or seven sweeps'
/// worth, against about one for the runs elsewhere in this file.
const CRITICAL_SAMPLES: usize = 2000;

/// Mean energy per site of a `Q`-state chain sitting exactly on the
/// square-lattice critical coupling.
///
/// Sixteen squared is chosen against two errors pulling in opposite directions.
/// The finite-size correction to the critical energy falls off slowly, so a
/// bigger box would sit closer to the exact value; but the autocorrelation grows
/// roughly as `L^2` at criticality, and past this size a run affordable in a
/// test stops resolving its own mean. Measured at `q = 3` over three seeds each,
/// the gap below the exact value and its seed-to-seed scatter are:
///
/// ```text
/// L        16       24       32       48
/// gap    0.038    0.028    0.025    0.012
/// scatter 0.005    0.009    0.007    0.007
/// ```
///
/// so by thirty-two the scatter has caught up with the ground a larger box
/// would gain, and the trade stops paying. The small box is therefore
/// deliberate, and the residual gap is accounted for in the tolerance rather
/// than sampled away.
fn energy_at_criticality<const Q: usize>(seed: u64) -> Estimate {
    let toml = format!(
        "shape = [16, 16]\n\
         j = 1.0\n\
         beta = {}\n\
         updater = \"metropolis\"\n\
         thermalize = 4000\n\
         sweeps_between = 10\n\
         n_samples = {CRITICAL_SAMPLES}\n\
         seed = {seed}\n\
         start = \"hot\"\n",
        beta_c(Q)
    );
    drive::<Q, 2>(&toml, CRITICAL_SAMPLES).energy
}

/// How far the measured critical energy may sit from the exact one.
///
/// Set by the finite-size correction rather than by sampling error, which is the
/// unusual part. At sixteen squared the measured energy runs about `0.038` below
/// the infinite-volume value, with a seed-to-seed scatter of roughly `0.013` on
/// top — so the bound has to clear about `0.05`, and `0.06` leaves margin. It
/// stays a real check despite the width, because every plausible way of getting
/// the formula wrong misses by more than twice this: the nearest miss is reading
/// it at the wrong `q`, which is `0.13` away, and dropping the square root
/// entirely is `0.24`.
const CRITICAL_ENERGY_TOLERANCE: f64 = 0.06;

/// The internal energy at the critical point, against the exact duality value.
///
/// This is the one place a sampled Potts chain meets a closed form, and it is
/// the counterpart of `gauge_e2e.rs`'s exact area law. Self-duality of the
/// square lattice fixes both the transition and the energy at it,
///
/// ```text
/// beta_c * J = ln(1 + sqrt(q)),     E_c / N = -J * (1 + 1 / sqrt(q)),
/// ```
///
/// the second following from differentiating the duality relation at the
/// self-dual point. The pair is checked together, since the run is placed at the
/// first coupling and read against the second: a wrong `beta_c` would move the
/// measured energy off the value, and a wrong energy formula would not be met by
/// a correctly placed run.
///
/// Both `q = 2` and `q = 3` are run, because the whole content of the formula is
/// its `q`-dependence and one value cannot show that. The `q = 2` case is also
/// independently anchored: pushing Onsager's exact critical Ising energy of
/// minus root two per site through the constant offset between the two models
/// lands on exactly minus one plus one over root two, which is what this asserts.
///
/// What is asserted is deliberately not that the measured energy *equals* the
/// exact one, because on a finite lattice it does not and should not. The
/// measured value sits a little below it, by a gap that shrinks as the lattice
/// grows — roughly as `1/L` over the sizes in
/// [`energy_at_criticality`]'s table, and always with the same sign, since a
/// finite lattice is more ordered than an infinite one at the same coupling. So
/// the assertions are that the gap has that sign and is within
/// [`CRITICAL_ENERGY_TOLERANCE`], which together say the deviation is a
/// finite-size correction rather than a wrong number: noise would not keep a
/// sign, and a wrong formula would not land this close at both `q`.
///
/// The chain's own `n_eff` is asserted alongside, so a run that never
/// decorrelated cannot pass by drifting near the answer for the wrong reason.
#[test]
fn the_critical_energy_sits_just_below_the_exact_duality_value() {
    let two = energy_at_criticality::<2>(20260940);
    let three = energy_at_criticality::<3>(20260941);

    for (q, measured) in [(2usize, &two), (3usize, &three)] {
        let exact = -(1.0 + 1.0 / (q as f64).sqrt());
        let gap = measured.mean - exact;
        eprintln!(
            "q = {q} at beta_c = {:.6}: E/N = {:.4}({:.4}), exact {exact:.4}, \
             gap {gap:+.4}, tau = {:.2}, n_eff = {:.0}",
            beta_c(q),
            measured.mean,
            measured.stderr,
            measured.tau_int,
            measured.n_eff
        );
        assert!(
            measured.is_reliable(),
            "q = {q}: the chain did not resolve its own autocorrelation \
             (n_eff = {:.0}), so the comparison below would be meaningless",
            measured.n_eff
        );
        assert!(
            gap < 0.0,
            "q = {q}: a finite lattice should measure below the exact critical \
             energy, but {:.4} sits above {exact:.4}",
            measured.mean
        );
        assert!(
            gap.abs() < CRITICAL_ENERGY_TOLERANCE,
            "q = {q}: measured critical energy {:.4} vs exact {exact:.4}, \
             off by {gap:+.4}, more than a finite-size correction at this size",
            measured.mean
        );
    }
}

/// Drive a `Q = 3` run with the coupling switched *off* and one label offset,
/// on the named backend.
///
/// Turning `j` off is what makes this exact rather than approximate: with no
/// neighbor term the sites stop interacting entirely, so each one is an
/// independent draw from a three-outcome Boltzmann distribution and the label
/// populations have a closed form at any lattice size. It also decorrelates in a
/// sweep, so the run needs none of the warmup the critical test does.
fn run_decoupled(offset: f64, beta: f64, seed: u64, updater: &str) -> Measured {
    let toml = format!(
        "shape = [16, 16]\n\
         j = 0.0\n\
         h = [{offset}, 0.0, 0.0]\n\
         beta = {beta}\n\
         updater = \"{updater}\"\n\
         thermalize = 100\n\
         sweeps_between = 2\n\
         n_samples = {N_SAMPLES}\n\
         seed = {seed}\n\
         start = \"hot\"\n"
    );
    drive::<3, 2>(&toml, N_SAMPLES)
}

/// A per-label offset with the coupling switched off reproduces the exact
/// single-site Boltzmann populations, on every backend.
///
/// This is the sharpest check on the offset term, and the only exact one that
/// costs nothing: at `j = 0` the sites decouple completely, so the fraction
/// carrying the favoured label is
///
/// ```text
/// f = exp(beta * h) / (exp(beta * h) + (q - 1)),
/// ```
///
/// with no finite-size correction and no critical slowing down to work around —
/// the prediction holds at any lattice size, and one sweep decorrelates the
/// chain. Both the energy density, which at `j = 0` is just `-h * f`, and the
/// order parameter follow from it, so a sign error or a factor in the offset
/// shows up immediately in a number rather than in a trend.
///
/// Running all three backends is the point: the GPU kernel carries the offsets
/// in its own uniform block, packed four to a `vec4`, and unpacks them with
/// arithmetic the CPU path does not share. Nothing else in this file would catch
/// that unpacking being wrong.
#[test]
fn a_decoupled_run_reproduces_the_exact_boltzmann_populations() {
    const Q: f64 = 3.0;
    let (offset, beta): (f64, f64) = (2.0, 1.0);

    let weight = (beta * offset).exp();
    let favoured = weight / (weight + (Q - 1.0));
    let exact_energy = -offset * favoured;
    let exact_order = (Q * favoured - 1.0) / (Q - 1.0);

    let backends = [
        (20260950, "metropolis"),
        (20260951, "site_checkerboard"),
        (20260952, "gpu_site_checkerboard"),
    ];
    for (seed, updater) in backends {
        if updater.starts_with("gpu") && !gpu_available() {
            continue;
        }
        let measured = run_decoupled(offset, beta, seed, updater);
        report(updater, &measured);

        // The sites are independent, so `reduce` sees a nearly uncorrelated
        // series and its error bar is trustworthy; four of them is a wide
        // enough window to be seed-robust and narrow enough to catch a wrong
        // offset, which would move these by tenths.
        for (what, estimate, exact) in [
            ("energy density", &measured.energy, exact_energy),
            ("order parameter", &measured.order, exact_order),
        ] {
            let gap = (estimate.mean - exact).abs();
            assert!(
                gap < 4.0 * estimate.stderr.max(1e-3),
                "{updater}: {what} {:.4} vs exact {exact:.4}, off by {gap:.4}",
                estimate.mean
            );
        }
    }
}

/// A config for a different dimension is a message, not a panic.
///
/// The dimension is one of the two parameters a driver names in its own source
/// rather than reading from the file, so a file and a program can disagree about
/// it. `check_dimension` is what turns the disagreement into something a driver
/// can print and exit on, and it names both numbers so the reader knows which
/// end to change. The state count has no counterpart to this, because nothing in
/// the file mentions it — which is exactly why the dimension check is worth
/// having where it is possible.
#[test]
fn a_config_for_another_dimension_is_reported() {
    let text = run_toml([8, 8, 8], 1.0, 0.8, 1, "metropolis");
    let config = PottsRunConfig::parse(&text).expect("run config should parse");

    assert!(config.check_dimension::<3>().is_ok());

    let message = config
        .check_dimension::<2>()
        .expect_err("a three-axis shape is not a two-dimensional run")
        .to_string();
    assert!(message.contains("built for 2 dimensions"), "{message}");
    assert!(message.contains("names 3"), "{message}");
}
