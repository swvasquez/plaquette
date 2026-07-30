//! End-to-end sampled physics for the 3D Z2 gauge theory.
//!
//! This file is the full-stack 3D validation for the gauge model, the last of
//! the CPU-completeness tests. The geometry, action, and observables are each
//! covered by unit tests in isolation, and `model.rs` checks the *exact* 2D area
//! law where Z2 is solvable — but nothing drove a whole 3D chain through the
//! public runtime and asserted that the physics comes out, which is the gap this
//! closes. It runs `GaugeRunConfig` -> `GaugeSampler` at a strong and a weak
//! coupling and confirms the defining behaviour: confinement at strong coupling,
//! weakening toward the deconfined phase as the coupling grows past the
//! transition near `beta_c ~ 0.76`.
//!
//! It reads two things off the sampled stream. The mean plaquette rises from near
//! its strong-coupling value toward one as `beta` grows, the coarsest signal that
//! the field is ordering. The Creutz ratio, computed from the Wilson-loop table
//! through `creutz_ratio`, is the string tension: large in the confined phase,
//! small in the deconfined one. Asserting the *contrast* between a strong and a
//! weak coupling — rather than a precise value — keeps the test robust to the
//! statistics of a short run while still exercising the whole stack, sampler and
//! observables and reduction together.

use plaquette::{
    GaugeRunConfig, GaugeSampler, creutz_ratio, gauge_measure, reduce, wilson_rectangles,
};

/// Largest Wilson side to collect. Two is enough for the 2x2 Creutz ratio, and
/// stays within half the extent of the 6^3 box.
const MAX_SIDE: usize = 2;

/// Samples per run. A few hundred is enough for the coarse contrasts asserted
/// here without making the test slow.
const N_SAMPLES: usize = 300;

/// What one driven run reports back.
struct Measured {
    /// Chain-mean plaquette, `<sigma_plaq>`.
    plaquette: f64,
    /// Creutz ratio at the smallest loop, `chi(1,1) = -log<W(1,1)>`.
    chi_11: f64,
    /// Creutz ratio of the 2x2 block, `chi(2,2)`.
    chi_22: f64,
}

/// Drive a 6^3 Z2 gauge chain at inverse temperature `beta` and read the
/// plaquette and two Creutz ratios off the stream.
fn run(beta: f64, seed: u64) -> Measured {
    let toml = format!(
        "shape = [6, 6, 6]\n\
         j = 1.0\n\
         beta = {beta}\n\
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
        plaquette: reduce(&plaquette).mean,
        chi_11: creutz_ratio(&w11, &ones, &ones, &ones).value,
        chi_22: creutz_ratio(&w22, &w11, &w12, &w21).value,
    }
}

/// Confinement is strong at small `beta` and weakens as the coupling grows past
/// the transition: the mean plaquette rises toward one and the string tension
/// (Creutz ratio) falls.
#[test]
fn confinement_weakens_as_the_coupling_grows() {
    let strong = run(0.2, 20260728);
    let weak = run(1.0, 20260729);

    eprintln!(
        "strong (beta=0.2): plaq={:.4} chi11={:.4} chi22={:.4}",
        strong.plaquette, strong.chi_11, strong.chi_22
    );
    eprintln!(
        "weak   (beta=1.0): plaq={:.4} chi11={:.4} chi22={:.4}",
        weak.plaquette, weak.chi_11, weak.chi_22
    );

    // Leading strong-coupling result, pinned quantitatively: <plaquette> -> tanh(beta)
    // as beta -> 0, and at beta = 0.2 the correction is well under a percent.
    let tanh_strong = 0.2_f64.tanh();
    assert!(
        (strong.plaquette - tanh_strong).abs() < 0.02,
        "strong plaquette {} vs tanh(0.2) {}",
        strong.plaquette,
        tanh_strong
    );
    // Weak coupling sits deep in the ordered phase: the plaquette is near one.
    assert!(weak.plaquette > 0.9, "weak plaquette {}", weak.plaquette);

    // The string tension read off the smallest loop, chi(1,1) = -log<plaquette>,
    // is large in the confined phase and vanishes as the field orders.
    assert!(strong.chi_11 > 1.0, "strong chi(1,1) {}", strong.chi_11);
    assert!(weak.chi_11 < 0.1, "weak chi(1,1) {}", weak.chi_11);

    // The genuine 2x2 Creutz ratio (four real Wilson series) resolves to a small
    // value in the deconfined phase. At strong coupling the 2x2 loop average sits
    // below the noise of a short run, so creutz_ratio honestly returns NaN there
    // rather than a fabricated tension — the reason only the weak side is asserted.
    assert!(
        weak.chi_22.is_finite() && weak.chi_22 < 0.1,
        "weak chi(2,2) {}",
        weak.chi_22
    );
}
