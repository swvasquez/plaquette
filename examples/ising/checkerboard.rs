//! Compare the Metropolis and checkerboard updaters on identical physics.
//!
//! On CPU the checkerboard schedule is just single-spin Metropolis visited in a
//! fixed color order rather than at random, so it samples the *same* Boltzmann
//! distribution. This example runs both on one `(shape, beta, J)` point and
//! prints their averaged observables side by side: `<E>/N` and `<|m|>` should
//! agree within their error bars. Its worth is entirely as a check — the
//! checkerboard exists to be the sequential twin of a parallel (GPU) sweep, and
//! agreement here is what says the twin is faithful.
//!
//! The two runs differ only in the `updater` field, which is the whole point of
//! selecting the algorithm from config: a value read at runtime picks the schedule
//! without either updater's type leaking into the driver.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example checkerboard
//! ```

use plaquette::config::{Start, UpdaterKind};
use plaquette::ising_config::IsingRunConfig;
use plaquette::{Estimate, IsingSampler, Sample, measure, reduce};

/// Thermalize, stream `n_samples`, and reduce the energy density and `|m|`.
///
/// Returns the two estimates the comparison prints. The generic details (which
/// updater, how long) all come from `run`, so both algorithms share this path.
fn measure_run(run: &IsingRunConfig) -> (Estimate, Estimate) {
    let mut sampler = IsingSampler::new(run);
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_sites = lattice.n_sites() as f64;

    let samples: Vec<Sample> = sampler
        .samples()
        .take(run.n_samples)
        .map(|c| measure(&model, &lattice, &c))
        .collect();

    let e_density: Vec<f64> = samples.iter().map(|s| s.energy / n_sites).collect();
    let abs_m: Vec<f64> = samples
        .iter()
        .map(|s| (s.magnetization / n_sites).abs())
        .collect();

    (reduce(&e_density), reduce(&abs_m))
}

/// One labeled row of the comparison table.
fn row(label: &str, energy: &Estimate, abs_mag: &Estimate) {
    println!(
        "  {label:<12}  <E>/N = {:+.4} +/- {:.4}    <|m|> = {:.4} +/- {:.4}",
        energy.mean, energy.stderr, abs_mag.mean, abs_mag.stderr
    );
}

fn main() {
    // One physics point; the two runs differ only in the updater below. Both draw
    // from the same seed — the schedules consume randomness differently, so the
    // sample streams are not identical, but the distributions they sample are.
    let base = IsingRunConfig {
        shape: [16, 16],
        j: 1.0,
        h: 0.0,
        beta: 0.44, // beta_c = ln(1 + sqrt(2)) / 2 ~ 0.4407
        updater: UpdaterKind::Metropolis,
        thermalize: 1000,
        sweeps_between: 5,
        n_samples: 2000,
        seed: 20260724,
        start: Start::Hot,
        description: None,
    };

    let metropolis = IsingRunConfig {
        updater: UpdaterKind::Metropolis,
        ..base.clone()
    };
    let checkerboard = IsingRunConfig {
        updater: UpdaterKind::SiteCheckerboard,
        ..base.clone()
    };

    let (e_m, m_m) = measure_run(&metropolis);
    let (e_c, m_c) = measure_run(&checkerboard);

    println!(
        "2D Ising, shape = {:?}, beta = {}, J = {}   ({} samples, sweeps_between = {})",
        base.shape, base.beta, base.j, base.n_samples, base.sweeps_between
    );
    row("metropolis", &e_m, &m_m);
    row("checkerboard", &e_c, &m_c);
    println!("  (both sample the same distribution — the rows should agree within error bars)");
}
