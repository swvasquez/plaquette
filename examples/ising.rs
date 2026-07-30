//! Run the driver on the 2D Ising model from a config file and report a couple
//! of averaged observables.
//!
//! The smallest end-to-end use of the library: load a [`RunConfig`], hand it to
//! a [`Sampler`] (which assembles the pieces and thermalizes), then stream
//! samples from it, measuring each one and reducing the collected series.
//!
//! It collects the whole `Vec<Sample>` before reducing, because estimating
//! autocorrelation is a function of the *series* — folding each config into a
//! running sum and dropping it structurally cannot produce `tau_int`. A printed
//! `n_eff` far below `samples` is the signal that the error bars are
//! untrustworthy.
//!
//! The parameters live in a TOML file rather than in this source, so a different
//! run is an edit to the file rather than a recompile, and the file doubles as
//! the record of what was run.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example ising                      # uses examples/run.toml
//! cargo run --example ising -- path/to/run.toml  # or any other config
//! ```

use plaquette::config::RunConfig;
use plaquette::{
    Sample, Sampler, binder_cumulant, correlator, measure, reduce, specific_heat, susceptibility,
};

/// Where the run parameters come from when none is given on the command line.
const DEFAULT_CONFIG: &str = "examples/run.toml";

/// Whether to also report the two-point correlator.
///
/// A *consumer* choice rather than a run parameter, which is why it lives here
/// and not in the config file: the run produces configurations, and what gets
/// computed from them is decided afterwards. It costs an extra lattice scan per
/// sample, so it is off by default.
const SHOW_CORRELATOR: bool = false;

/// Marker appended to any quantity whose run was too short to measure its own
/// autocorrelation. When this shows up the error bar is understated, and the
/// remedy is more samples — not a different reduction.
fn warn(reliable: bool) -> &'static str {
    if reliable {
        ""
    } else {
        "  <-- TOO SHORT: error understated"
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_CONFIG.to_string());

    // A bad file fails here, before any sweeps are burned.
    let run = match RunConfig::load(&path) {
        Ok(run) => run,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("usage: cargo run --example ising -- [config.toml]");
            std::process::exit(1);
        }
    };

    // The sampler assembles the pieces and thermalizes, so the stream is at
    // equilibrium from the first config.
    let mut sampler = Sampler::new(&run);
    let chain = sampler.samples();

    // The geometry comes off the chain rather than a second lattice. These are
    // `'a` borrows, so they outlive the chain being moved into `.take` below.
    let (lattice, model) = (chain.lattice(), chain.action());
    let n_sites = lattice.n_sites() as f64;

    // One `Sample` per config is `O(n_samples)` memory, not `O(L²·n_samples)`.
    let mut samples: Vec<Sample> = Vec::with_capacity(run.n_samples);
    let mut sum_correlator: Option<Vec<f64>> = None;

    for config in chain.take(run.n_samples) {
        samples.push(measure(model, lattice, &config));

        if SHOW_CORRELATOR {
            let c = correlator(model, lattice, &config);
            let axis0 = &c.per_axis[0];
            let acc = sum_correlator.get_or_insert_with(|| vec![0.0; axis0.len()]);
            for (slot, &value) in acc.iter_mut().zip(axis0) {
                *slot += value;
            }
        }
    }

    // Primary reductions run on the physical densities so the printed means are
    // in the usual units (<E>/N and <|m|>); the fluctuation quantities take the
    // raw signed totals their formulas are written in.
    let energies: Vec<f64> = samples.iter().map(|s| s.energy).collect();
    let magnetizations: Vec<f64> = samples.iter().map(|s| s.magnetization).collect();
    let e_density: Vec<f64> = samples.iter().map(|s| s.energy / n_sites).collect();
    let abs_m: Vec<f64> = samples
        .iter()
        .map(|s| (s.magnetization / n_sites).abs())
        .collect();

    let energy = reduce(&e_density);
    let abs_mag = reduce(&abs_m);
    let heat = specific_heat(&energies, run.beta, n_sites);
    let chi = susceptibility(&magnetizations, run.beta, n_sites);
    let binder = binder_cumulant(&magnetizations);

    if let Some(description) = &run.description {
        println!("{description}");
    }
    println!(
        "2D Ising, shape = {:?}, beta = {}, J = {}, h = {}",
        run.shape, run.beta, run.j, run.h
    );
    println!(
        "  seed = {}, start = {:?}, thermalize = {}, sweeps_between = {}, samples = {}",
        run.seed, run.start, run.thermalize, run.sweeps_between, run.n_samples
    );
    println!(
        "  <E>/N   = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
        energy.mean,
        energy.stderr,
        energy.tau_int,
        energy.n_eff,
        warn(energy.is_reliable())
    );
    println!(
        "  <|m|>   = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
        abs_mag.mean,
        abs_mag.stderr,
        abs_mag.tau_int,
        abs_mag.n_eff,
        warn(abs_mag.is_reliable())
    );
    // `blocks` is to these what `n_eff` is to the means above: the count of
    // independent pieces the error bar rests on.
    println!(
        "  C       = {:.4} +/- {:.4}   (blocks = {}){}",
        heat.value,
        heat.stderr,
        heat.n_blocks,
        warn(heat.is_reliable())
    );
    println!(
        "  chi     = {:.4} +/- {:.4}   (blocks = {}){}",
        chi.value,
        chi.stderr,
        chi.n_blocks,
        warn(chi.is_reliable())
    );
    println!(
        "  U_4     = {:.4} +/- {:.4}   (blocks = {}){}",
        binder.value,
        binder.stderr,
        binder.n_blocks,
        warn(binder.is_reliable())
    );

    if let Some(acc) = sum_correlator {
        // C_r along axis 0, averaged over the chain. Raw (unconnected) — the
        // `- <s>^2` subtraction and the correlation-length fit are Statistics.
        let n = run.n_samples as f64;
        let values: Vec<String> = acc.iter().map(|&c| format!("{:.4}", c / n)).collect();
        println!("  C_r     = [{}]  (axis 0, raw)", values.join(", "));
    }
}
