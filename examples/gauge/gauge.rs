//! Run the driver on the 3D Z2 gauge theory from a config file and report a
//! couple of averaged observables.
//!
//! The gauge counterpart of the `ising` example, and the same shape end to end:
//! load a [`GaugeRunConfig`], hand it to a [`GaugeSampler`] (which assembles the
//! pieces and thermalizes), then stream samples from it, measuring each one and
//! reducing the collected series. What differs is what there is to measure. A
//! gauge theory has no magnetization — flipping every link that touches one site
//! is a symmetry, so any average of a single link variable is exactly zero
//! however long the chain runs — and the question the run is asking is answered
//! by closed loops instead: whether `W(r, t)` falls off with the enclosed area,
//! which means the theory confines, or merely with the perimeter, which means it
//! does not.
//!
//! In three dimensions that changes over at `beta_c ~ 0.7613`, so the shipped
//! config sits just below it, in the confined phase near the transition. The
//! loop table on its own does not say how strong that area falloff is, so the run
//! also reports the Creutz ratio `chi(r, t)` beside it — the 2x2 combination of
//! adjacent loops in which the perimeter and constant parts of `-log W` cancel,
//! leaving the area coefficient alone. That is a function of four averages over
//! the series rather than of any one configuration, which is why it comes from
//! `statistics` and not from a measurement.
//!
//! Reading a string tension off that column is still the reader's call. `chi`
//! only settles onto it once the loops are big enough for the short-distance
//! corrections to die out, and `MAX_WILSON_SIDE` below keeps them small, so what
//! prints is the start of that approach rather than a converged number.
//!
//! It collects the whole series before reducing, for the reason the `ising`
//! example does: estimating autocorrelation is a function of the *series*, and
//! folding each config into a running sum structurally cannot produce `tau_int`.
//! A printed `n_eff` far below `samples` is the signal that the error bars are
//! untrustworthy.
//!
//! Run it with:
//!
//! ```text
//! cargo run --release --example gauge                        # uses examples/gauge/gauge.toml
//! cargo run --release --example gauge -- path/to/gauge.toml  # or any other config
//! ```
//!
//! `--release` is worth it here: a Wilson table costs the rectangle count times
//! the lattice volume times the perimeter, per sample, which is a good deal more
//! work than the Ising example's two lattice scans.

use plaquette::gauge_config::GaugeRunConfig;
use plaquette::{
    Estimate, GaugeSample, GaugeSampler, creutz_ratio, gauge_measure, polyakov_loop, reduce,
    specific_heat, wilson_rectangles,
};

/// Where the run parameters come from when none is given on the command line.
const DEFAULT_CONFIG: &str = "examples/gauge/gauge.toml";

/// Largest Wilson rectangle side to report, capped internally at half the
/// smallest extent — past that a rectangle wraps the torus and stops measuring
/// what it is meant to.
///
/// A *consumer* choice rather than a run parameter, which is why it lives here
/// and not in the config file: the run produces configurations, and what gets
/// computed from them is decided afterwards.
const MAX_WILSON_SIDE: usize = 2;

/// Whether to also report the Polyakov loop along the last axis. Off by default:
/// it costs another walk of the lattice per sample, and it is the observable for
/// a *finite-temperature* run — one short axis read as time — rather than for the
/// symmetric box the shipped config uses.
const SHOW_POLYAKOV: bool = false;

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

    // A bad file fails here, before any sweeps are burned — including a config
    // asking for an updater that cannot drive a link field.
    let run = match GaugeRunConfig::load(&path) {
        Ok(run) => run,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("usage: cargo run --release --example gauge -- [config.toml]");
            std::process::exit(1);
        }
    };

    // The sampler assembles the pieces and thermalizes, so the stream is at
    // equilibrium from the first config.
    let mut sampler = GaugeSampler::new(&run);

    // Geometry for measurement comes off the sampler as owned values, read once
    // before streaming so the stream can borrow the sampler by itself.
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_plaquettes = lattice.n_plaquettes() as f64;
    let time_axis = lattice.shape().len() - 1;

    // One `GaugeSample` per config, plus one Wilson series per rectangle size.
    // The table's own dimensions decide how many series there are, since it caps
    // the requested side at half the smallest extent.
    let mut samples: Vec<GaugeSample> = Vec::with_capacity(run.n_samples);
    let mut wilson: Vec<Vec<f64>> = Vec::new();
    let mut sides = 0;
    let mut polyakov: Vec<f64> = Vec::new();

    for config in sampler.samples().take(run.n_samples) {
        samples.push(gauge_measure(&model, &lattice, &config));

        let table = wilson_rectangles(&model, &lattice, &config, MAX_WILSON_SIDE).per_size;
        if wilson.is_empty() {
            sides = table.len();
            wilson = vec![Vec::with_capacity(run.n_samples); sides * sides];
        }
        for (r, row) in table.iter().enumerate() {
            for (t, &w) in row.iter().enumerate() {
                wilson[r * sides + t].push(w);
            }
        }

        if SHOW_POLYAKOV {
            polyakov.push(polyakov_loop(&model, &lattice, &config, time_axis));
        }
    }

    // The mean plaquette is the energy per plaquette with the coupling divided
    // out, which is the form the literature reports and the one that survives
    // `j = 0`. The specific heat takes the raw total energies its formula is
    // written in, normalized by the plaquette count — in three dimensions there
    // are as many plaquettes as links, so that is also the per-degree-of-freedom
    // reading.
    let energies: Vec<f64> = samples.iter().map(|s| s.energy).collect();
    let mean_plaquette: Vec<f64> = samples
        .iter()
        .map(|s| s.plaquette_sum / n_plaquettes)
        .collect();

    let plaquette = reduce(&mean_plaquette);
    let heat = specific_heat(&energies, run.beta, n_plaquettes);

    if let Some(description) = &run.description {
        println!("{description}");
    }
    println!(
        "3D Z2 gauge, shape = {:?}, beta = {}, J = {}",
        run.shape, run.beta, run.j
    );
    println!(
        "  seed = {}, start = {:?}, thermalize = {}, sweeps_between = {}, samples = {}",
        run.seed, run.start, run.thermalize, run.sweeps_between, run.n_samples
    );
    println!(
        "  <plaq>  = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
        plaquette.mean,
        plaquette.stderr,
        plaquette.tau_int,
        plaquette.n_eff,
        warn(plaquette.is_reliable())
    );
    // `blocks` is to this what `n_eff` is to the mean above: the count of
    // independent pieces the error bar rests on.
    println!(
        "  C       = {:.4} +/- {:.4}   (blocks = {}){}",
        heat.value,
        heat.stderr,
        heat.n_blocks,
        warn(heat.is_reliable())
    );

    // Only `r <= t` is printed: the table is symmetric by construction, since
    // both assignments of the sides to a plane's two directions land in the same
    // entry. Row and column zero are the trivial `1.0` anchor and say nothing.
    for r in 1..sides {
        for t in r..sides {
            let w: Estimate = reduce(&wilson[r * sides + t]);
            println!(
                "  W({r},{t})  = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
                w.mean,
                w.stderr,
                w.tau_int,
                w.n_eff,
                warn(w.is_reliable())
            );
        }
    }

    // The same sizes again, now as Creutz ratios. Each one needs the three
    // smaller neighbours of `(r, t)`, and row and column zero being the trivial
    // `1.0` anchor is exactly what makes `chi(1,1)` fall back to `-log <W(1,1)>`.
    for r in 1..sides {
        for t in r..sides {
            let chi = creutz_ratio(
                &wilson[r * sides + t],
                &wilson[(r - 1) * sides + (t - 1)],
                &wilson[(r - 1) * sides + t],
                &wilson[r * sides + (t - 1)],
            );
            // A `NaN` is the estimator declining to take a logarithm of a
            // non-positive ratio, which at this size means the loops are lost in
            // the noise rather than that anything went wrong.
            if chi.value.is_nan() {
                println!("  chi({r},{t})= unresolved (loop ratio not positive)");
            } else {
                println!(
                    "  chi({r},{t})= {:.4} +/- {:.4}   (blocks = {}){}",
                    chi.value,
                    chi.stderr,
                    chi.n_blocks,
                    warn(chi.is_reliable())
                );
            }
        }
    }

    if SHOW_POLYAKOV {
        // Kept signed, so this sits near zero in the confined phase by symmetry
        // rather than by accident; it is the magnitude that orders.
        let loop_ = reduce(&polyakov);
        println!(
            "  P       = {:.4} +/- {:.4}   (axis {time_axis}, signed){}",
            loop_.mean,
            loop_.stderr,
            warn(loop_.is_reliable())
        );
    }
}
