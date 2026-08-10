//! Run the driver on the `q`-state Potts model from a config file and report a
//! few averaged observables.
//!
//! The Potts counterpart of `examples/ising/ising.rs`, and the same shape: load a
//! [`PottsRunConfig`], hand it to a [`PottsSampler`] (which assembles the pieces
//! and thermalizes), then stream samples from it, measuring each one and reducing
//! the collected series.
//!
//! Two run parameters are named here rather than in the config, because both are
//! compile-time constants: the number of states `Q` and the lattice dimension
//! `D`. Running a different pair means editing the two lines below and
//! rebuilding; everything else — extents, coupling, temperature, schedule — still
//! comes from the TOML. The library is generic over both, so nothing but the
//! constants changes. A config whose `shape` names a different number of axes is
//! reported at load by `check_dimension`; nothing in the file mentions `Q`, so
//! there is nothing to check it against.
//!
//! It collects the whole `Vec<PottsSample>` before reducing, because estimating
//! autocorrelation is a function of the *series* — folding each config into a
//! running sum and dropping it structurally cannot produce `tau_int`. A printed
//! `n_eff` far below `samples` is the signal that the error bars are
//! untrustworthy.
//!
//! Run it with:
//!
//! ```text
//! cargo run --example potts                      # uses examples/potts/potts.toml
//! cargo run --example potts -- path/to/run.toml  # or any other config
//! ```

use plaquette::potts_config::{POTTS_D, POTTS_Q, PottsRunConfig};
use plaquette::{
    PottsSample, PottsSampler, binder_cumulant, potts_correlator, potts_measure, reduce,
    specific_heat, susceptibility,
};

/// The number of states this program is built for.
///
/// Edit and rebuild to run a different one — `3` is what the shipped config's
/// critical coupling is quoted for, `2` is the Ising model at half the coupling,
/// and above `4` the two-dimensional transition turns first order.
const Q: usize = POTTS_Q;

/// The lattice dimension this program is built for.
///
/// Edit and rebuild to run a different one — `2` for the square lattice the
/// shipped config uses, where the critical coupling is known exactly. The
/// config's `shape` must name this many axes.
const D: usize = POTTS_D;

/// Where the run parameters come from when none is given on the command line.
const DEFAULT_CONFIG: &str = "examples/potts/potts.toml";

/// Whether to also report the two-point function.
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
    let run = match PottsRunConfig::load(&path) {
        Ok(run) => run,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("usage: cargo run --example potts -- [config.toml]");
            std::process::exit(1);
        }
    };

    // The file has to be for the dimension this program was built for. Checked
    // here so a mismatch is a message rather than a panic inside the sampler.
    if let Err(e) = run.check_dimension::<D>() {
        eprintln!("error: {e}");
        eprintln!("edit `const D` in examples/potts/potts.rs and rebuild to run it");
        std::process::exit(1);
    }

    // The sampler assembles the pieces and thermalizes, so the stream is at
    // equilibrium from the first config.
    let mut sampler = PottsSampler::<Q, D>::new(&run);

    // Geometry for measurement comes off the sampler as owned values, read once
    // before streaming so the stream can borrow the sampler by itself.
    let lattice = sampler.lattice();
    let model = sampler.model();
    let n_sites = lattice.n_sites() as f64;

    let mut samples: Vec<PottsSample> = Vec::with_capacity(run.n_samples);
    let mut sum_correlator: Option<Vec<f64>> = None;

    for config in sampler.samples().take(run.n_samples) {
        samples.push(potts_measure(&model, &lattice, &config));

        if SHOW_CORRELATOR {
            let c = potts_correlator(&model, &lattice, &config);
            let axis0 = &c.per_axis[0];
            let acc = sum_correlator.get_or_insert_with(|| vec![0.0; axis0.len()]);
            for (slot, &value) in acc.iter_mut().zip(axis0) {
                *slot += value;
            }
        }
    }

    // Primary reductions run on the physical densities so the printed means are
    // in the usual units. The order parameter is already a density, unlike the
    // Ising magnetization, so the two series need different handling on the way
    // into the fluctuation quantities below.
    let energies: Vec<f64> = samples.iter().map(|s| s.energy).collect();
    let e_density: Vec<f64> = samples.iter().map(|s| s.energy / n_sites).collect();
    let order: Vec<f64> = samples.iter().map(|s| s.order).collect();
    let simplex: Vec<f64> = samples.iter().map(|s| s.simplex_order).collect();
    // `susceptibility` is written for a signed *total* and divides by `n_sites`
    // itself, so the order parameter is scaled up to meet it rather than the
    // formula being rewritten here. `binder_cumulant` is a ratio of moments and
    // scale-invariant, so it takes the series as it stands.
    let order_total: Vec<f64> = order.iter().map(|m| m * n_sites).collect();

    let energy = reduce(&e_density);
    let order_estimate = reduce(&order);
    let simplex_estimate = reduce(&simplex);
    let heat = specific_heat(&energies, run.beta, n_sites);
    let chi = susceptibility(&order_total, run.beta, n_sites);
    let binder = binder_cumulant(&order);

    if let Some(description) = &run.description {
        println!("{description}");
    }
    // The offsets are printed only when set, so a symmetric run — which is what
    // the critical coupling is quoted for — does not carry an empty list around.
    let offsets = if run.h.is_empty() {
        String::new()
    } else {
        format!(", h = {:?}", run.h)
    };
    println!(
        "Potts with q = {Q} in {D}D, shape = {:?}, beta = {}, J = {}{offsets}",
        run.shape, run.beta, run.j
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
    // Both conventions, since a comparison against a published curve needs
    // whichever that curve plotted — see docs/potts.md. `m` reads the most
    // populated label, `m_v` the vector (simplex) form.
    println!(
        "  <m>     = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
        order_estimate.mean,
        order_estimate.stderr,
        order_estimate.tau_int,
        order_estimate.n_eff,
        warn(order_estimate.is_reliable())
    );
    println!(
        "  <m_v>   = {:.4} +/- {:.4}   (tau_int = {:.2}, n_eff = {:.0}){}",
        simplex_estimate.mean,
        simplex_estimate.stderr,
        simplex_estimate.tau_int,
        simplex_estimate.n_eff,
        warn(simplex_estimate.is_reliable())
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
    // Reported in the scalar Ising normalization, whose disordered anchor of
    // zero does not carry over to a non-negative order parameter. What survives
    // is the crossing across lattice sizes; see docs/potts.md.
    println!(
        "  U_4     = {:.4} +/- {:.4}   (blocks = {}, Ising normalization){}",
        binder.value,
        binder.stderr,
        binder.n_blocks,
        warn(binder.is_reliable())
    );

    if let Some(acc) = sum_correlator {
        // C_r along axis 0, averaged over the chain. Already connected — the
        // model subtracts the 1/q two independent labels agree at — so unlike
        // the Ising correlator nothing is left to take off.
        let n = run.n_samples as f64;
        let values: Vec<String> = acc.iter().map(|&c| format!("{:.4}", c / n)).collect();
        println!("  C_r     = [{}]  (axis 0, connected)", values.join(", "));
    }
}
