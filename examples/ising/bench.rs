//! Throughput comparison: CPU single-spin Metropolis vs the GPU checkerboard.
//!
//! Both advance the *same* 2D Ising model and do `n_sites` single-site updates per
//! sweep, so the fair metric is site-updates per second (independent of lattice
//! size). They differ in schedule — CPU picks random sites, the GPU walks the two
//! colors in parallel — but that is exactly the point of the comparison.
//!
//! Timing isolates the sweeps: a warmup pass is run and discarded first (it pays
//! the GPU's one-time device/pipeline setup), then a fixed budget of sweeps is
//! timed. The GPU's `advance` only *submits* work, so a single `next()` follows
//! it to force a device sync before the clock stops.
//!
//! Run it with (needs the sandbox disabled for GPU access):
//!
//! ```text
//! cargo run --release --example bench
//! ```

use std::time::Instant;

use plaquette::chain::Chain;
use plaquette::model::Ising;
use plaquette::rng::RandRng;
use plaquette::{Cell, Configuration, Gpu, GpuChain, Lattice, Metropolis};

const BETA: f64 = 0.44;
const J: f64 = 1.0;
const H: f64 = 0.0;

/// Roughly how many single-site updates to time per backend. The GPU needs far
/// more work than the CPU to run long enough to measure cleanly.
const CPU_UPDATE_BUDGET: usize = 20_000_000;
const GPU_UPDATE_BUDGET: usize = 400_000_000;

fn main() {
    let Some(_probe) = Gpu::new() else {
        eprintln!("no GPU adapter available (is the sandbox disabled?); nothing to compare");
        return;
    };

    println!(
        "2D Ising, beta = {BETA}, J = {J}   (Mups = million site-updates/sec)\n\
         {:>6} {:>9} {:>12} {:>12} {:>9}",
        "L", "sites", "CPU Mups", "GPU Mups", "speedup"
    );

    for l in [16usize, 32, 64, 128, 256, 512] {
        let n_sites = l * l;
        // Clamp so a run is neither too short to time nor pathologically long.
        let cpu = cpu_mups(l, (CPU_UPDATE_BUDGET / n_sites).clamp(50, 3000));
        let gpu = gpu_mups(l, (GPU_UPDATE_BUDGET / n_sites).clamp(200, 5000));
        println!(
            "{l:>6} {n_sites:>9} {cpu:>12.1} {gpu:>12.1} {:>8.1}x",
            gpu / cpu
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Million site-updates per second for CPU random-site Metropolis.
fn cpu_mups(l: usize, sweeps: usize) -> f64 {
    let lattice = Lattice::new([l, l]);
    let model = Ising::new(J, H);
    let mut rng = RandRng::seed_from_u64(1);
    let mut config = Configuration::<2>::hot(&lattice, Cell::Site, &mut rng);
    let updater = Metropolis;
    let mut chain = Chain::new(&mut config, &lattice, &model, &updater, BETA, &mut rng, 1);

    chain.advance(50); // warmup, discarded
    let t = Instant::now();
    chain.advance(sweeps);
    mups(sweeps * l * l, t.elapsed().as_secs_f64())
}

/// Million site-updates per second for the GPU checkerboard.
fn gpu_mups(l: usize, sweeps: usize) -> f64 {
    let gpu = Gpu::new().expect("GPU adapter");
    let lattice = Lattice::new([l, l]);
    let mut rng = RandRng::seed_from_u64(2);
    let start = Configuration::<2>::hot(&lattice, Cell::Site, &mut rng);
    let mut chain = GpuChain::new(gpu, &lattice, J, H, BETA, 12345, &start, 1, 1);

    // Warmup pays the one-time device/pipeline cost; `next` forces a sync.
    chain.advance(50);
    let _ = chain.next();

    let t = Instant::now();
    chain.advance(sweeps);
    let _ = chain.next(); // one extra sweep, then blocks until the queue drains
    mups((sweeps + 1) * l * l, t.elapsed().as_secs_f64())
}

fn mups(updates: usize, secs: f64) -> f64 {
    updates as f64 / secs / 1.0e6
}
