//! Throughput comparison for the 3D Z2 gauge model: CPU random-site Metropolis,
//! the CPU link checkerboard, and the GPU link checkerboard.
//!
//! All three advance the *same* model and attempt `n_links` single-link updates
//! per sweep, so the fair metric is link-updates per second, independent of
//! lattice size. They differ only in the order those updates are visited in —
//! random, color order, or color order in parallel — which is exactly what the
//! comparison is about.
//!
//! A link update costs more than the Ising site update `bench.rs` measures, and
//! that is the whole reason the numbers here are lower: pricing a flip reads the
//! link's `2(D-1) = 4` staples of three links each, so twelve neighbor reads
//! against the Ising kernel's four.
//!
//! Timing isolates the sweeps: a warmup pass is run and discarded first (it pays
//! the GPU's one-time device and pipeline setup), then a fixed budget of sweeps
//! is timed. The GPU's `advance` only *submits* work, so a single `next()`
//! follows it to force a device sync before the clock stops.
//!
//! Extents are even throughout because the GPU schedule requires them — an odd
//! extent wraps two links of a shared plaquette into one color.
//!
//! Run it with (needs the sandbox disabled for GPU access):
//!
//! ```text
//! cargo run --release --example gauge_bench
//! ```

use std::time::Instant;

use plaquette::chain::Chain;
use plaquette::models::gauge::GpuGaugeChain;
use plaquette::models::gauge::Z2Gauge;
use plaquette::rng::RandRng;
use plaquette::{Cell, Configuration, Gpu, Lattice, LinkCheckerboard, Metropolis, Updater};

const BETA: f64 = 0.75;
const J: f64 = 1.0;

/// Roughly how many single-link updates to time per backend. The GPU needs far
/// more work than the CPU to run long enough to measure cleanly.
const CPU_UPDATE_BUDGET: usize = 8_000_000;
const GPU_UPDATE_BUDGET: usize = 300_000_000;

fn main() {
    let Some(_probe) = Gpu::new() else {
        eprintln!("no GPU adapter available (is the sandbox disabled?); nothing to compare");
        return;
    };

    println!(
        "3D Z2 gauge, beta = {BETA}, J = {J}   (Mlups = million link-updates/sec)\n\
         {:>4} {:>9} {:>11} {:>12} {:>12} {:>10} {:>11}",
        "L", "links", "CPU metro", "CPU cboard", "GPU cboard", "vs metro", "vs cboard"
    );

    for l in [4usize, 8, 12, 16, 24, 32, 48, 64] {
        let n_links = 3 * l * l * l;
        // Clamp so a run is neither too short to time nor pathologically long.
        let cpu_sweeps = (CPU_UPDATE_BUDGET / n_links).clamp(20, 2000);
        let gpu_sweeps = (GPU_UPDATE_BUDGET / n_links).clamp(100, 4000);

        let metro = cpu_mlups(l, cpu_sweeps, Metropolis);
        let cboard = cpu_mlups(l, cpu_sweeps, LinkCheckerboard);
        let gpu = gpu_mlups(l, gpu_sweeps);

        println!(
            "{l:>4} {n_links:>9} {metro:>11.1} {cboard:>12.1} {gpu:>12.1} \
             {:>9.1}x {:>10.1}x",
            gpu / metro,
            gpu / cboard
        );
        use std::io::Write;
        let _ = std::io::stdout().flush();
    }
}

/// Million link-updates per second for a CPU schedule.
///
/// Generic over the updater so the two CPU rows differ in nothing but the
/// schedule handed in — same lattice, same seed, same sweep count.
fn cpu_mlups(l: usize, sweeps: usize, updater: impl Updater<2, 3>) -> f64 {
    let lattice = Lattice::new([l, l, l]);
    let model = Z2Gauge::new(J);
    let mut rng = RandRng::seed_from_u64(1);
    let mut config = Configuration::<2>::hot(&lattice, Cell::Link, &mut rng);
    let mut chain = Chain::new(&mut config, &lattice, &model, &updater, BETA, &mut rng, 1);

    chain.advance(20); // warmup, discarded
    let t = Instant::now();
    chain.advance(sweeps);
    mlups(sweeps * 3 * l * l * l, t.elapsed().as_secs_f64())
}

/// Million link-updates per second for the GPU link checkerboard.
fn gpu_mlups(l: usize, sweeps: usize) -> f64 {
    let gpu = Gpu::new().expect("GPU adapter");
    let lattice = Lattice::new([l, l, l]);
    let mut rng = RandRng::seed_from_u64(2);
    let start = Configuration::<2>::hot(&lattice, Cell::Link, &mut rng);
    let mut chain = GpuGaugeChain::new(gpu, &lattice, J, BETA, 12345, &start, 1, 1);

    // Warmup pays the one-time device and pipeline cost; `next` forces a sync.
    chain.advance(20);
    let _ = chain.next();

    let t = Instant::now();
    chain.advance(sweeps);
    let _ = chain.next(); // one extra sweep, then blocks until the queue drains
    mlups((sweeps + 1) * 3 * l * l * l, t.elapsed().as_secs_f64())
}

fn mlups(updates: usize, secs: f64) -> f64 {
    updates as f64 / secs / 1.0e6
}
