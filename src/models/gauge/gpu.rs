//! GPU gauge backend: the model's side of the device seam, as a constructor
//! for the shared [`GpuChain`].
//!
//! Everything about running a sweep on the device lives in
//! [`device`](crate::device) and is shared by every model; what this module
//! owns is the part that is about the Z2 gauge model — the `Params` layout,
//! the staple table, and the `model.wgsl` snippet whose `energy_delta` is the
//! WGSL half of [`Action::energy_delta`](crate::action::Action::energy_delta).
//!
//! The variables live on links, so the shared chain assembles the *link*
//! schedule fragment: `2D` direction–parity colors, launched one thread per
//! site, exactly the walk the CPU checkerboard
//! [`LocalUpdate`](crate::updater::LocalUpdate) makes on a link field — which
//! is what lets that CPU sweep serve as the sequential reference. Randomness
//! is counter-based, keyed on `(seed, link, sweep)`. See `docs/metropolis.md`
//! for why the direction-and-parity coloring is the one a plaquette
//! interaction needs.

use crate::configuration::{Cell, Configuration};
use crate::device::{Gpu, GpuChain, GpuModelSetup, Kernel, fold_seed};
use crate::lattice::Lattice;
use crate::models::gauge::Z2Gauge;

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]`, already a 16-byte multiple, matching the WGSL `Params` struct
/// in `model.wgsl`. The head — `n_sites`, `seed`, `beta` — is the layout
/// contract with the shared kernel and schedule fragments; the tail is this
/// model's own.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
}

/// Build a Z2 gauge Markov chain on `gpu` over a copy of `start` — a
/// [`GpuChain`] on links, fixed at `Q = 2`.
///
/// `j` is the plaquette coupling and `beta` the inverse temperature; `seed`
/// keys the counter-based RNG; `sweeps_between` is the decorrelation stride,
/// `batch` how many samples are produced per device round-trip, and `kernel`
/// which single-variable rule a thread runs. Runs no sweeps — warmup is the
/// caller's job via [`GpuChain::advance`].
///
/// # Panics
///
/// Panics if `batch` is zero, if `start` is not a link field of this lattice,
/// if any extent is odd (see [`GpuChain`]), or if `D < 2`, where there is no
/// plaquette for the action to score.
#[allow(clippy::too_many_arguments)]
pub fn gpu_chain<const D: usize>(
    gpu: Gpu,
    lattice: &Lattice<D>,
    j: f64,
    beta: f64,
    seed: u64,
    start: &Configuration<2>,
    sweeps_between: usize,
    batch: usize,
    kernel: Kernel,
) -> GpuChain<2> {
    // Below two dimensions there is no direction pair, so the lattice has no
    // plaquettes and the action is identically zero. That is a sweep with
    // nothing to price rather than an error the arithmetic would raise, so it
    // has to be said here.
    assert!(
        D >= Z2Gauge::MIN_DIMENSION,
        "{}",
        Z2Gauge::TOO_FEW_DIMENSIONS
    );

    // The staple table verbatim: for each link, the `2 * (D - 1)` groups of
    // three link indices it is priced against, already flat and already in
    // group order, so the shader reads it with the same arithmetic
    // `link_staples` does.
    let n_links = lattice.n_links();
    let mut staples: Vec<u32> = Vec::with_capacity(n_links * Lattice::<D>::staple_stride());
    for link in 0..n_links {
        staples.extend(lattice.link_staples(link).iter().map(|&l| l as u32));
    }

    let params = Params {
        n_sites: lattice.n_sites() as u32,
        seed: fold_seed(seed),
        beta: beta as f32,
        j: j as f32,
    };

    GpuChain::build(
        gpu,
        lattice,
        GpuModelSetup {
            label: "gauge",
            source: include_str!("model.wgsl").to_string(),
            table: staples,
            params: bytemuck::bytes_of(&params).to_vec(),
            cell: Cell::Link,
        },
        kernel,
        start,
        sweeps_between,
        batch,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::require_gpu;
    use crate::models::gauge::Z2Gauge;
    use crate::models::gauge::gauge_measure;
    use crate::rng::RandRng;

    /// With `sweeps_between = 0` a "sample" runs no sweeps, so the round-trip
    /// upload → device buffer → read-back must return the start configuration
    /// unchanged. This exercises the buffer plumbing in isolation from the sweep.
    #[test]
    fn uploads_and_reads_back_unchanged() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        fn round_trip<const D: usize>(gpu: Gpu, shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(0);
            let start = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

            let mut chain = gpu_chain(gpu, &lat, 1.0, 0.5, 7, &start, 0, 1, Kernel::Metropolis);
            let got = chain.next().expect("open-ended stream yields");

            assert_eq!(
                got, start,
                "{shape:?}: zero-sweep round-trip must be the identity"
            );
        }

        // Two, three, and four dimensions: the staple stride is `6(D - 1)`
        // entries per link, so the buffer this uploads and the stride the kernel
        // indexes it with are a different width in each. A `D = 3` case alone
        // would not notice the two disagreeing, since three is what the shader
        // used to be written for.
        round_trip(gpu, [4, 4, 4]);
        let Some(gpu) = require_gpu() else { return };
        round_trip(gpu, [4, 4]);
        let Some(gpu) = require_gpu() else { return };
        round_trip(gpu, [4, 4, 4, 4]);
    }

    /// A cold start at strong coupling stays cold: every plaquette is already
    /// +1, so every flip is uphill by ΔE = +8 at `j = 1`, and at `beta = 4` the
    /// acceptance is about 1e-14. This is the cheapest check that the shader's
    /// staple arithmetic is right rather than merely self-consistent — a wrong
    /// staple table or a sign error would let flips through and drive the
    /// plaquette off 1.
    #[test]
    fn a_strongly_coupled_cold_start_stays_ordered() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([4, 4, 4]);
        let model = Z2Gauge::new(1.0);
        let start = Configuration::<2>::cold(&lat, Cell::Link);
        let n_plaquettes = lat.n_plaquettes() as f64;

        let mut chain = gpu_chain(gpu, &lat, 1.0, 4.0, 99, &start, 5, 4, Kernel::Metropolis);
        chain.advance(20);
        let mean = chain
            .take(8)
            .map(|c| gauge_measure(&model, &lat, &c).plaquette_sum / n_plaquettes)
            .sum::<f64>()
            / 8.0;

        assert!(mean > 0.99, "mean plaquette at beta = 4 was {mean}");
    }

    /// The GPU link checkerboard samples the same distribution as the CPU one:
    /// at a coupling well inside the confined phase the two agree on the mean
    /// plaquette. This is a distributional check — the CPU draws from a stream
    /// and the GPU from a keyed counter, so it is not bit-for-bit.
    #[test]
    fn matches_the_cpu_link_checkerboard_distribution() {
        let Some(gpu) = require_gpu() else {
            return;
        };

        let shape = [8, 8, 8];
        let (j, beta) = (1.0, 0.5);
        let (thermalize, sweeps_between, n) = (300, 4, 400);
        let model = Z2Gauge::new(j);
        let lat = Lattice::new(shape);
        let n_plaquettes = lat.n_plaquettes() as f64;

        let mean_plaquette = |samples: Vec<Configuration<2>>| -> f64 {
            let count = samples.len() as f64;
            samples
                .iter()
                .map(|c| gauge_measure(&model, &lat, c).plaquette_sum / n_plaquettes)
                .sum::<f64>()
                / count
        };

        // CPU reference: Chain driven by the checkerboard LocalUpdate.
        let cpu = {
            use crate::chain::Chain;
            use crate::updater::{LocalUpdate, Schedule};
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
            let updater = LocalUpdate::new(Kernel::Metropolis, Schedule::Checkerboard);
            let mut chain = Chain::new(
                &mut cfg,
                &lat,
                &model,
                &updater,
                beta,
                &mut rng,
                sweeps_between,
            );
            chain.advance(thermalize);
            mean_plaquette(chain.take(n).collect())
        };

        // GPU: the shared chain over the same model and geometry.
        let gpu_mean = {
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
            let mut chain = gpu_chain(
                gpu,
                &lat,
                j,
                beta,
                12345,
                &start,
                sweeps_between,
                64,
                Kernel::Metropolis,
            );
            chain.advance(thermalize);
            mean_plaquette(chain.take(n).collect())
        };

        eprintln!("mean plaquette at beta = {beta}: cpu {cpu:.4}, gpu {gpu_mean:.4}");
        // Both sides must land on the physics, not merely on each other: at this
        // coupling the plaquette sits near 0.5, so a shader that froze the field
        // or randomized it would fail here even if the CPU somehow agreed.
        assert!(
            (0.3..0.7).contains(&cpu) && (0.3..0.7).contains(&gpu_mean),
            "mean plaquette left the confined-phase range: cpu {cpu:.4}, gpu {gpu_mean:.4}"
        );
        assert!(
            (cpu - gpu_mean).abs() < 0.02,
            "mean plaquette mismatch: cpu {cpu:.4} vs gpu {gpu_mean:.4}"
        );
    }

    /// An odd extent is rejected rather than silently sampling the wrong
    /// distribution: the periodic wrap would put two links of a shared plaquette
    /// in one color, and a parallel pass would then update both against each
    /// other's stale values.
    #[test]
    #[should_panic(expected = "even lattice extents")]
    fn rejects_odd_extents() {
        let Some(gpu) = require_gpu() else {
            // The assertion under test fires before any device work, but the
            // constructor needs a device to reach it, so a machine without one
            // cannot run this. Panic to keep `should_panic` satisfied.
            panic!("no GPU adapter: GPU link checkerboard needs even lattice extents (skipped)");
        };
        let lat = Lattice::new([4, 3, 4]);
        let start = Configuration::<2>::cold(&lat, Cell::Link);
        let _ = gpu_chain(gpu, &lat, 1.0, 0.5, 1, &start, 1, 1, Kernel::Metropolis);
    }
}
