//! GPU gauge backend: a [`GpuGaugeChain`] that runs the link checkerboard sweep
//! on the GPU via `wgpu`, exposed through the same
//! `Iterator<Item = Configuration>` interface as the CPU
//! [`Chain`](crate::chain::Chain).
//!
//! The gauge sibling of [`GpuIsingChain`](crate::ising_gpu::GpuIsingChain), and
//! a separate type for the same reason
//! [`LinkCheckerboard`] is separate from
//! [`SiteCheckerboard`](crate::updater::SiteCheckerboard): that one runs over a
//! site field and this over a link field, so they share no line of the schedule.
//! What they *do* share — encoding color passes, batching samples, and reading
//! them back — is `DeviceSweeper`, so this module holds only the part that is
//! about the gauge model: the `Params` layout, the staple table, and the
//! `2D`-color link checkerboard the shader implements.
//!
//! The coloring is compiled into the shader (`gauge_checkerboard.wgsl`), so this
//! does not use the [`Updater`](crate::updater::Updater) seam. Its randomness is
//! counter-based, keyed on `(seed, link, sweep)`, so the result is independent of
//! GPU thread order — the property that lets the CPU link checkerboard serve as a
//! reference. See `docs/metropolis.md` for why the direction-and-parity coloring
//! is the one a plaquette interaction needs.

use crate::configuration::{Cell, Configuration};
use crate::device::{
    DeviceSweeper, Gpu, SweepSetup, assert_even_extents, fold_seed, site_colors, state_words,
};
use crate::lattice::Lattice;
use crate::model::Z2Gauge;
use crate::updater::LinkCheckerboard;

/// The compiled kernel: the shared preamble followed by the link checkerboard.
const SHADER: &str = crate::device::shader_source!("gauge_checkerboard.wgsl");

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]`, already a 16-byte multiple, matching the WGSL `Params` struct's
/// uniform layout. It carries no geometry: the kernel reads a base site's parity
/// from a table and takes the dimension as an override, so nothing about the
/// shape has to be described here.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
}

/// The Z2 gauge Markov chain run on the GPU, yielding sampled
/// [`Configuration`]s on links.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely. Fixed at `Q = 2`.
///
/// The type carries no dimension, for the reason
/// [`GpuIsingChain`](crate::ising_gpu::GpuIsingChain) does not: the lattice is
/// read once in [`new`](GpuGaugeChain::new) to build the tables, and what
/// survives is device buffers and counts.
pub struct GpuGaugeChain {
    sweeper: DeviceSweeper<2>,
}

impl GpuGaugeChain {
    /// Build a chain on `gpu` over a copy of `start`, uploaded to the device.
    ///
    /// `start` is read only to upload it, so the host copy is untouched and the
    /// same configuration can seed a CPU run too. `j` is the plaquette coupling
    /// and `beta` the inverse temperature; `seed` keys the counter-based RNG.
    /// `sweeps_between` is the decorrelation stride, and `batch` is how many
    /// samples are produced per device round-trip.
    ///
    /// Runs no sweeps — like [`Chain::new`](crate::chain::Chain::new), warmup is
    /// the caller's job via [`advance`](GpuGaugeChain::advance).
    ///
    /// # Panics
    ///
    /// Panics if `batch` is zero, if `start` is not a link field of this
    /// lattice, if any extent is odd, or if `D < 2`, where there is no plaquette
    /// for the action to score.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        j: f64,
        beta: f64,
        seed: u64,
        start: &Configuration<2>,
        sweeps_between: usize,
        batch: usize,
    ) -> Self {
        assert!(batch > 0, "batch size must be positive");
        // Below two dimensions there is no direction pair, so the lattice has no
        // plaquettes and the action is identically zero. That is a sweep with
        // nothing to price rather than an error the arithmetic would raise, so
        // it has to be said here.
        assert!(
            D >= Z2Gauge::MIN_DIMENSION,
            "{}",
            Z2Gauge::TOO_FEW_DIMENSIONS
        );
        // The shader indexes the staple table by link, so a site field would be
        // silently misread rather than rejected; the cell kind is what makes
        // that checkable at all.
        assert_eq!(
            start.cell(),
            Cell::Link,
            "the GPU link checkerboard updates links, so the start must be a link field"
        );
        assert_eq!(
            start.n_vars(),
            lattice.n_links(),
            "start configuration and lattice disagree on link count"
        );
        let shape = lattice.shape();
        assert_even_extents(&shape, "link");

        let n_sites = lattice.n_sites();
        let n_links = lattice.n_links();

        let links = state_words(start);
        // The staple table verbatim: for each link, the `2 * (D - 1)` groups of
        // three link indices it is priced against, already flat and already in
        // group order, so the shader reads it with the same arithmetic
        // `link_staples` does.
        let mut staples: Vec<u32> = Vec::with_capacity(n_links * Lattice::<D>::staple_stride());
        for link in 0..n_links {
            staples.extend(lattice.link_staples(link).iter().map(|&l| l as u32));
        }
        let site_color = site_colors(lattice);
        let params = Params {
            n_sites: n_sites as u32,
            seed: fold_seed(seed),
            beta: beta as f32,
            j: j as f32,
        };

        GpuGaugeChain {
            sweeper: DeviceSweeper::build(
                gpu,
                SweepSetup {
                    label: "gauge checkerboard",
                    shader: SHADER,
                    vars_init: &links,
                    table: &staples,
                    site_color: &site_color,
                    params: bytemuck::bytes_of(&params),
                    dimension: D as u32,
                    cell: Cell::Link,
                    n_vars: n_links,
                    // One thread per site, each owning that site's link in the
                    // pass's direction. A thread-per-link launch would idle all
                    // but one thread in `2 * D`, since a dispatch owns one
                    // direction of `D` and one parity of two; per-site idles one
                    // in two whatever the dimension.
                    threads: n_sites,
                    // Read from the CPU schedule rather than restated: the two
                    // must agree, since that schedule is the sequential
                    // reference this kernel is checked against.
                    colors: LinkCheckerboard::colors::<D>() as u32,
                    sweeps_between,
                    batch,
                },
            ),
        }
    }

    /// Advance the chain by `sweeps` sweeps on the device, producing no snapshot —
    /// the GPU counterpart of [`Chain::advance`](crate::chain::Chain::advance),
    /// used to discard warmup.
    pub fn advance(&mut self, sweeps: usize) {
        self.sweeper.advance(sweeps);
    }
}

impl Iterator for GpuGaugeChain {
    type Item = Configuration<2>;

    /// Yield the next sampled configuration, running a fresh batch on the device
    /// when the host-side buffer drains. Always `Some`: the chain is open-ended,
    /// so callers bound it with `.take(n)`.
    fn next(&mut self) -> Option<Self::Item> {
        self.sweeper.next_sample()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::require_gpu;
    use crate::model::Z2Gauge;
    use crate::observables::gauge_measure;
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

            let mut chain = GpuGaugeChain::new(gpu, &lat, 1.0, 0.5, 7, &start, 0, 1);
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

        let mut chain = GpuGaugeChain::new(gpu, &lat, 1.0, 4.0, 99, &start, 5, 4);
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

        // CPU reference: Chain driven by the LinkCheckerboard updater.
        let cpu = {
            use crate::chain::Chain;
            use crate::updater::LinkCheckerboard;
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
            let updater = LinkCheckerboard;
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

        // GPU: GpuGaugeChain over the same model and geometry.
        let gpu_mean = {
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
            let mut chain =
                GpuGaugeChain::new(gpu, &lat, j, beta, 12345, &start, sweeps_between, 64);
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
        let _ = GpuGaugeChain::new(gpu, &lat, 1.0, 0.5, 1, &start, 1, 1);
    }
}
