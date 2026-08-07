//! GPU Ising backend: a [`GpuIsingChain`] that runs the site checkerboard sweep
//! on the GPU via `wgpu`, exposed through the same
//! `Iterator<Item = Configuration>` interface as the CPU
//! [`Chain`](crate::chain::Chain).
//!
//! `GpuIsingChain` is a *sibling* of `Chain`, not a variant of it: they share
//! only the iterator interface. Where `Chain` borrows a host `Configuration` and
//! mutates it a sweep at a time, this owns its device resources and keeps the
//! configuration in a device buffer, crossing the host boundary once per batch.
//! All of that batching lives in `DeviceSweeper`, shared with the gauge
//! backend; what this module owns is the part that is about the Ising model —
//! the `Params` layout, the neighbor table, and the two-color site
//! checkerboard the shader implements.
//!
//! The coloring is compiled into the shader (`ising_checkerboard.wgsl`), so this
//! does not use the [`Updater`](crate::updater::Updater) seam. Its randomness is
//! counter-based, keyed on `(seed, site, sweep)`, so the result is independent of
//! GPU thread order — the property that lets the CPU site checkerboard serve as a
//! reference.

use crate::configuration::{Cell, Configuration};
use crate::device::{DeviceSweeper, Gpu, SweepSetup, assert_even_extents, fold_seed};
use crate::lattice::Lattice;

/// The dimension the Ising shader is written for, matching
/// [`IsingRunConfig`](crate::ising_config::IsingRunConfig).
const D: usize = 2;

/// Color passes per sweep: the two coordinate-sum parities.
const N_COLORS: u32 = 2;

/// The compiled kernel: the shared preamble followed by the site checkerboard.
const SHADER: &str = crate::device::shader_source!("ising_checkerboard.wgsl");

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct's uniform layout.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    width: u32,
    seed: u32,
    _pad0: u32,
    beta: f32,
    j: f32,
    h: f32,
    _pad1: f32,
}

/// The Ising Markov chain run on the GPU, yielding sampled [`Configuration`]s on
/// sites.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely. Fixed at `Q = 2`, `D = 2`.
pub struct GpuIsingChain {
    sweeper: DeviceSweeper,
}

impl GpuIsingChain {
    /// Build a chain on `gpu` over a copy of `start`, uploaded to the device.
    ///
    /// `start` is read only to upload it, so the host copy is untouched and the
    /// same configuration can seed a CPU run too. `j`, `h`, `beta` are the Ising
    /// parameters; `seed` keys the counter-based RNG. `sweeps_between` is the
    /// decorrelation stride, and `batch` is how many samples are produced per
    /// device round-trip.
    ///
    /// Runs no sweeps — like [`Chain::new`](crate::chain::Chain::new), warmup is
    /// the caller's job via [`advance`](GpuIsingChain::advance).
    ///
    /// # Panics
    ///
    /// Panics if `batch` is zero, if `start` is not a site field of this
    /// lattice, or if either extent is odd.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Gpu,
        lattice: &Lattice<D>,
        j: f64,
        h: f64,
        beta: f64,
        seed: u64,
        start: &Configuration<2>,
        sweeps_between: usize,
        batch: usize,
    ) -> Self {
        assert!(batch > 0, "batch size must be positive");
        // The shader indexes the neighbor table by site, so a link field would
        // be silently misread rather than rejected; the cell kind is what makes
        // that checkable at all.
        assert_eq!(
            start.cell(),
            Cell::Site,
            "the GPU site checkerboard updates sites, so the start must be a site field"
        );
        assert_eq!(
            start.n_vars(),
            lattice.n_sites(),
            "start configuration and lattice disagree on site count"
        );
        let shape = lattice.shape();
        assert_even_extents(&shape, "site");

        let n_sites = lattice.n_sites();

        let spins: Vec<u32> = start.variables().iter().map(|s| s.index() as u32).collect();
        let mut neighbors: Vec<u32> = Vec::with_capacity(n_sites * 2 * D);
        for site in 0..n_sites {
            neighbors.extend(lattice.site_neighbors(site).iter().map(|&nb| nb as u32));
        }
        let params = Params {
            n_sites: n_sites as u32,
            width: shape[0] as u32,
            seed: fold_seed(seed),
            _pad0: 0,
            beta: beta as f32,
            j: j as f32,
            h: h as f32,
            _pad1: 0.0,
        };

        GpuIsingChain {
            sweeper: DeviceSweeper::build(
                gpu,
                SweepSetup {
                    label: "ising checkerboard",
                    shader: SHADER,
                    vars_init: &spins,
                    table: &neighbors,
                    params: bytemuck::bytes_of(&params),
                    cell: Cell::Site,
                    n_vars: n_sites,
                    // One thread per site, which on a site field is also one per
                    // variable.
                    threads: n_sites,
                    colors: N_COLORS,
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

impl Iterator for GpuIsingChain {
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
    use crate::model::Ising;
    use crate::observables::measure;
    use crate::rng::RandRng;

    /// A device can be acquired on this machine.
    #[test]
    fn initializes_a_device() {
        let Some(_gpu) = require_gpu() else {
            return;
        };
    }

    /// With `sweeps_between = 0` a "sample" runs no sweeps, so the round-trip
    /// upload → device buffer → read-back must return the start configuration
    /// unchanged. This exercises the buffer plumbing in isolation from the sweep.
    #[test]
    fn uploads_and_reads_back_unchanged() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([4, 4]);
        let mut rng = RandRng::seed_from_u64(0);
        let start = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        let mut chain = GpuIsingChain::new(gpu, &lat, 1.0, 0.0, 0.5, 7, &start, 0, 1);
        let got = chain.next().expect("open-ended stream yields");

        assert_eq!(got, start, "zero-sweep round-trip must be the identity");
    }

    /// The GPU checkerboard samples the same distribution as the CPU one: at a
    /// disordered temperature (fast mixing, self-averaging energy) the two agree
    /// on the mean energy density and magnetization within a loose tolerance.
    /// This is a distributional check — the RNGs differ, so it is not bit-for-bit.
    #[test]
    fn matches_the_cpu_checkerboard_distribution() {
        let Some(gpu) = require_gpu() else {
            return;
        };

        let shape = [16, 16];
        let (j, h, beta) = (1.0, 0.0, 0.25); // disordered: T well above T_c
        let (thermalize, sweeps_between, n) = (300, 4, 500);
        let n_sites = (shape[0] * shape[1]) as f64;
        let model = Ising::new(j, h);

        // CPU reference: Chain driven by the SiteCheckerboard updater.
        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            use crate::updater::SiteCheckerboard;
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let updater = SiteCheckerboard;
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
            let samples: Vec<_> = chain.take(n).map(|c| measure(&model, &lat, &c)).collect();
            mean_densities(&samples, n_sites)
        };

        // GPU: GpuIsingChain over the same model and geometry.
        let (e_gpu, m_gpu) = {
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let mut chain =
                GpuIsingChain::new(gpu, &lat, j, h, beta, 12345, &start, sweeps_between, 64);
            chain.advance(thermalize);
            let samples: Vec<_> = chain.take(n).map(|c| measure(&model, &lat, &c)).collect();
            mean_densities(&samples, n_sites)
        };

        assert!(
            (e_cpu - e_gpu).abs() < 0.05,
            "energy density mismatch: cpu {e_cpu:.4} vs gpu {e_gpu:.4}"
        );
        assert!(
            (m_cpu - m_gpu).abs() < 0.08,
            "|m| mismatch: cpu {m_cpu:.4} vs gpu {m_gpu:.4}"
        );
    }

    /// Mean energy density and mean |m| over a set of samples.
    fn mean_densities(samples: &[crate::observables::Sample], n_sites: f64) -> (f64, f64) {
        let count = samples.len() as f64;
        let e = samples.iter().map(|s| s.energy / n_sites).sum::<f64>() / count;
        let m = samples
            .iter()
            .map(|s| (s.magnetization / n_sites).abs())
            .sum::<f64>()
            / count;
        (e, m)
    }
}
