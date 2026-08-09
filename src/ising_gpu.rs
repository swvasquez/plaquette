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
use crate::device::{DeviceSweeper, Gpu, SweepSetup, assert_even_extents, fold_seed, site_colors};
use crate::lattice::Lattice;

/// Color passes per sweep: the two coordinate-sum parities. A site's color does
/// not depend on the dimension — a step along any axis flips the parity, in one
/// dimension as in six — so unlike the link coloring this count is a constant.
const N_COLORS: u32 = 2;

/// The compiled kernel: the shared preamble followed by the site checkerboard.
const SHADER: &str = crate::device::shader_source!("ising_checkerboard.wgsl");

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct's uniform layout. It carries no geometry: the kernel reads a
/// site's color from a table and takes the dimension as an override, so nothing
/// about the shape has to be described here.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
    h: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

/// The Ising Markov chain run on the GPU, yielding sampled [`Configuration`]s on
/// sites.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely. Fixed at `Q = 2`.
///
/// The type carries no dimension. It reads the lattice once, in
/// [`new`](GpuIsingChain::new), to build the tables it uploads, and afterwards
/// holds only device buffers and counts — so `D` is a parameter of the
/// constructor rather than of the chain.
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
    /// lattice, or if any extent is odd.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize>(
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
        let mut neighbors: Vec<u32> = Vec::with_capacity(n_sites * Lattice::<D>::neighbor_stride());
        for site in 0..n_sites {
            neighbors.extend(lattice.site_neighbors(site).iter().map(|&nb| nb as u32));
        }
        let site_color = site_colors(lattice);
        let params = Params {
            n_sites: n_sites as u32,
            seed: fold_seed(seed),
            beta: beta as f32,
            j: j as f32,
            h: h as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        GpuIsingChain {
            sweeper: DeviceSweeper::build(
                gpu,
                SweepSetup {
                    label: "ising checkerboard",
                    shader: SHADER,
                    vars_init: &spins,
                    table: &neighbors,
                    site_color: &site_color,
                    params: bytemuck::bytes_of(&params),
                    dimension: D as u32,
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

    /// Every site is reached when the launch has to span more than one row of
    /// workgroups.
    ///
    /// One row is `max_compute_workgroups_per_dimension * 64` threads, about 4.2
    /// million on any adapter, and it is an API-level cap rather than a
    /// conservative default — no limits request raises it. Past it the host
    /// launches a rectangle and the kernel folds the two axes back into a site
    /// index. Nothing smaller exercises that folding, so this lattice is sized
    /// deliberately just over the edge: `2048^2 = 4_194_304` sites against a
    /// `65535 * 64 = 4_194_240` row.
    ///
    /// The physics is chosen to make a missed site unmistakable rather than
    /// merely unlikely. At `j = 0` a flip costs `2 * s_i * h`, so with `h < 0`
    /// every spin of a cold field flips on its first visit, whatever its
    /// neighbors are doing. After one sweep the magnetization must be exactly
    /// `-N`: a site the grid failed to reach would still be `+1` and shift it by
    /// a countable amount, and a site reached twice would flip back.
    #[test]
    fn a_launch_spanning_two_rows_reaches_every_site() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([2048, 2048]);
        let n_sites = lat.n_sites();
        assert!(
            n_sites > 65535 * 64,
            "this lattice must not fit in one row of workgroups"
        );

        let start = Configuration::<2>::cold(&lat, Cell::Site);
        let mut chain = GpuIsingChain::new(gpu, &lat, 0.0, -1.0, 1.0, 7, &start, 1, 1);
        let after = chain.next().expect("open-ended stream yields");

        let model = Ising::new(0.0, -1.0);
        assert_eq!(
            measure(&model, &lat, &after).magnetization,
            -(n_sites as f64),
            "every site should have flipped exactly once"
        );
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
