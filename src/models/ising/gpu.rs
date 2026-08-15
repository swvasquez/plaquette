//! GPU Ising backend: the model's side of the device seam, as a constructor
//! for the shared [`GpuChain`].
//!
//! Everything about running a sweep on the device — assembling the shader
//! from prelude, model snippet, kernel, and schedule; encoding color passes;
//! batching samples — lives in [`device`](crate::device) and is shared by
//! every model. What this module owns is the part that is about the Ising
//! model: the `Params` layout, the neighbor table, and the `model.wgsl`
//! snippet whose `energy_delta` is the WGSL half of
//! [`Action::energy_delta`](crate::action::Action::energy_delta).
//!
//! The device runs the checkerboard schedule — on this model's site field,
//! two parity colors — with whichever [`Kernel`] the caller names. Its
//! randomness is counter-based, keyed on `(seed, site, sweep)`, so the result
//! is independent of GPU thread order — the property that lets the CPU
//! checkerboard [`LocalUpdate`](crate::updater::LocalUpdate) serve as the
//! sequential reference.

use crate::configuration::{Cell, Configuration};
use crate::device::{Gpu, GpuChain, GpuModelSetup, Kernel, fold_seed, site_neighbor_table};
use crate::lattice::Lattice;

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct in `model.wgsl`. The head — `n_sites`, `seed`, `beta` — is
/// the layout contract with the shared kernel and schedule fragments; the tail
/// is this model's own. It carries no geometry: the kernel reads a site's
/// color from a table and takes the dimension as an override.
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

/// Build an Ising Markov chain on `gpu` over a copy of `start` — a
/// [`GpuChain`] on sites, fixed at `Q = 2`.
///
/// `j`, `h`, `beta` are the Ising parameters; `seed` keys the counter-based
/// RNG; `sweeps_between` is the decorrelation stride, `batch` how many samples
/// are produced per device round-trip, and `kernel` which single-variable rule
/// a thread runs. Runs no sweeps — warmup is the caller's job via
/// [`GpuChain::advance`].
///
/// # Panics
///
/// Panics if `batch` is zero, if `start` is not a site field of this lattice,
/// or if any extent is odd (see [`GpuChain`]).
#[allow(clippy::too_many_arguments)]
pub fn gpu_chain<const D: usize>(
    gpu: Gpu,
    lattice: &Lattice<D>,
    j: f64,
    h: f64,
    beta: f64,
    seed: u64,
    start: &Configuration<2>,
    sweeps_between: usize,
    batch: usize,
    kernel: Kernel,
) -> GpuChain<2> {
    let params = Params {
        n_sites: lattice.n_sites() as u32,
        seed: fold_seed(seed),
        beta: beta as f32,
        j: j as f32,
        h: h as f32,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    };

    GpuChain::build(
        gpu,
        lattice,
        GpuModelSetup {
            label: "ising",
            source: include_str!("model.wgsl").to_string(),
            table: site_neighbor_table(lattice),
            params: bytemuck::bytes_of(&params).to_vec(),
            cell: Cell::Site,
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
    use crate::models::ising::Ising;
    use crate::models::ising::measure;
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

        let mut chain = gpu_chain(
            gpu,
            &lat,
            1.0,
            0.0,
            0.5,
            7,
            &start,
            0,
            1,
            Kernel::Metropolis,
        );
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
        let mut chain = gpu_chain(
            gpu,
            &lat,
            0.0,
            -1.0,
            1.0,
            7,
            &start,
            1,
            1,
            Kernel::Metropolis,
        );
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

        // CPU reference: Chain driven by the checkerboard LocalUpdate.
        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            use crate::updater::{LocalUpdate, Schedule};
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
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
            let samples: Vec<_> = chain.take(n).map(|c| measure(&model, &lat, &c)).collect();
            mean_densities(&samples, n_sites)
        };

        // GPU: the shared chain over the same model and geometry.
        let (e_gpu, m_gpu) = {
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = gpu_chain(
                gpu,
                &lat,
                j,
                h,
                beta,
                12345,
                &start,
                sweeps_between,
                64,
                Kernel::Metropolis,
            );
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
    fn mean_densities(samples: &[crate::models::ising::Sample], n_sites: f64) -> (f64, f64) {
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
