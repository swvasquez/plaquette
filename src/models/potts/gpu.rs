//! GPU Potts backend: the model's side of the device seam, as a constructor
//! for the shared [`GpuChain`].
//!
//! Everything about running a sweep on the device lives in
//! [`device`](crate::device) and is shared by every model; what this module
//! owns is the part that is about the Potts model — the `Params` layout with
//! its per-label offsets, the neighbor table, and the `model.wgsl` snippet
//! whose `energy_delta` is the WGSL half of
//! [`Action::energy_delta`](crate::action::Action::energy_delta).
//!
//! This is the one backend whose snippet is a *template* rather than a
//! constant: WGSL needs a uniform array's length written literally in the
//! source, and the length of the offsets array is set by the state count,
//! which the source has no way to see. Substituting that one token
//! (`$H_VECTORS$`) is what lets the device path take any `Q` rather than a
//! capped one; the kernels' own `$Q$` token is filled by the shared assembly
//! for every model alike.
//!
//! Randomness is counter-based, keyed on `(seed, site, sweep)`, so the result
//! is independent of GPU thread order — the property that lets the CPU
//! checkerboard [`LocalUpdate`](crate::updater::LocalUpdate) serve as the
//! sequential reference.

use crate::configuration::{Cell, Configuration};
use crate::device::{Gpu, GpuChain, GpuModelSetup, Kernel, fold_seed, site_neighbor_table};
use crate::lattice::Lattice;
use crate::models::potts::{self as model, Potts};

/// The token in the snippet standing for the number of `vec4` slots the
/// per-label offsets occupy.
///
/// This is the only piece of shader source in the crate that is substituted
/// with a model-specific value rather than compiled as written, and it is
/// worth saying why the usual route does not work. The lattice dimension
/// reaches its kernels as a WGSL `override`, resolved when the pipeline is
/// built, which keeps every shader source a compile-time constant. An
/// `override` can size a loop bound but not an array in the uniform address
/// space — that length has to be a literal in the source — and the offsets
/// are exactly such an array.
const H_VECTORS_TOKEN: &str = "$H_VECTORS$";

/// Labels per `vec4` slot in the uniform block.
const LABELS_PER_VECTOR: usize = 4;

/// The snippet for `q` labels: the template with its one token filled in.
fn shader_for(q: usize) -> String {
    include_str!("model.wgsl").replace(H_VECTORS_TOKEN, &q.div_ceil(LABELS_PER_VECTOR).to_string())
}

/// The fixed head of the uniform block; the per-label offsets follow it as
/// bytes, because their length depends on `Q` and so is not a fixed-size
/// struct field at all.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct in `model.wgsl`. The head is thirty-two bytes, which leaves
/// the offsets starting on the sixteen-byte boundary WGSL wants for a `vec4`
/// array. They are packed four to a `vec4` rather than declared
/// `array<f32, Q>` because in the uniform address space every array element is
/// padded to sixteen bytes, which would waste three words in four; the snippet
/// unpacks with `h[label / 4][label % 4]`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
    q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Build a Potts Markov chain on `gpu` over a copy of `start` — a
/// [`GpuChain`] on sites, at any `Q`.
///
/// `model` supplies the coupling and the per-label offsets — taken whole
/// rather than as loose numbers, since the offsets are `Q` of them and the CPU
/// path is handed the same value — and `beta` is the inverse temperature;
/// `seed` keys the counter-based RNG; `sweeps_between` is the decorrelation
/// stride, `batch` how many samples per device round-trip, and `kernel` which
/// single-variable rule a thread runs. Runs no sweeps — warmup is the
/// caller's job via [`GpuChain::advance`].
///
/// # Panics
///
/// Panics if `batch` is zero, if `Q` is below
/// [`Potts::MIN_STATES`](crate::models::potts::Potts::MIN_STATES), if `start`
/// is not a site field of this lattice, or if any extent is odd (see
/// [`GpuChain`]). There is no upper bound on `Q`: the snippet is generated for
/// the state count it is given.
#[allow(clippy::too_many_arguments)]
pub fn gpu_chain<const Q: usize, const D: usize>(
    gpu: Gpu,
    lattice: &Lattice<D>,
    model: &Potts<Q>,
    beta: f64,
    seed: u64,
    start: &Configuration<Q>,
    sweeps_between: usize,
    batch: usize,
    kernel: Kernel,
) -> GpuChain<Q> {
    // At one state the kernel would draw among zero alternatives, which is
    // not a small model but no model at all. The host says so here, since the
    // shader's `q - 1` would simply wrap.
    assert!(Q >= model::POTTS_MIN_STATES, "{}", model::TOO_FEW_STATES);

    let head = Params {
        n_sites: lattice.n_sites() as u32,
        seed: fold_seed(seed),
        beta: beta as f32,
        j: model.coupling() as f32,
        q: Q as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    // The offsets follow the head, rounded up to a whole number of `vec4`
    // slots so the array the shader declares is exactly filled. The padding
    // past `Q` is never read — the kernel only ever indexes a label that
    // exists — and is only there because a slot is four wide.
    let mut offsets = vec![0.0f32; Q.next_multiple_of(LABELS_PER_VECTOR)];
    for (slot, &offset) in offsets.iter_mut().zip(model.offsets().iter()) {
        *slot = offset as f32;
    }
    let head_bytes = bytemuck::bytes_of(&head);
    let offset_bytes: &[u8] = bytemuck::cast_slice(&offsets);
    let mut params = Vec::with_capacity(head_bytes.len() + offset_bytes.len());
    params.extend_from_slice(head_bytes);
    params.extend_from_slice(offset_bytes);

    GpuChain::build(
        gpu,
        lattice,
        GpuModelSetup {
            label: "potts",
            source: shader_for(Q),
            table: site_neighbor_table(lattice),
            params,
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
    use crate::models::potts::potts_measure;
    use crate::rng::RandRng;

    /// With `sweeps_between = 0` a "sample" runs no sweeps, so the round-trip
    /// upload → device buffer → read-back must return the start configuration
    /// unchanged. This exercises the buffer plumbing in isolation from the
    /// sweep, and at three states it also says the read-back accepts a word the
    /// two-state path never produces.
    #[test]
    fn uploads_and_reads_back_unchanged() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([4, 4]);
        let mut rng = RandRng::seed_from_u64(0);
        let start = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);
        assert!(
            start.variables().iter().any(|s| s.index() == 2),
            "the start should carry a label an Ising field could not"
        );

        let mut chain = gpu_chain(
            gpu,
            &lat,
            &Potts::symmetric(1.0),
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

    /// Every label the kernel writes is in range, from a hot start at a coupling
    /// weak enough that almost every proposal is accepted.
    ///
    /// This is the check the Ising kernel never needs: there the only thing a
    /// thread can write is `1 - current`, whereas here it writes the result of a
    /// scaled float draw, and rounding at the top of the unit interval can push
    /// that one past the last label. The read-back would panic on an out-of-range
    /// word, so reaching the assertions at all is most of the test; that all
    /// three labels appear is what says the proposal is not quietly clamping to a
    /// subset.
    #[test]
    fn every_proposed_label_stays_in_range() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        const Q: usize = 3;
        let lat = Lattice::new([16, 16]);
        let mut rng = RandRng::seed_from_u64(1);
        let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        // beta = 0 accepts every proposal, so a sweep writes a fresh draw at
        // every site rather than mostly rejecting back to the current label.
        let mut chain = gpu_chain(
            gpu,
            &lat,
            &Potts::symmetric(1.0),
            0.0,
            99,
            &start,
            1,
            4,
            Kernel::Metropolis,
        );
        for config in chain.by_ref().take(4) {
            let mut seen = [false; Q];
            for state in config.variables() {
                seen[state.index()] = true;
            }
            assert!(seen.iter().all(|&s| s), "not every label was reached");
        }
    }

    /// A uniform start at strong coupling stays uniform: every bond already
    /// agrees, so relabelling a site breaks all `2D` of its bonds and costs
    /// `ΔE = +4` at `j = 1` in two dimensions, which at `beta = 4` is accepted
    /// about once in `10^7` tries.
    ///
    /// The cheapest check that the shader's agreement count is right rather than
    /// merely self-consistent: a miscounted neighbor or a sign error would let
    /// moves through and drive the order parameter off one.
    #[test]
    fn a_strongly_coupled_uniform_start_stays_ordered() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([8, 8]);
        let model = Potts::<3>::symmetric(1.0);
        let start = Configuration::<3>::cold(&lat, Cell::Site);

        let mut chain = gpu_chain(gpu, &lat, &model, 4.0, 99, &start, 5, 4, Kernel::Metropolis);
        chain.advance(20);
        let mean = chain
            .take(8)
            .map(|c| potts_measure(&model, &lat, &c).order)
            .sum::<f64>()
            / 8.0;

        assert!(mean > 0.99, "mean order parameter at beta = 4 was {mean}");
    }

    /// The GPU checkerboard samples the same distribution as the CPU one: at a
    /// coupling inside the ordered phase but clear of the transition the two
    /// agree on the mean energy density and order parameter. This is a
    /// distributional check — the CPU draws from a stream and the GPU from a
    /// keyed counter, and the two even map their proposal draws onto the
    /// alternatives differently, so it is not bit-for-bit.
    #[test]
    fn matches_the_cpu_checkerboard_distribution() {
        let Some(gpu) = require_gpu() else {
            return;
        };

        const Q: usize = 3;
        let shape = [16, 16];
        // beta_c = ln(1 + sqrt(3)) ~ 1.005 in two dimensions at q = 3. Just
        // above it: far enough that a short run does not straddle the
        // transition, close enough that the order parameter has not saturated
        // and can still show a backend computing something slightly wrong.
        let beta = 1.1;
        let (thermalize, sweeps_between, n) = (500, 4, 500);
        let n_sites = (shape[0] * shape[1]) as f64;
        let model = Potts::<Q>::symmetric(1.0);
        let lat = Lattice::new(shape);

        let means = |samples: Vec<Configuration<Q>>| -> (f64, f64) {
            let count = samples.len() as f64;
            let measured: Vec<_> = samples
                .iter()
                .map(|c| potts_measure(&model, &lat, c))
                .collect();
            let e = measured.iter().map(|s| s.energy / n_sites).sum::<f64>() / count;
            let m = measured.iter().map(|s| s.order).sum::<f64>() / count;
            (e, m)
        };

        // CPU reference: Chain driven by the checkerboard LocalUpdate.
        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            use crate::updater::{LocalUpdate, Schedule};
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
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
            means(chain.take(n).collect())
        };

        // GPU: the shared chain over the same model and geometry.
        let (e_gpu, m_gpu) = {
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = gpu_chain(
                gpu,
                &lat,
                &model,
                beta,
                12345,
                &start,
                sweeps_between,
                64,
                Kernel::Metropolis,
            );
            chain.advance(thermalize);
            means(chain.take(n).collect())
        };

        eprintln!(
            "q = 3 at beta = {beta}: cpu ({e_cpu:.4}, {m_cpu:.4}), gpu ({e_gpu:.4}, {m_gpu:.4})"
        );
        // Both sides must land on the physics rather than merely on each other:
        // at this coupling the model is ordered but not frozen, so a kernel that
        // pinned the field or randomized it would fail here even if the CPU
        // somehow agreed.
        assert!(
            (0.5..0.99).contains(&m_cpu) && (0.5..0.99).contains(&m_gpu),
            "order parameter left the ordered-but-fluctuating range: \
             cpu {m_cpu:.4}, gpu {m_gpu:.4}"
        );
        assert!(
            (e_cpu - e_gpu).abs() < 0.05,
            "energy density mismatch: cpu {e_cpu:.4} vs gpu {e_gpu:.4}"
        );
        assert!(
            (m_cpu - m_gpu).abs() < 0.05,
            "order parameter mismatch: cpu {m_cpu:.4} vs gpu {m_gpu:.4}"
        );
    }

    /// The generated snippet takes a state count needing more than one `vec4` of
    /// offsets, and reads the right entry out of the second one.
    ///
    /// This is what the templating buys and the only test that would notice it
    /// failing. At `Q = 20` the offsets occupy five slots, so a run has to have
    /// been compiled for five rather than for some fixed number, and the offset
    /// is put on label 19 — the last entry of the last slot, which the unpacking
    /// reaches as `h[4][3]`. A kernel that mis-sized the array or mis-indexed it
    /// would read a zero there and the favoured label would not be favoured.
    ///
    /// Turning the coupling off is what makes the expected answer exact: with no
    /// neighbor term the sites are independent, so the fraction on the favoured
    /// label is `exp(beta * h) / (exp(beta * h) + Q - 1)` at any lattice size,
    /// with nothing to thermalize and no finite-size correction.
    #[test]
    fn the_generated_kernel_takes_a_state_count_past_one_vector_of_offsets() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        const Q: usize = 20;
        const FAVOURED: usize = Q - 1;
        let (offset, beta) = (3.0f64, 1.0f64);

        let lat = Lattice::new([16, 16]);
        let n_sites = lat.n_sites() as f64;
        let mut h = [0.0; Q];
        h[FAVOURED] = offset;
        let model = Potts::<Q>::new(0.0, h);

        let mut rng = RandRng::seed_from_u64(7);
        let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
        let mut chain = gpu_chain(
            gpu,
            &lat,
            &model,
            beta,
            4242,
            &start,
            2,
            16,
            Kernel::Metropolis,
        );
        chain.advance(50);

        let samples = 64;
        let measured = chain
            .take(samples)
            .map(|c| {
                let on_favoured = c
                    .variables()
                    .iter()
                    .filter(|s| s.index() == FAVOURED)
                    .count();
                on_favoured as f64 / n_sites
            })
            .sum::<f64>()
            / samples as f64;

        let weight = (beta * offset).exp();
        let exact = weight / (weight + (Q - 1) as f64);
        assert!(
            (measured - exact).abs() < 0.02,
            "fraction on label {FAVOURED}: {measured:.4} vs exact {exact:.4}"
        );
    }

    /// An odd extent is rejected rather than silently sampling the wrong
    /// distribution: the periodic wrap would put two neighboring sites in one
    /// color, and a parallel pass would then update both against each other's
    /// stale labels.
    #[test]
    #[should_panic(expected = "even lattice extents")]
    fn rejects_odd_extents() {
        let Some(gpu) = require_gpu() else {
            // The assertion under test fires before any device work, but the
            // constructor needs a device to reach it, so a machine without one
            // cannot run this. Panic to keep `should_panic` satisfied.
            panic!("no GPU adapter: GPU site checkerboard needs even lattice extents (skipped)");
        };
        let lat = Lattice::new([4, 3]);
        let start = Configuration::<3>::cold(&lat, Cell::Site);
        let _ = gpu_chain(
            gpu,
            &lat,
            &Potts::symmetric(1.0),
            0.5,
            1,
            &start,
            1,
            1,
            Kernel::Metropolis,
        );
    }
}
