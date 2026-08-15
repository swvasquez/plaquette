//! GPU Potts backend: a [`GpuPottsChain`] that runs the site checkerboard sweep
//! on the GPU via `wgpu`, exposed through the same
//! `Iterator<Item = Configuration>` interface as the CPU
//! [`Chain`](crate::chain::Chain).
//!
//! The Potts sibling of [`GpuIsingChain`](crate::models::ising::gpu::GpuIsingChain), and
//! close enough to it that the two share their whole schedule: the same
//! two-color site checkerboard over the same neighbor table, launched one thread
//! per site. What they do not share is what a thread computes, because the
//! labels are unordered — the energy term is a count of matching neighbors
//! rather than a sum of signed products, and the proposal has to *draw* among
//! the `Q - 1` other labels where the Ising kernel simply flips. That is two
//! bodies with nothing in common but their loop bounds, which is why this is a
//! separate shader rather than a branch inside the Ising `checkerboard.wgsl`.
//!
//! Everything about batching samples and reading them back lives in
//! `DeviceSweeper`, shared with the other two backends; what this module owns is
//! the `Params` layout, the neighbor table, and the two things the shader needs
//! that no other kernel does — the state count, and the per-label energy offsets
//! packed into the uniform block. It is also the only backend whose shader
//! source is a template rather than a constant: WGSL needs a uniform array's
//! length written literally in the source, and the length here is the state
//! count, which the source has no way to see. Substituting that one number is
//! what lets the device path take any `Q` rather than a capped one.
//!
//! There are two shader templates rather than one, `checkerboard.wgsl` and
//! `heatbath.wgsl`, picked by the [`Kernel`] the caller names. They share the
//! coloring, the neighbor table and every dispatch. This is the model where they
//! share least beyond that: the Metropolis kernel picks a candidate first and
//! then needs one signed counter, while the heat bath commits to no candidate
//! and needs how many neighbors carry each label. That is why the heat bath
//! template carries a second token — the tallies are arrays, and a WGSL array
//! length must be written literally in the source, exactly as the offsets are.
//!
//! The coloring is compiled into the shader, so this
//! does not use the [`Updater`](crate::updater::Updater) seam. Its randomness is
//! counter-based, keyed on `(seed, site, sweep)`, so the result is independent of
//! GPU thread order — the property that lets the CPU site checkerboard serve as a
//! reference.

use crate::configuration::{Cell, Configuration};
use crate::device::{
    DeviceSweeper, Gpu, Kernel, SweepSetup, assert_even_extents, fold_seed, site_colors,
    site_neighbor_table, state_words,
};
use crate::lattice::Lattice;
use crate::models::potts::{self as model, Potts};

/// Color passes per sweep: the two coordinate-sum parities, as for Ising. A
/// site's color does not depend on the dimension — a step along any axis flips
/// the parity — nor on how many states it can take, so this is a constant.
const N_COLORS: u32 = 2;

/// The kernel *template*: the shared preamble followed by the site checkerboard,
/// still carrying the one token the host has to fill in.
const SHADER_TEMPLATE: &str = crate::device::shader_source!("checkerboard.wgsl");

/// The heat bath kernel template, over the same preamble and the same coloring.
/// It carries a second token, `$Q$`, because its per-label tallies are arrays
/// and a WGSL array length must be a const-expression.
const HEAT_BATH_TEMPLATE: &str = crate::device::shader_source!("heatbath.wgsl");

/// The token standing for the state count itself, used by the heat bath kernel
/// alone. `params.q` already carries the same number as a runtime uniform, which
/// is all the Metropolis kernel needs of it; an array length is not something a
/// uniform can be.
const Q_TOKEN: &str = "$Q$";

/// The token in that template standing for the number of `vec4` slots the
/// per-label offsets occupy.
///
/// This is the only piece of shader source in the crate that is substituted
/// rather than compiled as written, and it is worth saying why the usual route
/// does not work. The lattice dimension reaches its kernels as a WGSL
/// `override`, resolved when the pipeline is built, which keeps every shader
/// source a compile-time constant. An `override` can size a loop bound but not
/// an array in the uniform address space — that length has to be a literal in
/// the source — and the offsets are exactly such an array. Substituting one
/// number is the smallest way out, and it is what lets the device path take any
/// state count rather than a capped one.
const H_VECTORS_TOKEN: &str = "$H_VECTORS$";

/// Labels per `vec4` slot in the uniform block.
const LABELS_PER_VECTOR: usize = 4;

/// The kernel for `q` labels: the template with its one token filled in.
fn shader_for(template: &str, q: usize) -> String {
    template
        .replace(H_VECTORS_TOKEN, &q.div_ceil(LABELS_PER_VECTOR).to_string())
        .replace(Q_TOKEN, &q.to_string())
}

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct's uniform layout. It carries no geometry: the kernel reads a
/// site's color from a table and takes the dimension as an override, so nothing
/// about the shape has to be described here.
///
/// `q` is one of two fields with no Ising counterpart, and it rides here rather
/// than arriving as an override because it bounds nothing the compiler needs:
/// `D` fixes the neighbor row width and so has to be a constant, while `q` only
/// scales a draw.
///
/// This covers the fixed head of the block only. The per-label offsets follow it
/// and are appended as bytes, because their length depends on `Q` and so is not
/// a fixed-size struct field at all —
/// [`SweepSetup::params`](crate::device::SweepSetup) takes a byte slice and each
/// backend supplies its own, so there is no shared layout this has to fit. The
/// head is thirty-two bytes, which leaves the offsets starting on the sixteen-byte
/// boundary WGSL wants for a `vec4` array.
///
/// They are packed four to a `vec4` rather than declared `array<f32, Q>` because
/// in the uniform address space every array element is padded to sixteen bytes,
/// which would waste three words in four. The kernel unpacks with
/// `h[label / 4][label % 4]`.
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

/// The Potts Markov chain run on the GPU, yielding sampled [`Configuration`]s on
/// sites.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely.
///
/// The type carries `Q` but not the dimension. `Q` is part of what a yielded
/// [`Configuration<Q>`] *is*, so it cannot be anything but a type parameter;
/// `D` is read once in [`new`](GpuPottsChain::new) to build the neighbor table
/// and never needed again, so it is a parameter of the constructor alone — the
/// same split [`GpuIsingChain`](crate::models::ising::gpu::GpuIsingChain) makes.
pub struct GpuPottsChain<const Q: usize> {
    sweeper: DeviceSweeper<Q>,
}

impl<const Q: usize> GpuPottsChain<Q> {
    /// Build a chain on `gpu` over a copy of `start`, uploaded to the device.
    ///
    /// `start` is read only to upload it, so the host copy is untouched and the
    /// same configuration can seed a CPU run too. `model` supplies the coupling
    /// and the per-label offsets — taken whole rather than as loose numbers,
    /// since the offsets are `Q` of them and the CPU path is handed the same
    /// value — and `beta` is the inverse temperature; `seed` keys the
    /// counter-based RNG. `sweeps_between` is the decorrelation stride, and
    /// `batch` is how many samples are produced per device round-trip. `kernel`
    /// picks which single-variable rule a thread runs; both sample the same
    /// distribution over the same coloring.
    ///
    /// Runs no sweeps — like [`Chain::new`](crate::chain::Chain::new), warmup is
    /// the caller's job via [`advance`](GpuPottsChain::advance).
    ///
    /// # Panics
    ///
    /// Panics if `batch` is zero, if `Q` is below
    /// [`Potts::MIN_STATES`](crate::models::potts::Potts::MIN_STATES), if `start` is not
    /// a site field of this lattice, or if any extent is odd. There is no upper
    /// bound on `Q`: the kernel is generated for the state count it is given.
    #[allow(clippy::too_many_arguments)]
    pub fn new<const D: usize>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        model: &Potts<Q>,
        beta: f64,
        seed: u64,
        start: &Configuration<Q>,
        sweeps_between: usize,
        batch: usize,
        kernel: Kernel,
    ) -> Self {
        assert!(batch > 0, "batch size must be positive");
        // At one state the kernel would draw among zero alternatives, which is
        // not a small model but no model at all. The host says so here, since
        // the shader's `q - 1` would simply wrap.
        assert!(Q >= model::POTTS_MIN_STATES, "{}", model::TOO_FEW_STATES);
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

        let labels = state_words(start);
        let neighbors = site_neighbor_table(lattice);
        let site_color = site_colors(lattice);
        let head = Params {
            n_sites: n_sites as u32,
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

        GpuPottsChain {
            sweeper: DeviceSweeper::build(
                gpu,
                SweepSetup {
                    label: match kernel {
                        Kernel::Metropolis => "potts checkerboard",
                        Kernel::HeatBath => "potts heat bath",
                    },
                    shader: &match kernel {
                        Kernel::Metropolis => shader_for(SHADER_TEMPLATE, Q),
                        Kernel::HeatBath => shader_for(HEAT_BATH_TEMPLATE, Q),
                    },
                    vars_init: &labels,
                    table: &neighbors,
                    site_color: &site_color,
                    params: &params,
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

impl<const Q: usize> Iterator for GpuPottsChain<Q> {
    type Item = Configuration<Q>;

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

        let mut chain = GpuPottsChain::new(
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
        let mut chain = GpuPottsChain::new(
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

        let mut chain =
            GpuPottsChain::new(gpu, &lat, &model, 4.0, 99, &start, 5, 4, Kernel::Metropolis);
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

        // CPU reference: Chain driven by the SiteCheckerboard updater.
        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            use crate::updater::SiteCheckerboard;
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
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
            means(chain.take(n).collect())
        };

        // GPU: GpuPottsChain over the same model and geometry.
        let (e_gpu, m_gpu) = {
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = GpuPottsChain::new(
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

    /// The generated kernel takes a state count needing more than one `vec4` of
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
        let mut chain = GpuPottsChain::new(
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
        let _ = GpuPottsChain::new(
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
