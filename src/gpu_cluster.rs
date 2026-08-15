//! The GPU cluster backend: a [`GpuClusterChain`] running a cluster sweep —
//! Swendsen–Wang or Wolff, composed from the same [`Extent`] and [`Relabel`]
//! axes as the CPU updater — on the device, exposed through the same
//! `Iterator<Item = Configuration>` interface as the CPU
//! [`Chain`](crate::chain::Chain).
//!
//! It is a sibling of the checkerboard backends rather than a mode of them,
//! because it cannot use `DeviceSweeper`. That type's contract is a fixed number
//! of dispatches per sweep, and cluster labeling needs an iterate-until-converged
//! loop whose trip count depends on data the device produces. So this module
//! owns its own driver and reuses only the free helpers from
//! [`device`](crate::device) — acquiring the adapter, folding the seed, building
//! the neighbor table, checking buffer limits, sizing a launch grid.
//!
//! # Why this one is not per-model
//!
//! The checkerboard backends are written per model because a thread's work
//! genuinely differs between them: the Ising kernel flips a bit where the Potts
//! kernel draws among the `Q - 1` other labels. A cluster sweep has no such
//! split. Placing bonds, seeding the labels, and iterating them to a fixed point
//! are graph work that knows only whether two sites agree, and the relabel that
//! follows draws uniformly from all `Q` states — which at two states *is* the
//! Ising cluster move, not an analogue of it. So one chain serves any model
//! implementing [`BondAction`], and it reaches the model through that trait
//! alone.
//!
//! The shaders stay split all the same. `cluster_prelude.wgsl` holds the graph
//! stages and `cluster_relabel.wgsl` the move — both moves, since the redraw
//! and the forced change differ only in the label arithmetic and share the
//! keyed draw. A model whose cluster move is neither — a clock model
//! reflecting about an axis, say — would supply its own fourth stage without
//! touching the first three. That seam is kept open on the strength of the
//! shaders being separable, not on a guess about which model will need it.
//!
//! Wolff's seeded extent runs here as a filter on the full decomposition
//! rather than as a frontier walk: the prelude labels every cluster as it does
//! for Swendsen–Wang, and the relabel stage moves only the cluster holding a
//! keyed uniformly drawn seed site. Picking the cluster under a uniformly
//! random site is exactly Wolff's size-biased cluster choice, so the move is
//! the same one the CPU growth makes; what differs is the cost. A device Wolff
//! sweep pays for labeling the whole lattice and updates one cluster, so it
//! buys agreement with the CPU updater rather than speed — Swendsen–Wang
//! remains the natural cluster move on the device, and `docs/wolff.md` says
//! when the seeded move is worth that price anyway.
//!
//! # Why the host is in the loop
//!
//! A compute dispatch has no global barrier: a workgroup cannot learn whether
//! another workgroup moved a label, so convergence is only observable *between*
//! dispatches. The driver therefore runs a batch of propagation passes, reads a
//! one-word flag back, and dispatches again if anything moved. Over-running is
//! harmless — a converged pass is a no-op — which is what lets several passes
//! share one round-trip.
//!
//! Unlike the checkerboard backends this one has no even-extent requirement.
//! That rule protects a *coloring*, and there is no coloring here.
//!
//! Randomness is counter-based, keyed on `(seed, bond, sweep)` for the bonds and
//! on `(seed, root, sweep)` for the labels, so a sweep is independent of thread
//! order. It is not the stream the CPU updater consumes, so the two agree
//! distributionally and not bit-for-bit — the same relationship the checkerboard
//! backends already have with their CPU reference.

use wgpu::util::DeviceExt;

use crate::action::BondAction;
use crate::configuration::{Cell, Configuration};
use crate::device::{
    Gpu, PUSH_CONSTANT_BYTES, check_fits, fold_seed, grid_for, site_neighbor_table, state_words,
    storage_entry, uniform_entry,
};
use crate::lattice::Lattice;
use crate::state::State;
use crate::updater::{ClusterUpdate, Extent, Relabel};

/// The four-stage kernel: the shared randomness, the graph stages, and the
/// relabel.
const SHADER: &str = concat!(
    include_str!("wgsl/rng.wgsl"),
    include_str!("wgsl/cluster_prelude.wgsl"),
    include_str!("wgsl/cluster_relabel.wgsl"),
);

/// The fewest states a cluster redraw is defined at.
///
/// At one state every bond agrees, the whole lattice is one cluster, and the
/// redraw has nothing to choose between. The shader's `n_states - 1` would
/// simply wrap, so the host says so instead.
const MIN_STATES: usize = 2;

/// Propagation passes run between two readbacks of the convergence flag.
///
/// A readback is a full device round-trip and a pass is cheap, so checking after
/// every one would spend most of a sweep waiting. Over-running costs nothing
/// because a pass at the fixed point writes nothing.
const PASSES_PER_CHECK: usize = 4;

/// Bytes in the convergence flag — one `u32`.
const FLAG_BYTES: u64 = std::mem::size_of::<u32>() as u64;

/// The static run parameters uploaded to the shader's uniform buffer.
///
/// `#[repr(C)]` with explicit padding to a 16-byte multiple, matching the WGSL
/// `Params` struct's uniform layout. It carries the bond probability rather than
/// `beta` and a coupling, because that probability is constant over a run and
/// the device has no reason to recompute it per bond — the same reasoning that
/// lets the CPU updater read the gap once at construction.
///
/// There are no per-label offsets here, unlike the checkerboard backend's
/// block: a cluster update refuses a model that has any, so the shader has no
/// array to size and needs no templating.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_sites: u32,
    n_bonds: u32,
    seed: u32,
    p: f32,
    n_states: u32,
    /// 1 when only the seed site's cluster is relabeled ([`Extent::Seeded`]).
    seeded: u32,
    /// 1 when the label is forced to change ([`Relabel::ForcedChange`]).
    forced: u32,
    _pad: u32,
}

/// A Markov chain advanced by cluster updates — Swendsen–Wang or Wolff — on
/// the GPU, yielding sampled [`Configuration`]s on sites.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely.
///
/// It names no model. Everything it needs from one is the bond gap and the
/// relabeling symmetry, both of which arrive through [`BondAction`], so the same
/// chain drives the Ising and Potts models and would drive any other model
/// implementing that trait.
///
/// The type carries `Q` but not the dimension, the same split
/// [`GpuChain`](crate::device::GpuChain) makes: `Q` is part
/// of what a yielded [`Configuration<Q>`] is, while `D` is read once in
/// [`new`](GpuClusterChain::new) to build the neighbor table and never
/// needed again.
pub struct GpuClusterChain<const Q: usize> {
    gpu: Gpu,
    /// One pipeline per stage, all over the same module and bind group.
    bonds: wgpu::ComputePipeline,
    label_init: wgpu::ComputePipeline,
    propagate: wgpu::ComputePipeline,
    relabel: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,

    /// The evolving labels, one `u32` per site.
    labels: wgpu::Buffer,
    /// The one-word convergence flag, cleared before each batch of passes.
    changed: wgpu::Buffer,
    /// Read-back target for one configuration.
    label_staging: wgpu::Buffer,
    /// Read-back target for the flag.
    flag_staging: wgpu::Buffer,
    /// Buffers reached through `bind_group`; held only to keep them alive.
    _resources: Vec<wgpu::Buffer>,

    n_sites: usize,
    /// Workgroups for a one-thread-per-site stage, and for the one-thread-per-bond
    /// stage, which launches `D` times as many.
    site_grid: (u32, u32),
    bond_grid: (u32, u32),
    /// Propagation passes allowed before the run is declared broken.
    pass_cap: usize,
    sweeps_between: usize,
    /// Global sweep counter — the RNG key, so every sweep draws differently.
    sweeps_done: u32,
}

impl<const Q: usize> GpuClusterChain<Q> {
    /// The Swendsen–Wang composition: every cluster, freshly redrawn — the
    /// device counterpart of [`ClusterUpdate::swendsen_wang`].
    pub fn swendsen_wang<const D: usize, M: BondAction<Q>>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        model: &M,
        beta: f64,
        seed: u64,
        start: &Configuration<Q>,
        sweeps_between: usize,
    ) -> Self {
        Self::new(
            gpu,
            lattice,
            model,
            Extent::All,
            Relabel::Redraw,
            beta,
            seed,
            start,
            sweeps_between,
        )
    }

    /// The Wolff composition: one seeded cluster, forced onto a different
    /// label — the device counterpart of [`ClusterUpdate::wolff`]. See
    /// `docs/wolff.md`, and the module docs for what the seeded extent costs
    /// on the device.
    pub fn wolff<const D: usize, M: BondAction<Q>>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        model: &M,
        beta: f64,
        seed: u64,
        start: &Configuration<Q>,
        sweeps_between: usize,
    ) -> Self {
        Self::new(
            gpu,
            lattice,
            model,
            Extent::Seeded,
            Relabel::ForcedChange,
            beta,
            seed,
            start,
            sweeps_between,
        )
    }

    /// Build a cluster chain on `gpu` over a copy of `start`, uploaded to the
    /// device, composed from the same axes as [`ClusterUpdate::new`].
    ///
    /// `start` is read only to upload it, so the host copy is untouched and the
    /// same configuration can seed a CPU run too. `model` supplies the bond gap,
    /// through the same [`BondAction`] seam the CPU
    /// updater reads it from, and `beta` turns it into the bond probability;
    /// `seed` keys the counter-based RNG and `sweeps_between` is the
    /// decorrelation stride.
    ///
    /// Runs no sweeps — like [`Chain::new`](crate::chain::Chain::new), warmup is
    /// the caller's job via [`advance`](GpuClusterChain::advance).
    ///
    /// There is no batch size, unlike the checkerboard backends. A sweep already
    /// costs several device round-trips for its convergence checks, so batching
    /// the sample read-back on top of that would buy little and cost a buffer
    /// sized to the batch.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below two, if `start` is not a site field of this
    /// lattice, or — through [`ClusterUpdate::new`] — if the model's
    /// symmetry-breaking term rules the cluster move out or the composition
    /// itself is refused. Extents may be odd.
    #[expect(
        clippy::too_many_arguments,
        reason = "the run parameters mirror ClusterUpdate::new plus the device \
                  chain's own; a builder for one call site is not worth it"
    )]
    pub fn new<const D: usize, M: BondAction<Q>>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        model: &M,
        extent: Extent,
        relabel: Relabel,
        beta: f64,
        seed: u64,
        start: &Configuration<Q>,
        sweeps_between: usize,
    ) -> Self {
        assert!(
            Q >= MIN_STATES,
            "a cluster redraw picks among the Q states, which needs at least two"
        );
        // The kernels index the neighbor table by site, so a link field would be
        // silently misread rather than rejected.
        assert_eq!(
            start.cell(),
            Cell::Site,
            "the cluster update bonds nearest-neighbor sites, so the start must \
             be a site field"
        );
        assert_eq!(
            start.n_vars(),
            lattice.n_sites(),
            "start configuration and lattice disagree on site count"
        );

        // The symmetry, coupling, and composition guards, and the probability
        // formula, come from the CPU updater rather than being restated here —
        // a device path that opened its bonds by a different rule would be the
        // whole bug.
        let p = ClusterUpdate::new(model, extent, relabel).bond_probability(beta);

        let device = &gpu.device;
        let n_sites = lattice.n_sites();
        let n_bonds = lattice.n_links();
        let word = std::mem::size_of::<u32>() as u64;

        check_fits(
            device,
            n_sites as u64 * word,
            &format!("cluster label buffer ({n_sites} sites)"),
        );
        check_fits(
            device,
            n_bonds as u64 * word,
            &format!("cluster bond buffer ({n_bonds} bonds)"),
        );
        let neighbor_table = site_neighbor_table(lattice);
        check_fits(
            device,
            neighbor_table.len() as u64 * word,
            &format!("cluster neighbor table ({} entries)", neighbor_table.len()),
        );

        let params = Params {
            n_sites: n_sites as u32,
            n_bonds: n_bonds as u32,
            seed: fold_seed(seed),
            p: p as f32,
            n_states: Q as u32,
            seeded: u32::from(extent == Extent::Seeded),
            forced: u32::from(relabel == Relabel::ForcedChange),
            _pad: 0,
        };

        let labels = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster labels"),
            contents: bytemuck::cast_slice(&state_words(start)),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster neighbors"),
            contents: bytemuck::cast_slice(&neighbor_table),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cluster params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Neither of the next two needs initializing: the bond stage writes
        // every bond and the init stage writes every site, both before anything
        // reads them.
        let bonds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster bonds"),
            size: n_bonds as u64 * word,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let cluster = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster labels-of-cluster"),
            size: n_sites as u64 * word,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let changed = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster convergence flag"),
            size: FLAG_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let label_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster label staging"),
            size: n_sites as u64 * word,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let flag_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cluster flag staging"),
            size: FLAG_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("potts cluster"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cluster bind group layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cluster pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
        // Every stage is the same module and layout, differing only in entry
        // point; `D` reaches all four the same way it reaches the checkerboard
        // kernels, as an override resolved here so loop bounds stay constants.
        let stage = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("cluster {entry}")),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("D", f64::from(D as u32))],
                    ..Default::default()
                },
                cache: None,
            })
        };
        let bonds = stage("bonds_pass");
        let label_init = stage("label_init");
        let propagate = stage("propagate");
        let relabel = stage("relabel");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cluster bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: labels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: neighbors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bonds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cluster.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: changed.as_entire_binding(),
                },
            ],
        });

        let per_axis = device.limits().max_compute_workgroups_per_dimension;
        GpuClusterChain {
            bonds,
            label_init,
            propagate,
            relabel,
            bind_group,
            labels,
            changed,
            label_staging,
            flag_staging,
            _resources: vec![neighbors, params, bonds_buffer, cluster],
            n_sites,
            site_grid: grid_for(per_axis, n_sites),
            bond_grid: grid_for(per_axis, n_bonds),
            // The torus diameter: the farthest any two sites sit apart, and
            // the scale a label actually has to travel.
            pass_cap: pass_cap(n_sites, lattice.shape().iter().map(|l| l / 2).sum()),
            sweeps_between,
            sweeps_done: 0,
            gpu,
        }
    }

    /// Advance the chain by `sweeps` cluster sweeps on the device, producing no
    /// snapshot — the counterpart of
    /// [`Chain::advance`](crate::chain::Chain::advance), used to discard warmup.
    pub fn advance(&mut self, sweeps: usize) {
        for _ in 0..sweeps {
            self.run_sweep();
        }
    }

    /// One cluster sweep: place the bonds, label the components, relabel each
    /// selected component under the composed axes.
    ///
    /// # Panics
    ///
    /// Panics if the labeling has not converged within
    /// [`pass_cap`] passes. Proceeding on a partial labeling would split a
    /// cluster across two labels and sample the wrong distribution silently,
    /// which is much worse than stopping.
    fn run_sweep(&mut self) {
        let mut encoder = self.encoder("cluster sweep setup");
        self.dispatch(&mut encoder, &self.bonds, self.bond_grid, "bonds");
        self.dispatch(&mut encoder, &self.label_init, self.site_grid, "label init");
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let mut passes = 0usize;
        loop {
            let mut encoder = self.encoder("label propagation");
            encoder.clear_buffer(&self.changed, 0, None);
            for _ in 0..PASSES_PER_CHECK {
                self.dispatch(&mut encoder, &self.propagate, self.site_grid, "propagate");
            }
            encoder.copy_buffer_to_buffer(&self.changed, 0, &self.flag_staging, 0, FLAG_BYTES);
            self.gpu.queue.submit(std::iter::once(encoder.finish()));
            passes += PASSES_PER_CHECK;

            if self.read_word(&self.flag_staging) == 0 {
                break;
            }
            assert!(
                passes <= self.pass_cap,
                "cluster labeling did not converge in {passes} passes on \
                 {} sites; the cap already allows many times the lattice \
                 diameter, so this is a bug rather than a hard lattice",
                self.n_sites
            );
        }

        let mut encoder = self.encoder("relabel");
        self.dispatch(&mut encoder, &self.relabel, self.site_grid, "relabel");
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        self.sweeps_done += 1;
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Encode one stage. Every stage shares the bind group and the push
    /// constants; only the entry point and the launch width differ.
    fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        grid: (u32, u32),
        label: &str,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // The second word pads out the push-constant block the device was asked
        // for, which the checkerboard kernels fill with a color.
        let push: [u32; 2] = [self.sweeps_done, 0];
        pass.set_push_constants(0, bytemuck::bytes_of(&push));
        pass.dispatch_workgroups(grid.0, grid.1, 1);
    }

    /// Block until the queue drains and read the first `u32` out of a mapped
    /// staging buffer.
    fn read_word(&self, staging: &wgpu::Buffer) -> u32 {
        let slice = staging.slice(..FLAG_BYTES);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.gpu.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("map callback dropped")
            .expect("buffer map failed");
        let value = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range())[0];
        staging.unmap();
        value
    }

    /// Copy the labels back and rebuild a host configuration from them.
    fn read_configuration(&mut self) -> Configuration<Q> {
        let mut encoder = self.encoder("sample read-back");
        encoder.copy_buffer_to_buffer(
            &self.labels,
            0,
            &self.label_staging,
            0,
            self.n_sites as u64 * std::mem::size_of::<u32>() as u64,
        );
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.label_staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.gpu.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("map callback dropped")
            .expect("buffer map failed");
        let config = {
            let data = slice.get_mapped_range();
            let variables = bytemuck::cast_slice::<u8, u32>(&data)
                .iter()
                .map(|&v| State::new(v as usize).expect("a kernel writes only states in 0..Q"))
                .collect();
            Configuration::from_variables(Cell::Site, variables)
        };
        self.label_staging.unmap();
        config
    }
}

/// Propagation passes allowed before a sweep is declared broken.
///
/// The label still has to *reach* a site before pointer jumping can shorten
/// its path, and it travels one lattice bond per pass, so a converged
/// labeling needs about a cluster's chemical diameter of passes — near a
/// critical point that is the lattice's linear extent times a slowly growing
/// factor, not the logarithm the jumping alone would suggest. (A 128×128
/// Ising run at its critical coupling converges in roughly 80–140 passes,
/// where four logarithms allowed 92 and broke it.) Eight times the torus
/// diameter, plus a log-and-constant floor for the smallest lattices, is
/// generous enough that no correct run reaches it — the factor covers the
/// mildly superlinear chemical distance of critical clusters — and still
/// finite, so a genuinely stuck loop stops rather than spinning.
fn pass_cap(n_sites: usize, torus_diameter: usize) -> usize {
    // `BITS - leading_zeros` is `floor(log2(n)) + 1`, which is at least the
    // ceiling the bound is quoted in terms of.
    let log2_ceiling = (usize::BITS - n_sites.max(1).leading_zeros()) as usize;
    8 * torus_diameter + 4 * log2_ceiling + 32
}

impl<const Q: usize> Iterator for GpuClusterChain<Q> {
    type Item = Configuration<Q>;

    /// Decorrelate with `sweeps_between` cluster sweeps, then read the labels
    /// back. Always `Some`: the chain is open-ended, so callers bound it with
    /// `.take(n)`.
    fn next(&mut self) -> Option<Self::Item> {
        self.advance(self.sweeps_between);
        Some(self.read_configuration())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::require_gpu;
    use crate::models::potts::{Potts, potts_measure};
    use crate::rng::RandRng;

    /// With `sweeps_between = 0` a "sample" runs no sweeps, so the round-trip
    /// upload → device buffer → read-back must return the start configuration
    /// unchanged. This exercises the buffer plumbing in isolation from the
    /// sweep.
    #[test]
    fn uploads_and_reads_back_unchanged() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([4, 4]);
        let mut rng = RandRng::seed_from_u64(0);
        let start = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);

        let mut chain =
            GpuClusterChain::swendsen_wang(gpu, &lat, &Potts::symmetric(1.0), 0.5, 7, &start, 0);
        let got = chain.next().expect("open-ended stream yields");

        assert_eq!(got, start, "zero-sweep round-trip must be the identity");
    }

    /// At `beta = 0` the bond probability is zero, so every site is its own
    /// cluster and gets an independent fresh label.
    ///
    /// Two things are being pinned. Every label the relabel stage writes is in
    /// range — the read-back would panic on a word past `Q`, so reaching the
    /// assertions is most of it — and every label is reached, which says the
    /// draw is not quietly clamping to a subset. The label populations also have
    /// to come out even, which a relabel keyed on the wrong index would not
    /// manage.
    #[test]
    fn at_zero_beta_every_site_is_redrawn_independently() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        const Q: usize = 3;
        let lat = Lattice::new([16, 16]);
        let n_sites = lat.n_sites() as f64;
        let mut rng = RandRng::seed_from_u64(1);
        let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        let mut chain = GpuClusterChain::swendsen_wang(
            gpu,
            &lat,
            &Potts::<Q>::symmetric(1.0),
            0.0,
            99,
            &start,
            1,
        );
        for config in chain.by_ref().take(4) {
            let mut counts = [0usize; Q];
            for state in config.variables() {
                counts[state.index()] += 1;
            }
            assert!(counts.iter().all(|&c| c > 0), "not every label was reached");
            // 256 independent three-way draws put each fraction near a third
            // with a standard deviation of about 0.03, so this window is a few
            // of those wide and would not survive a stuck or biased draw.
            for (label, &count) in counts.iter().enumerate() {
                let fraction = count as f64 / n_sites;
                assert!(
                    (0.22..0.45).contains(&fraction),
                    "label {label} holds {fraction:.3} of the lattice"
                );
            }
        }
    }

    /// At large `beta` a uniform start is one cluster, so a sweep repaints the
    /// whole lattice in a single label and it stays uniform forever.
    ///
    /// This is what says the propagation really converged: a labeling that
    /// stopped early would leave the lattice in several clusters, each drawing
    /// its own label, and the order parameter would drop off one immediately.
    #[test]
    fn a_percolating_cluster_is_labeled_as_one() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        // Large enough that a diameter-many propagation would be visibly slow
        // and a capped one would fail outright.
        let lat = Lattice::new([32, 32]);
        let model = Potts::<3>::symmetric(1.0);
        let start = Configuration::<3>::cold(&lat, Cell::Site);

        let mut chain = GpuClusterChain::swendsen_wang(gpu, &lat, &model, 8.0, 99, &start, 1);
        for config in chain.by_ref().take(4) {
            let first = config.peek(0);
            assert!(
                config.variables().iter().all(|&s| s == first),
                "one cluster must land on one label"
            );
        }
    }

    /// An odd lattice runs, which is the one shape rule this backend does not
    /// share with the checkerboard one. There is no coloring here to collide
    /// under the periodic wrap, only a graph.
    #[test]
    fn runs_on_odd_extents() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([5, 7, 3]);
        let model = Potts::<3>::symmetric(1.0);
        let mut rng = RandRng::seed_from_u64(5);
        let start = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);

        let mut chain = GpuClusterChain::swendsen_wang(gpu, &lat, &model, 0.5, 4242, &start, 2);
        chain.advance(10);
        let config = chain.next().expect("open-ended stream yields");
        assert_eq!(config.n_vars(), lat.n_sites());
    }

    /// Run the same physics through the host and the device cluster paths and
    /// compare the mean energy density and order parameter.
    ///
    /// A distributional check, not a bit-for-bit one: the host draws from a
    /// stream in bond order and the device from a counter keyed on the bond, so
    /// the two consume randomness differently by construction. `ordered` is the
    /// window both means must sit inside, which is what stops the comparison
    /// passing because two backends agree on something wrong.
    ///
    /// Both backends run the composition named by the axes, so the same
    /// harness compares the Swendsen–Wang and the Wolff pairs. The counts are
    /// per composition: a Wolff sweep is one cluster rather than a lattice
    /// pass, so its runs thermalize and decorrelate over more sweeps.
    fn compare_backends<const D: usize>(
        gpu: Gpu,
        shape: [usize; D],
        beta: f64,
        extent: Extent,
        relabel: Relabel,
        (thermalize, sweeps_between, n): (usize, usize, usize),
    ) {
        const Q: usize = 3;
        let lat = Lattice::new(shape);
        let n_sites = lat.n_sites() as f64;
        let model = Potts::<Q>::symmetric(1.0);

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

        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let updater = ClusterUpdate::new(&model, extent, relabel);
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

        let (e_gpu, m_gpu) = {
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = GpuClusterChain::new(
                gpu,
                &lat,
                &model,
                extent,
                relabel,
                beta,
                12345,
                &start,
                sweeps_between,
            );
            chain.advance(thermalize);
            means(chain.take(n).collect())
        };

        eprintln!(
            "{extent:?}/{relabel:?} q = 3 on {shape:?} at beta = {beta}: \
             cpu ({e_cpu:.4}, {m_cpu:.4}), gpu ({e_gpu:.4}, {m_gpu:.4})"
        );
        // Both sides must land on the physics rather than merely on each other:
        // a kernel that pinned the field or randomized it would fail here even
        // if the host somehow agreed.
        assert!(
            (0.5..0.99).contains(&m_cpu) && (0.5..0.99).contains(&m_gpu),
            "{shape:?}: order parameter left the ordered-but-fluctuating range: \
             cpu {m_cpu:.4}, gpu {m_gpu:.4}"
        );
        assert!(
            (e_cpu - e_gpu).abs() < 0.05,
            "{shape:?}: energy density mismatch: cpu {e_cpu:.4} vs gpu {e_gpu:.4}"
        );
        assert!(
            (m_cpu - m_gpu).abs() < 0.05,
            "{shape:?}: order parameter mismatch: cpu {m_cpu:.4} vs gpu {m_gpu:.4}"
        );
    }

    /// The device cluster sweep samples the same distribution the host one does,
    /// in two dimensions, at a coupling just inside the ordered phase where the
    /// order parameter is off its ceiling and a wrong bond probability still has
    /// room to show.
    #[test]
    fn matches_the_cpu_cluster_distribution() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        // beta_c = ln(1 + sqrt(3)) ~ 1.005
        compare_backends(
            gpu,
            [16, 16],
            1.1,
            Extent::All,
            Relabel::Redraw,
            (100, 2, 200),
        );
    }

    /// The same comparison in three dimensions, which is what the `D`-dependent
    /// indexing needs and the test above cannot give it.
    ///
    /// A bond is addressed `site * D + axis`, and a stride written as a literal
    /// `2` rather than taken from the override is exactly the mistake a
    /// two-dimensional test cannot see: it is the *same arithmetic* there.
    /// Substituting it and re-running leaves the run above passing unchanged and
    /// moves this one's energy density by more than four times the tolerance, so
    /// this test is the only thing standing between that error and a green
    /// suite.
    ///
    /// The other half of the walk is caught in either dimension, which is worth
    /// recording so it is not mistaken for this test's job. Propagation reaches a
    /// site's *backward* neighbor through that neighbor's own forward bond,
    /// `backward * D + axis`; dropping that branch fragments every cluster that
    /// does not wrap the lattice and collapses the order parameter to about
    /// `0.05` in both dimensions.
    ///
    /// The coupling sits on the ordered side of the three-dimensional transition
    /// near `beta_c ~ 0.55` and clear of it, since that transition is first order
    /// even at `q = 3` and a short run should not be deciding between coexisting
    /// phases.
    #[test]
    fn matches_the_cpu_cluster_distribution_in_three_dimensions() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        compare_backends(
            gpu,
            [6, 6, 6],
            0.65,
            Extent::All,
            Relabel::Redraw,
            (100, 2, 200),
        );
    }

    /// The device Wolff sweep samples the same distribution the host Wolff
    /// updater does — the two seeded extents are different constructions (a
    /// frontier growth against a filtered full decomposition), so their
    /// agreement is a real check on both, not a shared-code tautology. One
    /// sweep is one cluster, hence the larger thermalization and stride.
    #[test]
    fn matches_the_cpu_wolff_distribution() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        compare_backends(
            gpu,
            [16, 16],
            1.1,
            Extent::Seeded,
            Relabel::ForcedChange,
            (600, 8, 200),
        );
    }

    /// At very large `beta` a uniform start is one cluster holding every site,
    /// so a device Wolff sweep repaints the whole lattice onto a different
    /// label every time — uniform after each sweep, never the label it had.
    ///
    /// This pins the two Wolff-specific shader branches at once: the seeded
    /// filter keeps the (single) seed cluster rather than dropping everything,
    /// and the forced change never lands on the current label, which at
    /// `Q = 2` a redraw would half the time.
    #[test]
    fn a_device_wolff_sweep_repaints_a_uniform_lattice() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([8, 8]);
        let model = Potts::<2>::symmetric(1.0);
        let start = Configuration::<2>::cold(&lat, Cell::Site);

        let mut chain = GpuClusterChain::wolff(gpu, &lat, &model, 8.0, 3, &start, 1);
        let mut previous = start.peek(0);
        for config in chain.by_ref().take(6) {
            let first = config.peek(0);
            assert!(
                config.variables().iter().all(|&s| s == first),
                "one cluster must land on one label"
            );
            assert_ne!(first, previous, "the forced change must change the label");
            previous = first;
        }
    }

    /// At `beta = 0` no bond opens, so the seeded cluster is the seed alone
    /// and a device Wolff sweep moves exactly one site — the opposite limit,
    /// and the one that catches a seeded filter matching more than the seed's
    /// cluster.
    #[test]
    fn at_zero_beta_a_device_wolff_sweep_moves_one_site() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([8, 8]);
        let model = Potts::<3>::symmetric(1.0);
        let mut rng = RandRng::seed_from_u64(2);
        let start = Configuration::<3>::hot(&lat, Cell::Site, &mut rng);

        let mut chain = GpuClusterChain::wolff(gpu, &lat, &model, 0.0, 17, &start, 1);
        let mut previous = start;
        for config in chain.by_ref().take(6) {
            let changed = (0..lat.n_sites())
                .filter(|&s| config.peek(s) != previous.peek(s))
                .count();
            assert_eq!(changed, 1, "only the seed site may move at p = 0");
            previous = config;
        }
    }

    /// A model with per-label offsets is refused, through the same construction
    /// guard the CPU updater carries — the device path does not get its own
    /// wording or its own rule.
    #[test]
    #[should_panic(expected = "invariant under relabeling")]
    fn rejects_a_model_with_offsets() {
        let Some(gpu) = require_gpu() else {
            // The assertion under test fires before any device work, but the
            // constructor needs a device to reach it.
            panic!(
                "no GPU adapter: the cluster update needs a model invariant under relabeling (skipped)"
            );
        };
        let lat = Lattice::new([4, 4]);
        let start = Configuration::<3>::cold(&lat, Cell::Site);
        let _ = GpuClusterChain::swendsen_wang(
            gpu,
            &lat,
            &Potts::<3>::new(1.0, [0.5, 0.0, 0.0]),
            0.5,
            1,
            &start,
            1,
        );
    }

    /// A large lattice at the Ising critical coupling labels within the pass
    /// cap. This is the case the old four-logarithm cap broke — near
    /// criticality a label travels roughly the cluster's chemical diameter,
    /// which scales with the lattice's linear extent, and a 128×128 sweep
    /// needs on the order of a hundred passes where the old cap allowed 92.
    #[test]
    fn labeling_converges_on_a_large_critical_lattice() {
        let Some(gpu) = require_gpu() else {
            return;
        };
        let lat = Lattice::new([128, 128]);
        // The two-state symmetric Potts model is the Ising model at 2J; its
        // critical coupling in these conventions sits near 0.88.
        let model = Potts::<2>::symmetric(1.0);
        let mut rng = RandRng::seed_from_u64(20260718);
        let start = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        let mut chain = GpuClusterChain::swendsen_wang(gpu, &lat, &model, 0.88, 7, &start, 2);
        chain.advance(6);
        let config = chain.next().expect("open-ended stream yields");
        assert_eq!(config.n_vars(), lat.n_sites());
    }

    /// The pass cap grows with the lattice's diameter rather than only its
    /// logarithm, and leaves room above what a critical run actually needs.
    ///
    /// The concrete anchor: a 128×128 run at the Ising critical coupling was
    /// observed to need up to ~140 passes, so the cap for that shape must sit
    /// comfortably above that — the old four-logarithm cap (92) is the bug
    /// this pins against.
    #[test]
    fn the_pass_cap_leaves_room_above_the_diameter() {
        assert!(pass_cap(1, 0) >= 32);
        // 128 x 128: torus diameter 128, observed need ~140.
        assert!(
            pass_cap(128 * 128, 128) > 4 * 140,
            "cap {} too tight",
            pass_cap(128 * 128, 128)
        );
        assert!(
            pass_cap(1 << 20, 1024) < pass_cap(1 << 30, 32 * 1024),
            "the cap must grow"
        );
    }
}
