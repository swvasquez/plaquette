//! Device plumbing shared by every GPU backend: acquiring a device, and driving
//! a color-pass sweep in batches.
//!
//! [`Gpu`] owns the device and queue. [`GpuChain`] is the one chain type every
//! model runs on: a shader assembled from four fragments — prelude, model
//! snippet, kernel, schedule — mirroring the CPU composition of
//! [`LocalUpdate`](crate::updater::LocalUpdate), driven by a `DeviceSweeper`
//! that owns the loop: encode `colors` dispatches per sweep, run
//! `sweeps_between` of them per sample, copy a whole batch back in one
//! transfer, and hand out [`Configuration`]s until the batch drains. Each
//! model backend is a constructor (`gpu_chain` in its module) that builds a
//! `GpuModelSetup` — the model's tables, params, and `model.wgsl` snippet —
//! and everything else is shared.
//!
//! Neither the batching nor any kernel knows the lattice dimension: site parity
//! arrives as an uploaded table, and `D` itself is a WGSL `override` resolved at
//! pipeline build, so compiled kernels see literals. The configuration stays in
//! a device buffer and crosses the host boundary once per *batch*, not per
//! sample. `wgpu`'s async setup is driven with `pollster::block_on` at
//! construction so the public API stays synchronous.

use std::collections::VecDeque;

use wgpu::util::DeviceExt;

use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::state::State;

/// An initialized GPU device and its command queue.
pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl Gpu {
    /// Acquire a compute-capable device, or `None` if no adapter is available.
    ///
    /// Blocks on `wgpu`'s async setup so callers stay synchronous. Requests the
    /// push-constant feature, which `DeviceSweeper` uses to pass the
    /// per-dispatch sweep index and color.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        // From the environment so `WGPU_BACKEND` can pin the backend — for
        // debugging, and so tests can simulate a machine with no adapter.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;
        // Everything the adapter offers, not `Limits::default()`: the defaults
        // are the WebGPU browser baseline (a 128 MiB storage binding among
        // them), which the lookup tables outgrow quickly since they scale with
        // volume and dimension. Requesting the adapter's own limits always
        // succeeds. The cost is browser portability — revisit if a `wasm`
        // target ever matters.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("plaquette compute device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: PUSH_CONSTANT_BYTES,
                    ..adapter.limits()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    }
}

/// Threads per workgroup, matching the `@workgroup_size(64)` the schedule
/// fragments and cluster kernels declare and the `WORKGROUP_SIZE` in
/// `wgsl/rng.wgsl`.
const WORKGROUP_SIZE: u32 = 64;

/// How wide and how tall to launch, to cover `threads` threads under a cap of
/// `per_axis` workgroups per dispatch axis.
///
/// A dispatch is capped at `max_compute_workgroups_per_dimension` *per axis* —
/// an API-level ceiling (65535 on essentially every adapter) that requesting
/// the adapter's limits does not move. Past one row the launch becomes a
/// rectangle and the shader folds the two axes back into one index
/// (`linear_index` in the preamble); rows are filled completely, so the mapping
/// stays contiguous and the last row's leftover threads fall out on the
/// kernel's own bounds check. The cap is a parameter so the arithmetic can be
/// tested without a device.
pub(crate) fn grid_for(per_axis: u32, threads: usize) -> (u32, u32) {
    let total = (threads as u64).div_ceil(u64::from(WORKGROUP_SIZE)).max(1);
    let width = total.min(u64::from(per_axis));
    let height = total.div_ceil(width);
    assert!(
        height <= u64::from(per_axis),
        "a dispatch of {threads} threads needs {height} rows of workgroups, \
         above this device's limit of {per_axis} per axis"
    );
    (width as u32, height as u32)
}

/// Panic unless a buffer of `bytes` can be created and bound on `device`.
///
/// The two limits differ — a buffer may be larger than any one binding of it —
/// and both are checked, since every buffer here is bound whole. `what` names
/// the table, because a wgpu validation failure names only a byte count and a
/// binding index, which a caller cannot turn back into a lattice.
pub(crate) fn check_fits(device: &wgpu::Device, bytes: u64, what: &str) {
    let limits = device.limits();
    let binding = u64::from(limits.max_storage_buffer_binding_size);
    let buffer = limits.max_buffer_size;
    assert!(
        bytes <= binding && bytes <= buffer,
        "the {what} needs {:.1} MB, above this device's limit of {:.1} MB \
         per storage binding and {:.1} MB per buffer; \
         reduce the lattice extents or the dimension",
        bytes as f64 / 1e6,
        binding as f64 / 1e6,
        buffer as f64 / 1e6,
    );
}

/// Roughly how many compute passes one command buffer may encode.
///
/// The budget is in passes, not sweeps: a sweep is `colors` passes, so a
/// six-color gauge sweep fills a command buffer three times faster than a
/// two-color Ising one. Deriving the sweep chunk from this keeps the backends'
/// submission sizes comparable.
const MAX_PASSES_PER_SUBMIT: usize = 512;

/// The kernel enum is the same one the CPU updaters compose with — the choice
/// of single-variable rule is backend-neutral, so it is defined once in
/// [`updater`](crate::updater) and re-exported here for the device backends
/// that pick a shader by it.
pub use crate::updater::Kernel;

/// Bytes of push constants a sweep dispatch carries: `(sweep, color)` as two
/// `u32`, matching the `Push` struct the shared prelude declares. It is
/// also the limit [`Gpu::new`] requests, so raising it means raising that too.
/// The cluster kernels use only the first word and pad out the second, rather
/// than asking the device for a second push-constant size.
pub(crate) const PUSH_CONSTANT_BYTES: u32 = 8;

/// Everything a model backend must decide before a sweep can run.
pub(crate) struct SweepSetup<'a> {
    /// Prefix for the device object labels, e.g. `"gauge checkerboard"`.
    pub label: &'a str,
    /// WGSL source, which must expose a `sweep` entry point taking the
    /// `(sweep, color)` push constants.
    pub shader: &'a str,
    /// Initial variables, one `u32` state index each, in lattice index order.
    pub vars_init: &'a [u32],
    /// The read-only lookup table the shader prices a move against — neighbors
    /// for a site kernel, staples for a link one.
    pub table: &'a [u32],
    /// Each site's coordinate-sum parity, one `u32` per site, from
    /// [`Lattice::site_parity`](crate::lattice::Lattice::site_parity). Kernels
    /// test a thread's color against this rather than recomputing it, which
    /// keeps them free of per-axis arithmetic.
    pub site_color: &'a [u32],
    /// The model's uniform block, already laid out for WGSL.
    pub params: &'a [u8],
    /// The lattice dimension, supplied to the shader as the `D` override.
    pub dimension: u32,
    /// Which lattice cell the variables sit on, so a read-back configuration
    /// knows what its indices mean.
    pub cell: Cell,
    /// How many variables the field holds — the configuration length and the
    /// staging stride.
    pub n_vars: usize,
    /// Dispatch width. *Not* `n_vars`: the gauge kernel launches one thread per
    /// site and updates one link per thread, so it dispatches a third of them.
    pub threads: usize,
    /// Color passes per sweep: `2` for site parity, `2 * D` for a link kernel
    /// splitting by direction as well.
    pub colors: u32,
    /// Decorrelation sweeps between recorded samples.
    pub sweeps_between: usize,
    /// Samples produced per device round-trip.
    pub batch: usize,
}

/// Drives a color-pass sweep on the device and yields sampled configurations in
/// batches.
///
/// Everything model-specific arrives in a [`SweepSetup`]. It deliberately does
/// *not* keep the lattice: rebuilding a configuration needs only a length and a
/// cell kind, and holding the geometry would pin a copy of the staple table —
/// 75 MB on a 64³ gauge run — for the chain's whole lifetime. `Q` is carried
/// only so the read-back knows what a device word means; nothing in the
/// batching reads a variable's value.
pub(crate) struct DeviceSweeper<const Q: usize> {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// The evolving configuration, one `u32` state index per variable.
    vars: wgpu::Buffer,
    /// Read-back target: `batch` configurations, filled per run and mapped once.
    staging: wgpu::Buffer,
    /// Buffers reached through `bind_group`; held only to keep them alive.
    _resources: Vec<wgpu::Buffer>,

    cell: Cell,
    n_vars: usize,
    colors: u32,
    sweeps_between: usize,
    batch: usize,
    /// Workgroups to launch, as a `(width, height)` grid — see `grid_for`.
    dispatch: (u32, u32),
    /// Global sweep counter — the RNG key, so every sweep draws differently.
    sweeps_done: u32,
    /// Host-side buffer of the current batch; `next` drains it, refilling on empty.
    buffer: VecDeque<Configuration<Q>>,
}

impl<const Q: usize> DeviceSweeper<Q> {
    /// Upload the buffers, compile the shader, build the pipeline, and assemble
    /// a sweeper over the lot.
    ///
    /// The bind group is the same four slots for every backend — variables
    /// read-write, lookup table read-only, uniform block, color table
    /// read-only. A backend needing a fifth has outgrown this builder; don't
    /// widen the layout for everyone.
    pub(crate) fn build(gpu: Gpu, setup: SweepSetup<'_>) -> Self {
        let SweepSetup {
            label,
            shader,
            vars_init,
            table,
            site_color,
            params,
            dimension,
            cell,
            n_vars,
            threads,
            colors,
            sweeps_between,
            batch,
        } = setup;
        let device = &gpu.device;

        // Size checks before anything is created, so a limit surfaces as a
        // message naming the table rather than a wgpu validation failure.
        debug_assert_eq!(
            vars_init.len(),
            n_vars,
            "the initial variables and the declared count must agree"
        );
        let word = std::mem::size_of::<u32>() as u64;
        let sample_bytes = n_vars as u64 * word;
        check_fits(
            device,
            sample_bytes,
            &format!("{label} variable buffer ({n_vars} variables)"),
        );
        check_fits(
            device,
            table.len() as u64 * word,
            &format!("{label} table ({} entries)", table.len()),
        );

        // The batch is a performance knob, not a physics one, so a staging
        // buffer that will not fit is answered by holding fewer samples per
        // round trip rather than by refusing the run. One slot is the variable
        // buffer's size, already checked above, so the clamp is all the staging
        // buffer needs.
        let batch =
            batch.min((device.limits().max_buffer_size / sample_bytes.max(1)).max(1) as usize);

        let vars = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} variables")),
            contents: bytemuck::cast_slice(vars_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let table = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} table")),
            contents: bytemuck::cast_slice(table),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let site_color = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} color table")),
            contents: bytemuck::cast_slice(site_color),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} params")),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} staging")),
            size: sample_bytes * batch as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{label} bind group layout")),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
                storage_entry(3, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{label} pipeline layout")),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..PUSH_CONSTANT_BYTES,
            }],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{label} sweep")),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("sweep"),
            // The one place the lattice dimension reaches the device: the `D`
            // override, resolved here so kernel loop bounds are still constants
            // when the shader is translated.
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("D", f64::from(dimension))],
                ..Default::default()
            },
            cache: None,
        });
        let dispatch = grid_for(
            device.limits().max_compute_workgroups_per_dimension,
            threads,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} bind group")),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vars.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: site_color.as_entire_binding(),
                },
            ],
        });

        DeviceSweeper {
            gpu,
            pipeline,
            bind_group,
            vars,
            staging,
            _resources: vec![table, site_color, params],
            cell,
            n_vars,
            colors,
            sweeps_between,
            batch,
            dispatch,
            sweeps_done: 0,
            buffer: VecDeque::new(),
        }
    }

    /// Advance by `sweeps` sweeps on the device, producing no snapshot — the GPU
    /// counterpart of [`Chain::advance`](crate::chain::Chain::advance), used to
    /// discard warmup. Submits the work but does not wait on it; the next batch's
    /// reads are ordered after it on the queue.
    pub(crate) fn advance(&mut self, sweeps: usize) {
        let mut remaining = sweeps;
        while remaining > 0 {
            let this = remaining.min(self.sweeps_per_submit());
            let mut encoder = self.encoder("warmup");
            for _ in 0..this {
                self.encode_sweep(&mut encoder, self.sweeps_done);
                self.sweeps_done += 1;
            }
            self.gpu.queue.submit(std::iter::once(encoder.finish()));
            remaining -= this;
        }
    }

    /// How many sweeps fit in one command buffer under the pass budget, at
    /// least one — see [`MAX_PASSES_PER_SUBMIT`].
    fn sweeps_per_submit(&self) -> usize {
        (MAX_PASSES_PER_SUBMIT / self.colors as usize).max(1)
    }

    fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
        self.gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
    }

    /// Encode one sweep as `colors` color passes into `encoder`. Separate passes
    /// are the barrier, so each color reads the previous colors' *new* values.
    /// They all share `sweep_index`, so a variable draws once per sweep — the
    /// colors touch disjoint variables, so there is no collision.
    fn encode_sweep(&self, encoder: &mut wgpu::CommandEncoder, sweep_index: u32) {
        for color in 0..self.colors {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("color pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let push: [u32; 2] = [sweep_index, color];
            pass.set_push_constants(0, bytemuck::bytes_of(&push));
            let (width, height) = self.dispatch;
            pass.dispatch_workgroups(width, height, 1);
        }
    }

    /// Run one batch: `batch` samples, each after `sweeps_between` sweeps, copying
    /// each into its staging slot; then map the whole batch back in one transfer
    /// and build the host configurations. Fills `self.buffer`.
    fn run_batch(&mut self) {
        let n = self.n_vars;
        let stride = (n * std::mem::size_of::<u32>()) as u64;

        // One transfer per batch, but not necessarily one submission: the same
        // pass budget `advance` respects applies here. Submissions stay
        // ordered, so splitting a batch across several is invisible to the
        // result.
        let mut encoder = self.encoder("sample batch");
        let mut sweeps_encoded = 0;
        for k in 0..self.batch {
            for _ in 0..self.sweeps_between {
                if sweeps_encoded == self.sweeps_per_submit() {
                    self.gpu.queue.submit(std::iter::once(encoder.finish()));
                    encoder = self.encoder("sample batch");
                    sweeps_encoded = 0;
                }
                self.encode_sweep(&mut encoder, self.sweeps_done);
                self.sweeps_done += 1;
                sweeps_encoded += 1;
            }
            encoder.copy_buffer_to_buffer(&self.vars, 0, &self.staging, k as u64 * stride, stride);
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        // Build the configurations inside the mapped scope, so the words are
        // read once rather than copied into an owned buffer first.
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.gpu.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("map callback dropped")
            .expect("buffer map failed");
        {
            let data = slice.get_mapped_range();
            let words = bytemuck::cast_slice::<u8, u32>(&data);
            for chunk in words.chunks_exact(n).take(self.batch) {
                let variables = chunk
                    .iter()
                    .map(|&v| State::new(v as usize).expect("a kernel writes only states in 0..Q"))
                    .collect();
                self.buffer
                    .push_back(Configuration::from_variables(self.cell, variables));
            }
        }
        self.staging.unmap();
    }

    /// Yield the next sampled configuration, running a fresh batch on the device
    /// when the host-side buffer drains. Always `Some`: the chain is open-ended,
    /// so callers bound it with `.take(n)`.
    pub(crate) fn next_sample(&mut self) -> Option<Configuration<Q>> {
        if self.buffer.is_empty() {
            self.run_batch();
        }
        self.buffer.pop_front()
    }
}

/// How many samples a GPU run produces per device round-trip.
///
/// A performance knob, not a physics one — the samples are identical regardless
/// — so it is a default here rather than a config field.
pub(crate) const GPU_BATCH: usize = 64;

/// The token in the kernel fragments standing for the state count. Substituted
/// for *every* model — `Q` is a const generic on the host, so the number is
/// always known — because the heat bath's per-state arrays need a const
/// length and the Metropolis two-state branch should constant-fold.
const Q_TOKEN: &str = "$Q$";

/// Assemble a local-update shader from its four fragments: the shared prelude,
/// a model snippet, a kernel picked by `kernel`, and a schedule picked by
/// `cell` — the device mirror of composing a
/// [`LocalUpdate`](crate::updater::LocalUpdate) and handing it an action.
///
/// The model snippet must declare bindings 1 and 2 (its lookup table and a
/// `Params` uniform whose head is `n_sites: u32, seed: u32, beta: f32`) and
/// define `fn energy_delta(v: u32, current: u32, proposed: u32) -> f32` — the
/// WGSL half of [`Action::energy_delta`](crate::action::Action::energy_delta).
/// The structural contract is enforced when the pipeline is built: naga
/// rejects a snippet that misses it, with the offending name in the error.
///
/// One part of the contract naga cannot check: `energy_delta` at a variable
/// must read only variables of *other* colors under the schedule's coloring —
/// nearest neighbors on a site field, the links of the variable's own
/// plaquettes on a link field. The parallel color passes rest on that
/// independence, and a longer-ranged action (an improved gauge action with
/// rectangle terms, say) would sample the wrong distribution here while
/// passing every validation; it needs a finer coloring, which means a new
/// schedule fragment, not a workaround in the snippet.
pub(crate) fn assemble_shader(model_snippet: &str, kernel: Kernel, cell: Cell, q: usize) -> String {
    let kernel_src = match kernel {
        Kernel::Metropolis => include_str!("wgsl/metropolis.wgsl"),
        Kernel::HeatBath => include_str!("wgsl/heat_bath.wgsl"),
    };
    let schedule_src = match cell {
        Cell::Site => include_str!("wgsl/site_schedule.wgsl"),
        Cell::Link => include_str!("wgsl/link_schedule.wgsl"),
    };
    let shader = format!(
        "{}{}{}{}{}",
        include_str!("wgsl/rng.wgsl"),
        include_str!("wgsl/prelude.wgsl"),
        model_snippet,
        kernel_src.replace(Q_TOKEN, &q.to_string()),
        schedule_src
    );
    // A leftover token would reach naga as a parse error naming a line in a
    // shader no file contains; failing here names the actual mistake — a
    // snippet handed over with its own tokens unfilled, or a new fragment
    // token this function does not know.
    assert!(
        !shader.contains('$'),
        "assembled {} shader still carries an unfilled substitution token",
        match cell {
            Cell::Site => "site",
            Cell::Link => "link",
        }
    );
    shader
}

/// A model's device-side description: everything [`GpuChain`] needs that is
/// about the model rather than the schedule, the kernel, or the batching —
/// the host half of the seam whose WGSL half is the model snippet.
pub(crate) struct GpuModelSetup {
    /// Prefix for the device object labels, e.g. `"ising"`.
    pub label: &'static str,
    /// The model snippet, with any model-specific tokens already filled.
    pub source: String,
    /// The read-only lookup table `energy_delta` prices a move against —
    /// neighbors for a site model, staples for a link one.
    pub table: Vec<u32>,
    /// The model's uniform block, already laid out for WGSL, head first.
    pub params: Vec<u8>,
    /// Which lattice cell the variables sit on. Decides the schedule fragment,
    /// the color count, and what a device word means on read-back.
    pub cell: Cell,
}

/// A Markov chain run on the GPU — one type for every model, as
/// [`LocalUpdate`] is one type on the CPU.
///
/// A sibling of [`Chain`](crate::chain::Chain): same
/// `Iterator<Item = Configuration>` interface, device machinery underneath.
/// Owns everything it needs, so it borrows nothing and can be moved and driven
/// freely. Everything model-specific arrived in a `GpuModelSetup` at
/// construction; the type carries no dimension, because the lattice is read
/// once to build the tables and what survives is device buffers and counts.
///
/// [`LocalUpdate`]: crate::updater::LocalUpdate
pub struct GpuChain<const Q: usize> {
    sweeper: DeviceSweeper<Q>,
}

impl<const Q: usize> GpuChain<Q> {
    /// Build a chain on `gpu` over a copy of `start`, uploaded to the device.
    ///
    /// `start` is read only to upload it, so the host copy is untouched and
    /// the same configuration can seed a CPU run too. `kernel` picks which
    /// single-variable rule a thread runs — a choice of algorithm, not of
    /// physics. The schedule is not a parameter: a device sweep *is* the
    /// checkerboard, in the coloring `setup.cell` implies. Runs no sweeps —
    /// warmup is the caller's job via [`advance`](GpuChain::advance).
    ///
    /// # Panics
    ///
    /// Panics if `batch` is zero, if `start` is not a field on `setup.cell`
    /// with one variable per cell of `lattice`, or if any extent is odd —
    /// the precondition every parallel color pass carries.
    pub(crate) fn build<const D: usize>(
        gpu: Gpu,
        lattice: &Lattice<D>,
        setup: GpuModelSetup,
        kernel: Kernel,
        start: &Configuration<Q>,
        sweeps_between: usize,
        batch: usize,
    ) -> Self {
        assert!(batch > 0, "batch size must be positive");
        // The shader indexes its table by variable, so a field on the wrong
        // cell would be silently misread rather than rejected; the cell kind
        // is what makes that checkable at all.
        assert_eq!(
            start.cell(),
            setup.cell,
            "the {} device chain updates {:?} variables, so the start must be a \
             field on that cell",
            setup.label,
            setup.cell,
        );
        let n_vars = match setup.cell {
            Cell::Site => lattice.n_sites(),
            Cell::Link => lattice.n_links(),
        };
        assert_eq!(
            start.n_vars(),
            n_vars,
            "start configuration and lattice disagree on variable count"
        );
        let shape = lattice.shape();
        assert_even_extents(
            &shape,
            match setup.cell {
                Cell::Site => "site",
                Cell::Link => "link",
            },
        );

        let shader = assemble_shader(&setup.source, kernel, setup.cell, Q);
        let label = format!(
            "{} {}",
            setup.label,
            match kernel {
                Kernel::Metropolis => "checkerboard",
                Kernel::HeatBath => "heat bath",
            }
        );

        GpuChain {
            sweeper: DeviceSweeper::build(
                gpu,
                SweepSetup {
                    label: &label,
                    shader: &shader,
                    vars_init: &state_words(start),
                    table: &setup.table,
                    site_color: &site_colors(lattice),
                    params: &setup.params,
                    dimension: D as u32,
                    cell: setup.cell,
                    n_vars,
                    // One thread per site on either cell: a site pass owns one
                    // variable per thread, and a link pass owns the one link
                    // each site bases in the pass's direction.
                    threads: lattice.n_sites(),
                    colors: crate::updater::checkerboard_colors(setup.cell, D) as u32,
                    sweeps_between,
                    batch,
                },
            ),
        }
    }

    /// Advance the chain by `sweeps` sweeps on the device, producing no
    /// snapshot — the GPU counterpart of
    /// [`Chain::advance`](crate::chain::Chain::advance), used to discard
    /// warmup.
    pub fn advance(&mut self, sweeps: usize) {
        self.sweeper.advance(sweeps);
    }
}

impl<const Q: usize> Iterator for GpuChain<Q> {
    type Item = Configuration<Q>;

    /// Yield the next sampled configuration, running a fresh batch on the
    /// device when the host-side buffer drains. Always `Some`: the chain is
    /// open-ended, so callers bound it with `.take(n)`.
    fn next(&mut self) -> Option<Self::Item> {
        self.sweeper.next_sample()
    }
}

/// Assert every extent is even, the precondition a parallel color pass carries.
///
/// Under periodic boundaries an odd extent wraps a variable onto a same-color
/// one, so a color stops being collision-free and detailed balance breaks
/// silently — see `docs/metropolis.md`. The config schemas check the same thing
/// through [`check_updater`](crate::config::check_updater); this guards the
/// constructors, which are reachable with no config at all.
///
/// # Panics
///
/// Panics if any extent is odd.
pub(crate) fn assert_even_extents(shape: &[usize], what: &str) {
    assert!(
        shape.iter().all(|l| l.is_multiple_of(2)),
        "the parallel {what} checkerboard needs even lattice extents, got {shape:?}"
    );
}

/// A configuration's variables in the form the kernels read: one `u32` state
/// index each, in lattice index order.
///
/// This line says what a device word *is*; `DeviceSweeper`'s read-back path
/// reverses exactly this encoding.
pub(crate) fn state_words<const Q: usize>(config: &Configuration<Q>) -> Vec<u32> {
    config
        .variables()
        .iter()
        .map(|s| s.index() as u32)
        .collect()
}

/// Every site's neighbor row, flattened in site order — the table a site kernel
/// prices a move against.
///
/// The flat layout is a contract with the shader's `site * N_NEIGHBORS + d`
/// indexing, with the stride from
/// [`neighbor_stride`](crate::lattice::Lattice::neighbor_stride). The gauge
/// backend has its own table, built from staples instead.
pub(crate) fn site_neighbor_table<const D: usize>(lattice: &Lattice<D>) -> Vec<u32> {
    let mut table = Vec::with_capacity(lattice.n_sites() * Lattice::<D>::neighbor_stride());
    for site in 0..lattice.n_sites() {
        table.extend(lattice.site_neighbors(site).iter().map(|&nb| nb as u32));
    }
    table
}

/// Each site's checkerboard color, in the form the kernels read.
///
/// Uploaded rather than derived on the device: a kernel recomputing coordinate
/// arithmetic is what tied the shaders to a fixed dimension before.
pub(crate) fn site_colors<const D: usize>(lattice: &Lattice<D>) -> Vec<u32> {
    (0..lattice.n_sites())
        .map(|site| lattice.site_parity(site) as u32)
        .collect()
}

/// Fold a 64-bit seed into the 32-bit key the shaders' RNG takes, so two seeds
/// differing only in their high bits do not collide on the device.
pub(crate) fn fold_seed(seed: u64) -> u32 {
    (seed ^ (seed >> 32)) as u32
}

/// A read-write (`read_only = false`) or read-only storage-buffer layout entry.
pub(crate) fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A uniform-buffer layout entry.
pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A device for a GPU test, or `None` when this machine has no adapter.
///
/// Skipping keeps the suite green without a GPU but would also hide a driver
/// that failed to load where one is expected. Setting `PLAQUETTE_REQUIRE_GPU` —
/// as CI does, alongside a software Vulkan driver — turns a missing adapter
/// into a failure instead.
#[cfg(test)]
pub(crate) fn require_gpu() -> Option<Gpu> {
    match Gpu::new() {
        Some(gpu) => Some(gpu),
        None if std::env::var_os("PLAQUETTE_REQUIRE_GPU").is_some() => {
            panic!("PLAQUETTE_REQUIRE_GPU is set but no GPU adapter is available")
        }
        None => {
            eprintln!("no GPU adapter available; skipping GPU test");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every model snippet assembles with both kernels into a shader whose
    /// tokens are all filled and whose contract names are all present.
    ///
    /// This runs without a device, so it is the check a machine with no
    /// adapter still gets: token drift or a renamed contract function fails
    /// here as a unit test rather than only at pipeline build. Full naga
    /// validation still happens at `GpuChain::build`, which the GPU tests
    /// exercise wherever an adapter exists. The Potts substitution is
    /// restated here rather than imported because `shader_for` is private to
    /// the Potts backend; two vec4 slots stand in for any real state count.
    #[test]
    fn every_model_and_kernel_pairing_assembles() {
        let snippets: [(&str, String, Cell); 3] = [
            (
                "ising",
                include_str!("models/ising/model.wgsl").to_string(),
                Cell::Site,
            ),
            (
                "gauge",
                include_str!("models/gauge/model.wgsl").to_string(),
                Cell::Link,
            ),
            (
                "potts",
                include_str!("models/potts/model.wgsl").replace("$H_VECTORS$", "2"),
                Cell::Site,
            ),
        ];
        for (label, snippet, cell) in snippets {
            for kernel in [Kernel::Metropolis, Kernel::HeatBath] {
                let shader = assemble_shader(&snippet, kernel, cell, 5);
                for needle in ["fn energy_delta(", "fn update(", "fn sweep("] {
                    assert!(
                        shader.contains(needle),
                        "{label} with {kernel:?}: assembled shader lacks {needle}"
                    );
                }
            }
        }
    }

    /// The launch grid covers every thread, never overruns an axis, and only
    /// becomes two-dimensional once one row cannot hold the work.
    ///
    /// Checked against a small cap as well as the usual one, since the
    /// interesting behavior starts past `per_axis * WORKGROUP_SIZE` threads and
    /// no lattice small enough for a test reaches that at the real limit.
    #[test]
    fn the_dispatch_grid_covers_every_thread() {
        for per_axis in [4u32, 65535] {
            let row = u64::from(per_axis) * u64::from(WORKGROUP_SIZE);
            for threads in [
                0,
                1,
                63,
                64,
                65,
                row as usize - 1,
                row as usize,
                row as usize + 1,
                row as usize * 3 + 7,
            ] {
                let (width, height) = grid_for(per_axis, threads);
                assert!(width <= per_axis && height <= per_axis, "{threads} threads");
                let covered = u64::from(width) * u64::from(height) * u64::from(WORKGROUP_SIZE);
                assert!(
                    covered >= threads as u64,
                    "{threads} threads: grid {width}x{height} covers only {covered}"
                );
                // One row while one row is enough: a rectangle costs nothing but
                // it should not appear before it has to.
                if threads as u64 <= row {
                    assert_eq!(height, 1, "{threads} threads should fit in one row");
                }
            }
        }
    }
}
