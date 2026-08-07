//! Device plumbing shared by every GPU backend: acquiring a device, and driving
//! a color-pass sweep in batches.
//!
//! What lives here is everything about running a checkerboard on a GPU that is
//! *not* about a particular model. [`Gpu`] owns the device and queue.
//! `DeviceSweeper` owns the loop: encode `colors` dispatches per sweep, run
//! `sweeps_between` of them per sample, copy a whole batch back in one transfer,
//! and hand out [`Configuration`]s until the batch drains.
//!
//! Both model backends —
//! [`GpuIsingChain`](crate::ising_gpu::GpuIsingChain) and
//! [`GpuGaugeChain`](crate::gauge_gpu::GpuGaugeChain) — are a constructor plus a
//! `DeviceSweeper`. They differ in what a color *is* (site parity, or direction
//! and base-site parity), how many there are, which shader prices a flip, and
//! which lattice grade the variables sit on; they do not differ in any of the
//! batching. Keeping that in one place is what stops a fix to the read-back path
//! landing in one backend and silently not the other.
//!
//! The batching is the reason a device round-trip is cheap: the configuration
//! stays in a device buffer and crosses the host boundary once per *batch*, not
//! once per sample. `batch` trades the per-sample launch and transfer overhead
//! against holding that many configurations at once.
//!
//! `wgpu`'s setup is async; the async calls are driven with `pollster::block_on`
//! at construction so the public API stays synchronous.

use std::collections::VecDeque;

use wgpu::util::DeviceExt;

use crate::configuration::{Cell, Configuration};
use crate::state::State;

/// An initialized GPU device and its command queue.
///
/// Owns the two handles every later step needs: the [`Device`](wgpu::Device) that
/// allocates buffers and compiles shaders, and the [`Queue`](wgpu::Queue) that
/// submits work.
pub struct Gpu {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

impl Gpu {
    /// Acquire a compute-capable device, or `None` if no adapter is available
    /// (a headless machine, or one without a supported backend).
    ///
    /// Blocks on `wgpu`'s async setup so callers stay synchronous. Requests the
    /// push-constant feature, which `DeviceSweeper` uses to pass the
    /// per-dispatch sweep index and color.
    pub fn new() -> Option<Self> {
        pollster::block_on(Self::new_async())
    }

    async fn new_async() -> Option<Self> {
        // Built from the environment so `WGPU_BACKEND` can pin the backend, which
        // both aids debugging and gives the tests a way to simulate a machine
        // with no adapter. Unset, it enables every backend, as the default does.
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok()?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("plaquette compute device"),
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: PUSH_CONSTANT_BYTES,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    }
}

/// Threads per workgroup, matching the `@workgroup_size(64)` both shaders
/// declare.
const WORKGROUP_SIZE: u32 = 64;

/// Roughly how many compute passes one command buffer may encode.
///
/// The bound that matters is passes per submission, not sweeps: a sweep is
/// `colors` passes, so a six-color gauge sweep fills a command buffer three
/// times faster than a two-color Ising one. Deriving the sweep chunk from this
/// is what keeps the two backends' submission sizes comparable without each
/// picking its own unexplained constant.
const MAX_PASSES_PER_SUBMIT: usize = 512;

/// The preamble every checkerboard shader is compiled with: the state encoding
/// and the counter-based random source, which must be byte-identical across
/// backends. WGSL has no include directive, so the concatenation happens here.
macro_rules! shader_source {
    ($model:literal) => {
        concat!(
            include_str!("checkerboard_prelude.wgsl"),
            include_str!($model)
        )
    };
}
pub(crate) use shader_source;

/// Bytes of push constants a sweep dispatch carries: `(sweep, color)` as two
/// `u32`, matching the `Push` struct both shaders declare. It is also the limit
/// [`Gpu::new`] requests, so raising it means raising that too.
const PUSH_CONSTANT_BYTES: u32 = 8;

/// Everything a model backend must decide before a sweep can run.
///
/// Grouped into one struct rather than passed loose because the list is long and
/// positional: the alternative is a thirteen-argument constructor where `n_vars`
/// and `threads` are both counts and nothing but their order says which is which.
pub(crate) struct SweepSetup<'a> {
    /// Prefix for the device object labels, e.g. `"gauge checkerboard"`.
    pub label: &'a str,
    /// WGSL source, which must expose a `sweep` entry point taking the
    /// `(sweep, color)` push constants.
    pub shader: &'a str,
    /// Initial variables, one `u32` (0 or 1) each, in lattice index order.
    pub vars_init: &'a [u32],
    /// The read-only lookup table the shader prices a flip against — neighbors
    /// for a site kernel, staples for a link one.
    pub table: &'a [u32],
    /// The model's uniform block, already laid out for WGSL.
    pub params: &'a [u8],
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
/// Everything model-specific — the shader, the lookup table, what a color means
/// — is decided by the caller and arrives in a [`SweepSetup`]; this type knows
/// only how many colors there are, how wide to dispatch, and how to get a batch
/// back to the host. It deliberately does *not* keep the lattice: rebuilding a
/// configuration needs a length and a cell kind, both of which it stores, and
/// holding the geometry instead would pin a copy of the staple table — 75 MB on
/// a 64³ gauge run — for the chain's whole lifetime.
pub(crate) struct DeviceSweeper {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// The evolving configuration, one `u32` (0 or 1) per variable.
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
    workgroups: u32,
    /// Global sweep counter — the RNG key, so every sweep draws differently.
    sweeps_done: u32,
    /// Host-side buffer of the current batch; `next` drains it, refilling on empty.
    buffer: VecDeque<Configuration<2>>,
}

impl DeviceSweeper {
    /// Upload the buffers, compile the shader, build the pipeline, and assemble
    /// a sweeper over the lot.
    ///
    /// The bind group is the same three slots for every backend — the variables
    /// read-write, the lookup table read-only, the uniform block — which is what
    /// lets one builder serve both. A backend that needed a fourth slot would
    /// have outgrown this, and should say so by not using it rather than by
    /// widening the layout for everyone.
    pub(crate) fn build(gpu: Gpu, setup: SweepSetup<'_>) -> Self {
        let SweepSetup {
            label,
            shader,
            vars_init,
            table,
            params,
            cell,
            n_vars,
            threads,
            colors,
            sweeps_between,
            batch,
        } = setup;
        let device = &gpu.device;

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
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{label} params")),
            contents: params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} staging")),
            size: (n_vars * batch * std::mem::size_of::<u32>()) as u64,
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
            compilation_options: Default::default(),
            cache: None,
        });
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
            ],
        });

        DeviceSweeper {
            gpu,
            pipeline,
            bind_group,
            vars,
            staging,
            _resources: vec![table, params],
            cell,
            n_vars,
            colors,
            sweeps_between,
            batch,
            workgroups: (threads as u32).div_ceil(WORKGROUP_SIZE),
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

    /// How many sweeps fit in one command buffer under the pass budget, at least
    /// one. The budget is in passes rather than sweeps because a sweep is
    /// `colors` of them, so a six-color gauge sweep fills a command buffer three
    /// times faster than a two-color Ising one.
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
            pass.dispatch_workgroups(self.workgroups, 1, 1);
        }
    }

    /// Run one batch: `batch` samples, each after `sweeps_between` sweeps, copying
    /// each into its staging slot; then map the whole batch back in one transfer
    /// and build the host configurations. Fills `self.buffer`.
    fn run_batch(&mut self) {
        let n = self.n_vars;
        let stride = (n * std::mem::size_of::<u32>()) as u64;

        // One transfer per batch, but not necessarily one submission: the same
        // pass budget `advance` respects applies here, and a batch of 64 samples
        // at a stride of 10 sweeps would otherwise encode several thousand passes
        // into a single command buffer. Submissions stay ordered, so splitting
        // one batch across several is invisible to the result.
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

        // Map the batch back in one transfer and build the configurations inside
        // the mapped scope, so the words are read once rather than copied into an
        // owned buffer first.
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
                    .map(|&v| State::new(v as usize).expect("a GPU variable is always 0 or 1"))
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
    pub(crate) fn next_sample(&mut self) -> Option<Configuration<2>> {
        if self.buffer.is_empty() {
            self.run_batch();
        }
        self.buffer.pop_front()
    }
}

/// How many samples a GPU run produces per device round-trip.
///
/// A performance knob, not a physics one — the samples are identical regardless
/// — so it is a default here rather than a config field, and it is a property of
/// the device round-trip rather than of either model.
pub(crate) const GPU_BATCH: usize = 64;

/// Assert every extent is even, the precondition a parallel color pass carries.
///
/// Under periodic boundaries an odd extent wraps a variable onto a same-color
/// one, so a color stops being collision-free and detailed balance breaks
/// silently. `what` names the variables for the message. A sequential CPU
/// schedule has no such requirement — see `docs/metropolis.md`.
///
/// The config schemas check the same thing through
/// [`check_updater`](crate::config::check_updater) so a bad shape is a load-time
/// error; this is the guard for the constructors, which are reachable with no
/// config at all.
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
/// Skipping keeps the suite green on a machine without a GPU, but it would also
/// hide a driver that failed to load where a device is expected, since a skipped
/// test and a passing one look alike. Setting `PLAQUETTE_REQUIRE_GPU` — as CI
/// does, alongside a software Vulkan driver — turns a missing adapter into a
/// failure instead.
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
