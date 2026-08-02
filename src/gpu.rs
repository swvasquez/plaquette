//! GPU backend: a [`GpuChain`] that runs the checkerboard sweep on the GPU via
//! `wgpu`, exposed through the same `Iterator<Item = Configuration>` interface as
//! the CPU [`Chain`](crate::chain::Chain).
//!
//! `GpuChain` is a *sibling* of `Chain`, not a variant of it: they share only the
//! iterator interface. Where `Chain` borrows a host `Configuration` and mutates
//! it a sweep at a time, `GpuChain` owns its device resources and keeps the
//! configuration in a device buffer, crossing the host boundary only once per
//! *batch* of samples. Sweeps run entirely on the GPU (two color dispatches
//! each), a batch of `B` samples is copied back in one transfer, and `next`
//! yields from that host-side buffer, refilling when it drains. The batch size
//! trades the per-sample launch/transfer overhead against holding `B`
//! configurations at once.
//!
//! The checkerboard is compiled into the shader (`checkerboard.wgsl`), so
//! `GpuChain` does not use the [`Updater`](crate::updater::Updater) seam. Its
//! randomness is counter-based, keyed on `(seed, site, sweep)`, so the result is
//! independent of GPU thread order — the property that lets the CPU checkerboard
//! serve as a reference.
//!
//! `wgpu`'s setup is async; the async calls are driven with `pollster::block_on`
//! at construction so the public API stays synchronous.

use std::collections::VecDeque;

use wgpu::util::DeviceExt;

use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::state::State;

/// An initialized GPU device and its command queue.
///
/// Owns the two handles every later step needs: the [`Device`](wgpu::Device) that
/// allocates buffers and compiles shaders, and the [`Queue`](wgpu::Queue) that
/// submits work.
pub struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Gpu {
    /// Acquire a compute-capable device, or `None` if no adapter is available
    /// (a headless machine, or one without a supported backend).
    ///
    /// Blocks on `wgpu`'s async setup so callers stay synchronous. Requests the
    /// push-constant feature, which [`GpuChain`] uses to pass the per-dispatch
    /// sweep index and color.
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
                    max_push_constant_size: 8,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .ok()?;
        Some(Gpu { device, queue })
    }
}

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

/// The Markov chain run on the GPU, yielding sampled [`Configuration`]s.
///
/// A sibling of [`Chain`](crate::chain::Chain): same iterator interface, device
/// machinery underneath. Owns everything it needs, so it borrows nothing and can
/// be moved and driven freely. Fixed at `Q = 2`, `D = 2` (the checkerboard shader
/// is written for the 2D Ising model).
pub struct GpuChain {
    gpu: Gpu,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    /// The evolving configuration, one `u32` (0 or 1) per site.
    spins: wgpu::Buffer,
    /// Read-back target: `batch` configurations, filled per run and mapped once.
    staging: wgpu::Buffer,
    // Kept alive for the run; referenced through `bind_group`.
    _neighbors: wgpu::Buffer,
    _params: wgpu::Buffer,

    lattice: Lattice<2>,
    n_sites: usize,
    sweeps_between: usize,
    batch: usize,
    workgroups: u32,
    /// Global sweep counter — the RNG key, so every sweep draws differently.
    sweeps_done: u32,
    /// Host-side buffer of the current batch; `next` drains it, refilling on empty.
    buffer: VecDeque<Configuration<2>>,
}

impl GpuChain {
    /// Build a chain on `gpu` over a copy of `start`, uploaded to the device.
    ///
    /// `start` is read only to upload it, so the host copy is untouched and the
    /// same configuration can seed a CPU run too. `j`, `h`, `beta` are the Ising
    /// parameters; `seed` keys the counter-based RNG. `sweeps_between` is the
    /// decorrelation stride, and `batch` is how many samples are produced per
    /// device round-trip.
    ///
    /// Runs no sweeps — like [`Chain::new`](crate::chain::Chain::new), warmup is
    /// the caller's job via [`advance`](GpuChain::advance).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gpu: Gpu,
        lattice: &Lattice<2>,
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
            "the GPU checkerboard updates sites, so the start must be a site field"
        );
        assert_eq!(
            start.n_vars(),
            lattice.n_sites(),
            "start configuration and lattice disagree on site count"
        );
        // The parallel checkerboard needs even extents: under periodic boundaries
        // an odd extent wraps a site onto a same-colour neighbour, so a colour is
        // no longer collision-free and detailed balance breaks silently. (The CPU
        // checkerboard, run in sequence, has no such constraint — see
        // docs/metropolis.md.)
        let shape = lattice.shape();
        assert!(
            shape[0] % 2 == 0 && shape[1] % 2 == 0,
            "GPU checkerboard needs even lattice extents, got {shape:?}"
        );

        let n_sites = lattice.n_sites();
        let width = shape[0] as u32;

        let spins_init: Vec<u32> = start.variables().iter().map(|s| s.index() as u32).collect();
        let mut nbrs: Vec<u32> = Vec::with_capacity(n_sites * 4);
        for site in 0..n_sites {
            for &nb in lattice.site_neighbors(site) {
                nbrs.push(nb as u32);
            }
        }
        let params = Params {
            n_sites: n_sites as u32,
            width,
            // Fold the 64-bit seed into the shader's 32-bit RNG key so seeds
            // differing only in their high bits don't collide.
            seed: (seed ^ (seed >> 32)) as u32,
            _pad0: 0,
            beta: beta as f32,
            j: j as f32,
            h: h as f32,
            _pad1: 0.0,
        };

        let device = &gpu.device;

        let spins = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("spins"),
            contents: bytemuck::cast_slice(&spins_init),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let neighbors = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("neighbors"),
            contents: bytemuck::cast_slice(&nbrs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n_sites * batch * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("checkerboard"),
            source: wgpu::ShaderSource::Wgsl(include_str!("checkerboard.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("checkerboard bind group layout"),
            entries: &[
                storage_entry(0, false),
                storage_entry(1, true),
                uniform_entry(2),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("checkerboard pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..8,
            }],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("checkerboard sweep"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("sweep"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("checkerboard bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: spins.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: neighbors.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n_sites as u32).div_ceil(64);

        GpuChain {
            gpu,
            pipeline,
            bind_group,
            spins,
            staging,
            _neighbors: neighbors,
            _params: params_buf,
            lattice: lattice.clone(),
            n_sites,
            sweeps_between,
            batch,
            workgroups,
            sweeps_done: 0,
            buffer: VecDeque::new(),
        }
    }

    /// Advance the chain by `sweeps` sweeps on the device, producing no snapshot —
    /// the GPU counterpart of [`Chain::advance`](crate::chain::Chain::advance),
    /// used to discard warmup. Submits the work but does not wait on it; the next
    /// batch's reads are ordered after it on the queue.
    pub fn advance(&mut self, sweeps: usize) {
        // Submit in chunks so a large sweep count never encodes one giant command
        // buffer (each sweep is two compute passes).
        const CHUNK: usize = 256;
        let mut remaining = sweeps;
        while remaining > 0 {
            let this = remaining.min(CHUNK);
            let mut encoder =
                self.gpu
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("warmup"),
                    });
            for _ in 0..this {
                self.encode_sweep(&mut encoder, self.sweeps_done);
                self.sweeps_done += 1;
            }
            self.gpu.queue.submit(std::iter::once(encoder.finish()));
            remaining -= this;
        }
    }

    /// Encode one sweep as two color passes into `encoder`. Separate passes make
    /// color 1 read color 0's *new* values (the barrier). Both passes share
    /// `sweep_index`, so a site draws once per sweep — the two colors touch
    /// disjoint sites, so there is no collision.
    fn encode_sweep(&self, encoder: &mut wgpu::CommandEncoder, sweep_index: u32) {
        for color in 0u32..2 {
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
        let n = self.n_sites;
        let stride = (n * std::mem::size_of::<u32>()) as u64;

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sample batch"),
            });
        for k in 0..self.batch {
            for _ in 0..self.sweeps_between {
                self.encode_sweep(&mut encoder, self.sweeps_done);
                self.sweeps_done += 1;
            }
            encoder.copy_buffer_to_buffer(&self.spins, 0, &self.staging, k as u64 * stride, stride);
        }
        self.gpu.queue.submit(std::iter::once(encoder.finish()));

        // Map the batch back in one transfer, copy it out, then release the map.
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.gpu.device.poll(wgpu::PollType::Wait);
        rx.recv()
            .expect("map callback dropped")
            .expect("buffer map failed");
        let words: Vec<u32> = {
            let data = slice.get_mapped_range();
            bytemuck::cast_slice::<u8, u32>(&data).to_vec()
        };
        self.staging.unmap();

        for k in 0..self.batch {
            let chunk = &words[k * n..(k + 1) * n];
            let mut cfg = Configuration::<2>::cold(&self.lattice, Cell::Site);
            for (site, &v) in chunk.iter().enumerate() {
                cfg.poke(
                    site,
                    State::new(v as usize).expect("a GPU spin is always 0 or 1"),
                );
            }
            self.buffer.push_back(cfg);
        }
    }
}

impl Iterator for GpuChain {
    type Item = Configuration<2>;

    /// Yield the next sampled configuration, running a fresh batch on the device
    /// when the host-side buffer drains. Always `Some`: the chain is open-ended,
    /// so callers bound it with `.take(n)`.
    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            self.run_batch();
        }
        self.buffer.pop_front()
    }
}

/// A read-write (`read_only = false`) or read-only storage-buffer layout entry.
fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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
fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(test)]
mod tests {
    use super::*;
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

        let mut chain = GpuChain::new(gpu, &lat, 1.0, 0.0, 0.5, 7, &start, 0, 1);
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

        // CPU reference: Chain driven by the Checkerboard updater.
        let (e_cpu, m_cpu) = {
            use crate::chain::Chain;
            use crate::updater::Checkerboard;
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(11);
            let mut cfg = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let updater = Checkerboard;
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

        // GPU: GpuChain over the same model and geometry.
        let (e_gpu, m_gpu) = {
            let lat = Lattice::new(shape);
            let mut rng = RandRng::seed_from_u64(22);
            let start = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let mut chain = GpuChain::new(gpu, &lat, j, h, beta, 12345, &start, sweeps_between, 64);
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
