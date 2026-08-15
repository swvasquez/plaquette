//! Potts sampler: the orchestrator that owns a warmed-up Potts run and streams
//! from it.
//!
//! [`PottsSampler`] is to [`PottsRunConfig`] what
//! [`IsingSampler`](crate::models::ising::sampler::IsingSampler) is to the Ising schema,
//! and it owns the same thing: the **phasing**. It assembles the pieces a run
//! needs, thermalizes once in its constructor, and then streams from its
//! warmed-up state. A bare [`Chain`] yields pre-equilibrium configurations; this
//! has already thermalized before it gives you anything.
//!
//! It also owns the **backend choice**. The run's [`BackendKind`] picks between
//! the CPU chain and the GPU chain, and the sampler holds whichever the config
//! asked for. [`samples`](PottsSampler::samples) returns an [`AnyPottsChain`]
//! over it — a thin front yielding [`Configuration`]s the same way regardless of
//! backend, so a consumer's loop never names CPU or GPU. What is on offer is
//! every kind that updates sites — the three Metropolis schedules and the two
//! cluster ones; [`PottsRunConfig::validate`] rejects the link schedules, whose
//! direction-and-parity coloring says nothing about a label on a site, and
//! rejects a cluster kind on a model whose per-label offsets break the
//! relabeling symmetry it rests on.
//!
//! It streams and keeps no history: the consumer decides what to retain. Because
//! the warmed-up state stays here, `samples` can be called again to draw more
//! after looking at the error bars, and the second batch continues the same chain
//! without re-thermalizing.
//!
//! Geometry for measurement comes off the *sampler*, not the chain:
//! [`lattice`](PottsSampler::lattice) and [`model`](PottsSampler::model) hand
//! back owned copies, so a consumer reads them once and then streams.
//!
//! # The two constants a caller names
//!
//! This is the one sampler carrying two compile-time parameters rather than one.
//! `D` is the usual story — a lattice's dimension is part of its type, and the
//! config's `shape` has to agree, which
//! [`check_dimension`](PottsRunConfig::check_dimension) reports on. `Q` is the
//! same story without the check: nothing in the file names a state count, so the
//! driver's choice is the whole of it. See
//! [`potts_config`](crate::models::potts::run_config) for why neither can come from the file,
//! and [`POTTS_Q`](crate::models::potts::run_config::POTTS_Q) and
//! [`POTTS_D`](crate::models::potts::run_config::POTTS_D) for the pair the shipped example
//! uses.
//!
//! It is a third near-duplicate of a sampler, which is the point at which that
//! stops being a coincidence. Writing it as a parallel of the other two is
//! deliberate: the evidence that a generic sampler is worth building is three of
//! these sitting side by side, not an abstraction guessed at from two.

use crate::chain::Chain;
use crate::config::{BackendKind, UpdaterRule, effective_schedule};
use crate::configuration::Configuration;
use crate::device::{GPU_BATCH, Gpu, GpuChain};
use crate::gpu_cluster::GpuClusterChain;
use crate::lattice::Lattice;
use crate::models::potts::Potts;
use crate::models::potts::gpu::gpu_chain;
use crate::models::potts::run_config::PottsRunConfig;
use crate::rng::RandRng;
use crate::updater::{AnyUpdater, ClusterUpdate, LocalUpdate};

/// The evolving state a [`PottsSampler`] streams from, one variant per backend.
///
/// The CPU variant holds the loose pieces a transient [`Chain`] borrows each
/// call; the GPU variant owns a persistent [`GpuChain`]. This is where the
/// two backends' opposite ownership models are reconciled behind one type.
enum Engine<const Q: usize> {
    Cpu {
        rng: RandRng,
        updater: AnyUpdater,
        /// The evolving configuration, lent to a transient [`Chain`] per call.
        state: Configuration<Q>,
    },
    /// Boxed because the device chain is far larger than the CPU variant, and an
    /// enum is sized by its largest one.
    Gpu(Box<GpuChain<Q>>),
    /// The device cluster chain, boxed for the same reason. A separate variant
    /// rather than a mode of `Gpu`, because the two device backends are separate
    /// types — one drives a fixed number of dispatches per sweep and the other
    /// iterates until its labeling converges.
    GpuCluster(Box<GpuClusterChain<Q>>),
}

/// A stream of thermalized label [`Configuration`]s, over either backend.
///
/// Both variants yield the same item, so a consumer bounds it with `.take(n)`
/// and measures each config without knowing which backend produced it. The CPU
/// variant is a transient [`Chain`] borrowing the sampler; the GPU variant a
/// mutable borrow of the sampler's persistent [`GpuChain`].
///
/// Only the CPU variant carries the dimension. The device chain reads the
/// lattice once when it is built and keeps buffers afterwards, so `D` never
/// reaches [`GpuChain`]'s type — the parameter here is the borrowed
/// [`Chain`]'s alone. `Q` reaches both, since it is part of what the yielded
/// configuration is.
pub enum AnyPottsChain<'a, const Q: usize, const D: usize> {
    /// A transient chain borrowing the sampler's state for the length of the run.
    Cpu(Chain<'a, Q, D, Potts<Q>, AnyUpdater, RandRng>),
    /// A mutable borrow of the sampler's persistent device chain.
    Gpu(&'a mut GpuChain<Q>),
    /// A mutable borrow of the sampler's persistent device *cluster* chain.
    GpuCluster(&'a mut GpuClusterChain<Q>),
}

impl<const Q: usize, const D: usize> Iterator for AnyPottsChain<'_, Q, D> {
    type Item = Configuration<Q>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AnyPottsChain::Cpu(chain) => chain.next(),
            AnyPottsChain::Gpu(chain) => chain.next(),
            AnyPottsChain::GpuCluster(chain) => chain.next(),
        }
    }
}

/// Owns a Potts run's assembled pieces and its evolving state, thermalized and
/// ready to stream.
///
/// Generic over the state count and the dimension, both named by the driver in
/// its own source — see the module docs. The backend is chosen from the config's
/// [`BackendKind`] and held in a private per-backend `Engine`, so neither the
/// CPU nor the GPU type leaks into the streaming interface.
pub struct PottsSampler<const Q: usize, const D: usize> {
    lattice: Lattice<D>,
    model: Potts<Q>,
    beta: f64,
    sweeps_between: usize,
    engine: Engine<Q>,
}

impl<const Q: usize, const D: usize> PottsSampler<Q, D> {
    /// Assemble a Potts run from its config and thermalize it.
    ///
    /// Runs `config.thermalize` warmup sweeps at stride 1 and discards them, so
    /// the sampler is at equilibrium before it streams.
    /// [`PottsRunConfig::build`] seeds the generator *before* drawing a
    /// [`Start::Hot`](crate::config::Start::Hot) configuration, which is what
    /// makes the whole run replay from the seed alone; the GPU backend starts
    /// from that same drawn configuration, so a CPU and a GPU run of one config
    /// begin identically.
    ///
    /// `config.n_samples` is deliberately not read here: how many samples to
    /// draw is the consumer's `.take(n)`.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid (via `build`), which includes asking for
    /// an updater that schedules links rather than sites or naming a dimension
    /// other than `D`; if `Q` is below
    /// [`MIN_STATES`](crate::models::potts::run_config::MIN_STATES); or if it selects the GPU
    /// backend on a machine with no GPU adapter.
    pub fn new(config: &PottsRunConfig) -> Self {
        let (lattice, model, mut rng, mut state, beta) = config.build::<Q, D>();
        let sweeps_between = config.sweeps_between;

        // A `match` over `(backend, rule)`: the schedule never picks the
        // engine — on the GPU it is pinned to the checkerboard by `validate`,
        // and on the CPU it is a parameter of the one `LocalUpdate`.
        let engine = match (config.backend, config.updater) {
            (BackendKind::Gpu, rule @ (UpdaterRule::Metropolis | UpdaterRule::HeatBath)) => {
                let kernel = rule.kernel().expect("a local rule has a kernel");
                let mut chain = gpu_chain(
                    require_adapter(),
                    &lattice,
                    &model,
                    beta,
                    config.seed,
                    &state,
                    sweeps_between,
                    GPU_BATCH,
                    kernel,
                );
                chain.advance(config.thermalize);
                Engine::Gpu(Box::new(chain))
            }
            (BackendKind::Gpu, UpdaterRule::SwendsenWang) => {
                let mut chain = GpuClusterChain::new(
                    require_adapter(),
                    &lattice,
                    &model,
                    beta,
                    config.seed,
                    &state,
                    sweeps_between,
                );
                chain.advance(config.thermalize);
                Engine::GpuCluster(Box::new(chain))
            }
            (BackendKind::Cpu, rule) => {
                let updater = match rule.kernel() {
                    Some(kernel) => AnyUpdater::Local(LocalUpdate::new(
                        kernel,
                        effective_schedule(config.schedule).into(),
                    )),
                    None => AnyUpdater::Cluster(ClusterUpdate::swendsen_wang(&model)),
                };
                // Warm up a transient chain over the loose pieces, then stow them.
                Chain::new(&mut state, &lattice, &model, &updater, beta, &mut rng, 1)
                    .advance(config.thermalize);
                Engine::Cpu {
                    rng,
                    updater,
                    state,
                }
            }
        };

        PottsSampler {
            lattice,
            model,
            beta,
            sweeps_between,
            engine,
        }
    }

    /// The lattice this run is on — an owned clone, for measuring the stream
    /// without holding a borrow of the sampler across
    /// [`samples`](PottsSampler::samples).
    pub fn lattice(&self) -> Lattice<D> {
        self.lattice.clone()
    }

    /// The model pricing this run's moves — for measuring the stream. `Potts` is
    /// `Copy`, so this is a cheap value, not a borrow.
    pub fn model(&self) -> Potts<Q> {
        self.model
    }

    /// Stream from the warmed-up state, one [`Configuration`] per
    /// `sweeps_between` sweeps. Bound it with `.take(n)`; the sampler retains
    /// nothing, and calling this again continues the same chain.
    ///
    /// ```
    /// # use plaquette::models::potts::run_config::PottsRunConfig;
    /// # use plaquette::models::potts::{PottsSampler, potts_measure};
    /// # let run = PottsRunConfig::parse("shape=[8,8]\nj=1.0\nbeta=1.5\nthermalize=10\nsweeps_between=1\nn_samples=5\nseed=1").unwrap();
    /// let mut sampler = PottsSampler::<3, 2>::new(&run);
    /// let (lattice, model) = (sampler.lattice(), sampler.model());
    /// let order: Vec<f64> = sampler
    ///     .samples()
    ///     .take(5)
    ///     .map(|c| potts_measure(&model, &lattice, &c).order)
    ///     .collect();
    /// ```
    pub fn samples(&mut self) -> AnyPottsChain<'_, Q, D> {
        let PottsSampler {
            lattice,
            model,
            beta,
            sweeps_between,
            engine,
        } = self;
        match engine {
            Engine::Cpu {
                rng,
                updater,
                state,
            } => AnyPottsChain::Cpu(Chain::new(
                state,
                lattice,
                model,
                updater,
                *beta,
                rng,
                *sweeps_between,
            )),
            Engine::Gpu(chain) => AnyPottsChain::Gpu(chain.as_mut()),
            Engine::GpuCluster(chain) => AnyPottsChain::GpuCluster(chain.as_mut()),
        }
    }
}

/// A device for a run whose config asked for one, or a panic saying so.
///
/// Shared by the two device branches above so the message a user without an
/// adapter sees does not depend on which backend they named.
fn require_adapter() -> Gpu {
    Gpu::new().expect("GPU backend requested but no GPU adapter is available")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ScheduleKind, Start};
    use crate::configuration::Cell;
    use crate::models::potts::potts_measure;
    use crate::models::potts::run_config::{POTTS_D, POTTS_Q};

    fn config() -> PottsRunConfig {
        PottsRunConfig {
            shape: vec![8, 8],
            j: 1.0,
            h: Vec::new(),
            beta: 1.5,
            updater: UpdaterRule::Metropolis,
            schedule: None,
            backend: BackendKind::Cpu,
            thermalize: 50,
            sweeps_between: 2,
            n_samples: 10,
            seed: 20260809,
            start: Start::Hot,
            description: None,
        }
    }

    /// The stream yields site configs of the run's lattice size, as many as
    /// asked for.
    #[test]
    fn streams_configs_of_the_right_size() {
        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
        assert!(configs.iter().all(|c| c.cell() == Cell::Site));
    }

    /// A checkerboard-configured run streams too: the schedule field from the
    /// config selects the CPU site checkerboard, driven like any other updater.
    #[test]
    fn streams_with_the_checkerboard_schedule() {
        let mut run = config();
        run.schedule = Some(ScheduleKind::Checkerboard);

        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
    }

    /// A GPU-configured run streams through the same interface. Skips when no
    /// GPU adapter is available, so the suite stays green on a headless runner.
    #[test]
    fn streams_with_the_gpu_backend() {
        if crate::device::require_gpu().is_none() {
            return;
        }
        let mut run = config();
        run.schedule = Some(ScheduleKind::Checkerboard);
        run.backend = BackendKind::Gpu;

        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
    }

    /// A cluster-configured run streams too, over the same interface.
    #[test]
    fn streams_with_the_cluster_updater() {
        let mut run = config();
        run.updater = UpdaterRule::SwendsenWang;

        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
        assert!(configs.iter().all(|c| c.cell() == Cell::Site));
    }

    /// The device cluster backend reaches the same interface, on an *odd*
    /// lattice — which no other GPU kind can run, and which the sampler
    /// therefore has to carry all the way through without an even-extent guard
    /// firing somewhere in the middle.
    #[test]
    fn streams_with_the_gpu_cluster_updater() {
        if crate::device::require_gpu().is_none() {
            return;
        }
        let mut run = config();
        run.updater = UpdaterRule::SwendsenWang;
        run.backend = BackendKind::Gpu;
        run.shape = vec![9, 7];

        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 63));
    }

    /// The state count is the driver's alone, and it really does reach the
    /// chain: the same config run at four states produces labels a three-state
    /// run could not.
    ///
    /// This is what the config file cannot say, so it is worth pinning that
    /// nothing downstream quietly assumes a particular `Q`.
    #[test]
    fn the_driver_chooses_the_state_count() {
        let mut run = config();
        run.beta = 0.2; // disordered, so every label is well populated

        let mut three = PottsSampler::<3, 2>::new(&run);
        let from_three = three.samples().next().unwrap();
        assert!(from_three.variables().iter().all(|s| s.index() < 3));

        let mut four = PottsSampler::<4, 2>::new(&run);
        let from_four = four.samples().next().unwrap();
        assert!(from_four.variables().iter().any(|s| s.index() == 3));
    }

    /// Geometry off the sampler lets a consumer measure the stream without
    /// owning a second lattice. The order parameter is bounded by its own
    /// definition, which is the check that the field, the lattice, and the model
    /// handed over actually belong to each other.
    #[test]
    fn measures_the_stream_via_sampler_geometry() {
        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        let lattice = sampler.lattice();
        let model = sampler.model();
        let n_bonds = lattice.n_links() as f64;

        let samples: Vec<_> = sampler
            .samples()
            .take(3)
            .map(|c| potts_measure(&model, &lattice, &c))
            .collect();

        assert_eq!(samples.len(), 3);
        assert!(
            samples.iter().all(|s| (0.0..=1.0).contains(&s.order)),
            "the order parameter runs from 0 to 1"
        );
        // Every bond either agrees or does not, so the energy sits between
        // `-j * n_bonds` and zero at `j = 1`.
        assert!(
            samples.iter().all(|s| (-n_bonds..=0.0).contains(&s.energy)),
            "energy within physical bounds"
        );
    }

    /// A sampler's stream equals driving a `Chain` by hand: warmup at stride 1,
    /// then samples at the configured stride. The run replays from the config
    /// alone.
    #[test]
    fn matches_a_hand_driven_chain() {
        let run = config();
        let (lattice, model, mut rng, mut state, beta) = run.build::<POTTS_Q, POTTS_D>();
        let updater = AnyUpdater::Local(LocalUpdate::default());
        Chain::new(&mut state, &lattice, &model, &updater, beta, &mut rng, 1)
            .advance(run.thermalize);
        let expected: Vec<_> = Chain::new(
            &mut state,
            &lattice,
            &model,
            &updater,
            beta,
            &mut rng,
            run.sweeps_between,
        )
        .take(run.n_samples)
        .collect();

        let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
        let got: Vec<_> = sampler.samples().take(run.n_samples).collect();

        assert_eq!(got, expected);
    }

    /// Two samplers from the same config produce the same configurations, which
    /// is the reproducibility guarantee end to end rather than at `build` alone.
    #[test]
    fn is_reproducible_from_the_config() {
        let mut a = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        let mut b = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        assert_eq!(
            a.samples().take(4).collect::<Vec<_>>(),
            b.samples().take(4).collect::<Vec<_>>()
        );
    }

    /// Calling `samples()` twice continues the same chain without
    /// re-thermalizing: `n` then `m` equals one run of `n + m`.
    #[test]
    fn a_second_call_continues_the_same_chain() {
        let mut split = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        let mut got: Vec<_> = split.samples().take(6).collect();
        got.extend(split.samples().take(4));

        let mut whole = PottsSampler::<POTTS_Q, POTTS_D>::new(&config());
        let expected: Vec<_> = whole.samples().take(10).collect();

        assert_eq!(got, expected);
    }

    /// The site checkerboard samples the same distribution the random-order
    /// Metropolis does: at the same coupling the two agree on the mean order
    /// parameter. The comparison is distributional rather than bit-for-bit,
    /// because the two schedules consume the generator differently — Metropolis
    /// draws a site index per step and the checkerboard draws none — so
    /// identical seeds put them on different streams.
    #[test]
    fn the_checkerboard_matches_the_metropolis_distribution() {
        fn mean_order(schedule: Option<ScheduleKind>) -> f64 {
            let mut run = config();
            run.schedule = schedule;
            run.beta = 1.1; // ordered, but not saturated: beta_c ~ 1.005
            run.shape = vec![16, 16];
            run.thermalize = 500;
            run.sweeps_between = 4;

            let mut sampler = PottsSampler::<POTTS_Q, POTTS_D>::new(&run);
            let (lattice, model) = (sampler.lattice(), sampler.model());
            let n = 400;
            sampler
                .samples()
                .take(n)
                .map(|c| potts_measure(&model, &lattice, &c).order)
                .sum::<f64>()
                / n as f64
        }

        let metropolis = mean_order(None);
        let checkerboard = mean_order(Some(ScheduleKind::Checkerboard));

        assert!(
            (metropolis - checkerboard).abs() < 0.03,
            "mean order parameter: metropolis {metropolis}, site checkerboard {checkerboard}"
        );
    }
}
