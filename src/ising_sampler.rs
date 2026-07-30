//! Ising sampler: the orchestrator that owns a warmed-up run and streams from
//! it.
//!
//! Where a chain is pure mechanism, [`IsingSampler`] owns the **phasing**. It
//! assembles the pieces a run needs, thermalizes once in its constructor, and
//! then streams from its warmed-up state. It is the blessed path: a bare
//! [`Chain`] yields pre-equilibrium states, an `IsingSampler` has already
//! thermalized before it gives you anything.
//!
//! It also owns the **backend choice**. The run's [`UpdaterKind`] selects between
//! the CPU chain and the GPU chain, and `IsingSampler` holds whichever one the
//! config asked for. [`samples`](IsingSampler::samples) returns an [`AnyChain`]
//! over it — a thin front that yields [`Configuration`]s the same way regardless
//! of backend, so a consumer's loop never names CPU or GPU.
//!
//! It streams and keeps no history: the consumer decides what to retain — fold
//! into running sums, flush batches to a file, or collect when the run is small.
//! Because the warmed-up state stays here, `samples` can be called again to draw
//! more after looking at the error bars, and the second batch continues the same
//! chain without re-thermalizing.
//!
//! Geometry for measurement comes off the *sampler*, not the chain:
//! [`lattice`](IsingSampler::lattice) and [`model`](IsingSampler::model) hand
//! back owned copies, so a consumer reads them once and then streams. (The CPU
//! `Chain` exposes its own borrowed accessors, but a `GpuChain` owns its geometry and
//! cannot lend it past a by-value consume, so the uniform seam puts them here.)

use crate::chain::Chain;
use crate::config::UpdaterKind;
use crate::configuration::Configuration;
use crate::gpu::{Gpu, GpuChain};
use crate::ising_config::IsingRunConfig;
use crate::lattice::Lattice;
use crate::model::Ising;
use crate::rng::RandRng;
use crate::updater::{AnyUpdater, Checkerboard, Metropolis};

/// How many samples a GPU run produces per device round-trip. A performance knob,
/// not a physics one — the samples are identical regardless — so it is a default
/// here rather than a config field.
const GPU_BATCH: usize = 64;

/// The evolving state an [`IsingSampler`] streams from, one variant per backend.
///
/// The CPU variant holds the loose pieces a transient [`Chain`] borrows each call;
/// the GPU variant owns a persistent [`GpuChain`]. This is where the two
/// backends' opposite ownership models are reconciled behind one type.
enum Engine {
    Cpu {
        rng: RandRng,
        updater: AnyUpdater,
        /// The evolving configuration, lent to a transient [`Chain`] per call.
        state: Configuration<2>,
    },
    Gpu(GpuChain),
}

/// A stream of thermalized [`Configuration`]s, over either backend.
///
/// Both variants yield the same item, so a consumer bounds it with `.take(n)` and
/// measures each config without knowing which backend produced it. The CPU
/// variant is a transient [`Chain`] borrowing the sampler; the GPU variant a
/// mutable borrow of the sampler's persistent [`GpuChain`].
pub enum AnyChain<'a> {
    Cpu(Chain<'a, 2, 2, Ising, AnyUpdater, RandRng>),
    Gpu(&'a mut GpuChain),
}

impl Iterator for AnyChain<'_> {
    type Item = Configuration<2>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AnyChain::Cpu(chain) => chain.next(),
            AnyChain::Gpu(chain) => chain.next(),
        }
    }
}

/// Owns a run's assembled pieces and its evolving state, thermalized and ready
/// to stream.
///
/// Fixed at `D = 2`, `Q = 2`, matching [`IsingRunConfig`]. The backend is chosen
/// from the config's [`UpdaterKind`] and held in a private per-backend
/// `Engine`, so neither the CPU nor the GPU type leaks into the streaming interface.
pub struct IsingSampler {
    lattice: Lattice<2>,
    model: Ising,
    beta: f64,
    sweeps_between: usize,
    engine: Engine,
}

impl IsingSampler {
    /// Assemble a run from its config and thermalize it.
    ///
    /// Runs `config.thermalize` warmup sweeps and discards them, so the sampler is
    /// at equilibrium before it streams. [`IsingRunConfig::build`] seeds the
    /// generator
    /// *before* drawing a [`Start::Hot`](crate::config::Start::Hot) configuration,
    /// which is what makes the whole run replay from the seed alone; the GPU
    /// backend starts from that same drawn configuration, so a CPU and GPU run of
    /// one config begin identically.
    ///
    /// `config.n_samples` is deliberately not read here: how many samples to draw
    /// is the consumer's `.take(n)`.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid (via `build`), or if it selects the GPU
    /// backend on a machine with no GPU adapter.
    pub fn new(config: &IsingRunConfig) -> Self {
        let (lattice, model, mut rng, mut state, beta) = config.build();
        let sweeps_between = config.sweeps_between;

        let engine = if let UpdaterKind::GpuCheckerboard = config.updater {
            let gpu = Gpu::new().expect("GPU backend requested but no GPU adapter is available");
            let mut chain = GpuChain::new(
                gpu,
                &lattice,
                config.j,
                config.h,
                beta,
                config.seed,
                &state,
                sweeps_between,
                GPU_BATCH,
            );
            chain.advance(config.thermalize);
            Engine::Gpu(chain)
        } else {
            let updater = match config.updater {
                UpdaterKind::Metropolis => AnyUpdater::Metropolis(Metropolis),
                UpdaterKind::Checkerboard => AnyUpdater::Checkerboard(Checkerboard),
                UpdaterKind::GpuCheckerboard => unreachable!("handled by the outer branch"),
            };
            // Warm up a transient chain over the loose pieces, then stow them.
            Chain::new(&mut state, &lattice, &model, &updater, beta, &mut rng, 1)
                .advance(config.thermalize);
            Engine::Cpu {
                rng,
                updater,
                state,
            }
        };

        IsingSampler {
            lattice,
            model,
            beta,
            sweeps_between,
            engine,
        }
    }

    /// The lattice this run is on — an owned clone, for measuring the stream
    /// without holding a borrow of the sampler across
    /// [`samples`](IsingSampler::samples).
    pub fn lattice(&self) -> Lattice<2> {
        self.lattice.clone()
    }

    /// The model pricing this run's moves — for measuring the stream. `Ising` is
    /// `Copy`, so this is a cheap value, not a borrow.
    pub fn model(&self) -> Ising {
        self.model
    }

    /// Stream from the warmed-up state, one [`Configuration`] per
    /// `sweeps_between` sweeps. Bound it with `.take(n)`; the sampler retains
    /// nothing, and calling this again continues the same chain.
    ///
    /// ```
    /// # use plaquette::ising_config::IsingRunConfig;
    /// # use plaquette::{measure, IsingSampler};
    /// # let run = IsingRunConfig::parse("shape=[8,8]\nj=1.0\nbeta=0.44\nthermalize=10\nsweeps_between=1\nn_samples=5\nseed=1").unwrap();
    /// let mut sampler = IsingSampler::new(&run);
    /// let (lattice, model) = (sampler.lattice(), sampler.model());
    /// let energies: Vec<f64> = sampler
    ///     .samples()
    ///     .take(5)
    ///     .map(|c| measure(&model, &lattice, &c).energy)
    ///     .collect();
    /// ```
    pub fn samples(&mut self) -> AnyChain<'_> {
        let IsingSampler {
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
            } => AnyChain::Cpu(Chain::new(
                state,
                lattice,
                model,
                updater,
                *beta,
                rng,
                *sweeps_between,
            )),
            Engine::Gpu(chain) => AnyChain::Gpu(chain),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Start;
    use crate::observables::measure;

    fn config() -> IsingRunConfig {
        IsingRunConfig {
            shape: [8, 8],
            j: 1.0,
            h: 0.0,
            beta: 0.44,
            updater: UpdaterKind::Metropolis,
            thermalize: 50,
            sweeps_between: 2,
            n_samples: 10,
            seed: 20260723,
            start: Start::Hot,
            description: None,
        }
    }

    /// The stream yields configs of the run's lattice size, as many as asked for.
    #[test]
    fn streams_configs_of_the_right_size() {
        let mut sampler = IsingSampler::new(&config());
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
    }

    /// A checkerboard-configured run streams too: the `UpdaterKind` from the
    /// config selects the CPU checkerboard, driven like any other updater.
    #[test]
    fn streams_with_the_checkerboard_updater() {
        let mut run = config();
        run.updater = UpdaterKind::Checkerboard;

        let mut sampler = IsingSampler::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
    }

    /// A GPU-configured run streams through the same interface. Skips when no GPU
    /// adapter is available, so the suite stays green on a headless runner.
    #[test]
    fn streams_with_the_gpu_checkerboard() {
        if crate::gpu::Gpu::new().is_none() {
            eprintln!("no GPU adapter available; skipping GPU test");
            return;
        }
        let mut run = config();
        run.updater = UpdaterKind::GpuCheckerboard;

        let mut sampler = IsingSampler::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 64));
    }

    /// Geometry off the sampler lets a consumer measure the stream without owning
    /// a second lattice.
    #[test]
    fn measures_the_stream_via_sampler_geometry() {
        let mut sampler = IsingSampler::new(&config());
        let lattice = sampler.lattice();
        let model = sampler.model();
        let n_sites = lattice.n_sites() as f64;

        let energies: Vec<f64> = sampler
            .samples()
            .take(3)
            .map(|c| measure(&model, &lattice, &c).energy)
            .collect();

        assert_eq!(energies.len(), 3);
        assert!(
            energies.iter().all(|&e| e.abs() <= 2.0 * n_sites),
            "energy within physical bounds"
        );
    }

    /// A sampler's stream equals driving a `Chain` by hand: warmup at stride 1,
    /// then samples at the configured stride. The run replays from the config
    /// alone.
    #[test]
    fn matches_a_hand_driven_chain() {
        let run = config();
        let (lattice, model, mut rng, mut state, beta) = run.build();
        let updater = Metropolis;
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

        let mut sampler = IsingSampler::new(&run);
        let got: Vec<_> = sampler.samples().take(run.n_samples).collect();

        assert_eq!(got, expected);
    }

    /// Calling `samples()` twice continues the same chain without re-thermalizing:
    /// `n` then `m` equals one run of `n + m`.
    #[test]
    fn a_second_call_continues_the_same_chain() {
        let mut split = IsingSampler::new(&config());
        let mut got: Vec<_> = split.samples().take(6).collect();
        got.extend(split.samples().take(4));

        let mut whole = IsingSampler::new(&config());
        let expected: Vec<_> = whole.samples().take(10).collect();

        assert_eq!(got, expected);
    }
}
