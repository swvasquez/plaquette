//! Gauge sampler: the orchestrator that owns a warmed-up gauge run and streams
//! from it.
//!
//! [`GaugeSampler`] is to [`GaugeRunConfig`] what
//! [`IsingSampler`](crate::models::ising::sampler::IsingSampler) is to the Ising schema,
//! and it owns the same thing: the **phasing**. It assembles the pieces a run
//! needs, thermalizes once in its constructor, and then streams from its
//! warmed-up state. A bare [`Chain`] yields pre-equilibrium configurations; this
//! has already thermalized before it gives you anything.
//!
//! It also owns the **backend choice**, the same way
//! [`IsingSampler`](crate::models::ising::sampler::IsingSampler) does. The run's
//! [`UpdaterKind`] picks between the CPU chain and the GPU chain, and the
//! sampler holds whichever the config asked for.
//! [`samples`](GaugeSampler::samples) returns an [`AnyGaugeChain`] over it — a
//! thin front yielding [`Configuration`]s the same way regardless of backend, so
//! a consumer's loop never names CPU or GPU.
//!
//! Which updaters are on offer is the one place this differs from the Ising
//! side, and the reason is the grade the variables live on rather than anything
//! about backends. A gauge run accepts the three schedules that update links —
//! [`Metropolis`], [`LinkCheckerboard`], and the GPU
//! [`GpuGaugeChain`] — and
//! [`GaugeRunConfig::validate`] rejects the site schedules, whose parity
//! coloring is a statement about sites and says nothing about links.
//!
//! It streams and keeps no history: the consumer decides what to retain. Because
//! the warmed-up state stays here, `samples` can be called again to draw more
//! after looking at the error bars, and the second batch continues the same chain
//! without re-thermalizing.
//!
//! Measurement stays entirely with the consumer, and the sampler's job is to hand
//! over what measuring needs: [`lattice`](GaugeSampler::lattice) and
//! [`model`](GaugeSampler::model) give owned copies, so a consumer reads them once
//! and then maps [`gauge_measure`](crate::models::gauge::gauge_measure) or
//! [`wilson_rectangles`](crate::models::gauge::wilson_rectangles) over the stream.
//! Which of those to call, and at what loop sizes, is not a run parameter — the
//! configurations are the same either way — which is why none of it appears in
//! the config.
//!
//! ```
//! # use plaquette::models::gauge::run_config::GaugeRunConfig;
//! # use plaquette::models::gauge::{GaugeSampler, gauge_measure};
//! # let run = GaugeRunConfig::parse("shape=[4,4,4]\nj=1.0\nbeta=0.75\nthermalize=10\nsweeps_between=1\nn_samples=5\nseed=1").unwrap();
//! let mut sampler = GaugeSampler::<3>::new(&run);
//! let (lattice, model) = (sampler.lattice(), sampler.model());
//! let plaquettes: Vec<f64> = sampler
//!     .samples()
//!     .take(5)
//!     .map(|c| gauge_measure(&model, &lattice, &c).plaquette_sum)
//!     .collect();
//! ```

use crate::chain::Chain;
use crate::config::UpdaterKind;
use crate::configuration::Configuration;
use crate::device::{GPU_BATCH, Gpu};
use crate::lattice::Lattice;
use crate::models::gauge::Z2Gauge;
use crate::models::gauge::gpu::GpuGaugeChain;
use crate::models::gauge::run_config::GaugeRunConfig;
use crate::rng::RandRng;
use crate::updater::{AnyUpdater, LinkCheckerboard, Metropolis};

/// The evolving state a [`GaugeSampler`] streams from, one variant per backend.
///
/// The CPU variant holds the loose pieces a transient [`Chain`] borrows each
/// call; the GPU variant owns a persistent [`GpuGaugeChain`]. This is where the
/// two backends' opposite ownership models are reconciled behind one type.
enum Engine {
    Cpu {
        rng: RandRng,
        updater: AnyUpdater,
        /// The evolving configuration, lent to a transient [`Chain`] per call.
        state: Configuration<2>,
    },
    /// Boxed because the device chain is far larger than the CPU variant, and an
    /// enum is sized by its largest one.
    Gpu(Box<GpuGaugeChain>),
}

/// A stream of thermalized link [`Configuration`]s, over either backend.
///
/// Both variants yield the same item, so a consumer bounds it with `.take(n)`
/// and measures each config without knowing which backend produced it. The CPU
/// variant is a transient [`Chain`] borrowing the sampler; the GPU variant a
/// mutable borrow of the sampler's persistent [`GpuGaugeChain`].
///
/// Named for its grade rather than called `AnyChain`, because the Ising sampler
/// has a structurally identical type and both are re-exported from the crate
/// root, where one plain name cannot cover two.
///
/// Only the CPU variant carries the dimension. The device chain reads the
/// lattice once when it is built and keeps buffers afterwards, so `D` never
/// reaches [`GpuGaugeChain`]'s type — the parameter here is the borrowed
/// [`Chain`]'s alone.
pub enum AnyGaugeChain<'a, const D: usize> {
    /// A transient chain borrowing the sampler's state for the length of the run.
    Cpu(Chain<'a, 2, D, Z2Gauge, AnyUpdater, RandRng>),
    /// A mutable borrow of the sampler's persistent device chain.
    Gpu(&'a mut GpuGaugeChain),
}

impl<const D: usize> Iterator for AnyGaugeChain<'_, D> {
    type Item = Configuration<2>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AnyGaugeChain::Cpu(chain) => chain.next(),
            AnyGaugeChain::Gpu(chain) => chain.next(),
        }
    }
}

/// Owns a gauge run's assembled pieces and its evolving state, thermalized and
/// ready to stream.
///
/// Fixed at `Q = 2` and generic over the dimension, which a driver names in its
/// own source and the config's shape must agree with. The backend is chosen from the config's
/// [`UpdaterKind`] and held in a private per-backend `Engine`, so neither the
/// CPU nor the GPU type leaks into the streaming interface.
pub struct GaugeSampler<const D: usize> {
    lattice: Lattice<D>,
    model: Z2Gauge,
    beta: f64,
    sweeps_between: usize,
    engine: Engine,
}

impl<const D: usize> GaugeSampler<D> {
    /// Assemble a gauge run from its config and thermalize it.
    ///
    /// Runs `config.thermalize` warmup sweeps at stride 1 and discards them, so
    /// the sampler is at equilibrium before it streams.
    /// [`GaugeRunConfig::build`] seeds the generator *before* drawing a
    /// [`Start::Hot`](crate::config::Start::Hot) configuration, which is what
    /// makes the whole run replay from the seed alone.
    ///
    /// `config.n_samples` is deliberately not read here: how many samples to
    /// draw is the consumer's `.take(n)`.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid (via `build`), which includes asking for
    /// an updater that schedules sites rather than links or naming a dimension
    /// other than `D`, or if it selects the GPU backend on a machine with no GPU
    /// adapter.
    pub fn new(config: &GaugeRunConfig) -> Self {
        let (lattice, model, mut rng, mut state, beta) = config.build::<D>();

        let engine = if let UpdaterKind::GpuLinkCheckerboard = config.updater {
            let gpu = Gpu::new().expect("GPU backend requested but no GPU adapter is available");
            let mut chain = GpuGaugeChain::new(
                gpu,
                &lattice,
                config.j,
                beta,
                config.seed,
                &state,
                config.sweeps_between,
                GPU_BATCH,
            );
            chain.advance(config.thermalize);
            Engine::Gpu(Box::new(chain))
        } else {
            let updater = match config.updater {
                UpdaterKind::Metropolis => AnyUpdater::Metropolis(Metropolis),
                UpdaterKind::LinkCheckerboard => AnyUpdater::LinkCheckerboard(LinkCheckerboard),
                other => unreachable!("rejected by GaugeRunConfig::validate: {other:?}"),
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

        GaugeSampler {
            lattice,
            model,
            beta,
            sweeps_between: config.sweeps_between,
            engine,
        }
    }

    /// The lattice this run is on — an owned clone, for measuring the stream
    /// without holding a borrow of the sampler across
    /// [`samples`](GaugeSampler::samples).
    pub fn lattice(&self) -> Lattice<D> {
        self.lattice.clone()
    }

    /// The model pricing this run's moves — for measuring the stream. `Z2Gauge`
    /// is `Copy`, so this is a cheap value, not a borrow.
    pub fn model(&self) -> Z2Gauge {
        self.model
    }

    /// Stream from the warmed-up state, one [`Configuration`] per
    /// `sweeps_between` sweeps. Bound it with `.take(n)`; the sampler retains
    /// nothing, and calling this again continues the same chain.
    pub fn samples(&mut self) -> AnyGaugeChain<'_, D> {
        let GaugeSampler {
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
            } => AnyGaugeChain::Cpu(Chain::new(
                state,
                lattice,
                model,
                updater,
                *beta,
                rng,
                *sweeps_between,
            )),
            Engine::Gpu(chain) => AnyGaugeChain::Gpu(chain.as_mut()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Start, UpdaterKind};
    use crate::configuration::Cell;
    use crate::models::gauge::{gauge_measure, wilson_rectangles};

    fn config() -> GaugeRunConfig {
        GaugeRunConfig {
            shape: vec![4, 4, 4],
            j: 1.0,
            beta: 0.75,
            updater: UpdaterKind::Metropolis,
            thermalize: 50,
            sweeps_between: 2,
            n_samples: 10,
            seed: 20260728,
            start: Start::Hot,
            description: None,
        }
    }

    /// A link-checkerboard-configured run streams too: the `UpdaterKind` from
    /// the config selects the CPU link schedule, driven like any other updater.
    #[test]
    fn streams_with_the_link_checkerboard_updater() {
        let mut run = config();
        run.updater = UpdaterKind::LinkCheckerboard;

        let mut sampler = GaugeSampler::<3>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 192));
        assert!(configs.iter().all(|c| c.cell() == Cell::Link));
    }

    /// A GPU-configured run streams through the same interface. Skips when no
    /// GPU adapter is available, so the suite stays green on a headless runner.
    #[test]
    fn streams_with_the_gpu_link_checkerboard() {
        if crate::device::require_gpu().is_none() {
            return;
        }
        let mut run = config();
        run.updater = UpdaterKind::GpuLinkCheckerboard;

        let mut sampler = GaugeSampler::<3>::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 192));
        assert!(configs.iter().all(|c| c.cell() == Cell::Link));
    }

    /// The stream yields link configs of the run's lattice size, as many as
    /// asked for: 3 links per site on 64 sites.
    #[test]
    fn streams_link_configs_of_the_right_size() {
        let mut sampler = GaugeSampler::<3>::new(&config());
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_vars() == 192));
        assert!(configs.iter().all(|c| c.cell() == Cell::Link));
    }

    /// A sampler's stream equals driving a `Chain` by hand: warmup at stride 1,
    /// then samples at the configured stride. The run replays from the config
    /// alone.
    #[test]
    fn matches_a_hand_driven_chain() {
        let run = config();
        let (lattice, model, mut rng, mut state, beta) = run.build::<3>();
        let updater = AnyUpdater::Metropolis(Metropolis);
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

        let mut sampler = GaugeSampler::<3>::new(&run);
        let got: Vec<_> = sampler.samples().take(run.n_samples).collect();

        assert_eq!(got, expected);
    }

    /// Two samplers from the same config produce the same configurations, which
    /// is the reproducibility guarantee end to end rather than at `build` alone.
    #[test]
    fn is_reproducible_from_the_config() {
        let mut a = GaugeSampler::<3>::new(&config());
        let mut b = GaugeSampler::<3>::new(&config());
        assert_eq!(
            a.samples().take(4).collect::<Vec<_>>(),
            b.samples().take(4).collect::<Vec<_>>()
        );
    }

    /// Calling `samples()` twice continues the same chain without
    /// re-thermalizing: `n` then `m` equals one run of `n + m`.
    #[test]
    fn a_second_call_continues_the_same_chain() {
        let mut split = GaugeSampler::<3>::new(&config());
        let mut got: Vec<_> = split.samples().take(6).collect();
        got.extend(split.samples().take(4));

        let mut whole = GaugeSampler::<3>::new(&config());
        let expected: Vec<_> = whole.samples().take(10).collect();

        assert_eq!(got, expected);
    }

    /// Geometry off the sampler lets a consumer measure the stream without
    /// owning a second lattice. The plaquette sum is bounded by the plaquette
    /// count — 3 planes times 64 sites — which is the check that the field, the
    /// lattice, and the model handed over actually belong to each other.
    #[test]
    fn measures_the_stream_via_sampler_geometry() {
        let mut sampler = GaugeSampler::<3>::new(&config());
        let lattice = sampler.lattice();
        let model = sampler.model();
        let n_plaquettes = lattice.n_plaquettes() as f64;
        assert_eq!(n_plaquettes, 192.0);

        let samples: Vec<_> = sampler
            .samples()
            .take(3)
            .map(|c| gauge_measure(&model, &lattice, &c))
            .collect();

        assert_eq!(samples.len(), 3);
        assert!(
            samples
                .iter()
                .all(|s| s.plaquette_sum.abs() <= n_plaquettes),
            "plaquette sum within physical bounds"
        );
        // The energy is `-j` times the plaquette sum, at `j = 1`.
        assert!(
            samples.iter().all(|s| s.energy == -s.plaquette_sum),
            "energy tracks the plaquette sum"
        );
    }

    /// A cold start with no warmup streams the ground state, whose loops are all
    /// exactly 1 — the cheapest end-to-end check that the whole path measures a
    /// gauge-invariant number rather than noise.
    #[test]
    fn a_cold_run_measures_unit_wilson_loops() {
        let mut run = config();
        run.start = Start::Cold;
        run.thermalize = 0;
        run.sweeps_between = 0;

        let mut sampler = GaugeSampler::<3>::new(&run);
        let (lattice, model) = (sampler.lattice(), sampler.model());
        let table = wilson_rectangles(&model, &lattice, &sampler.samples().next().unwrap(), 2);

        assert_eq!(table.per_size.len(), 3); // sides 0..=2, capped at 4 / 2
        assert!(
            table
                .per_size
                .iter()
                .all(|row| row.iter().all(|&w| w == 1.0)),
            "{:?}",
            table.per_size
        );
    }

    /// Physics smoke check: at strong coupling a driven run keeps its loops near
    /// 1, since a large `beta` suppresses every flipped plaquette. This is a
    /// sanity check on the plumbing, not a measurement — the exact `tanh(β)^(RT)`
    /// area law is a two-dimensional result, and the library tests already pin it
    /// there.
    #[test]
    fn a_strongly_coupled_run_keeps_its_loops_near_one() {
        let mut run = config();
        run.beta = 3.0;
        run.thermalize = 200;
        run.sweeps_between = 2;

        let mut sampler = GaugeSampler::<3>::new(&run);
        let (lattice, model) = (sampler.lattice(), sampler.model());

        let n = 40;
        let mean = sampler
            .samples()
            .take(n)
            .map(|c| wilson_rectangles(&model, &lattice, &c, 1).per_size[1][1])
            .sum::<f64>()
            / n as f64;

        assert!(mean > 0.9, "mean unit loop at beta = 3 was {mean}");
    }

    /// The link checkerboard samples the same distribution the random-order
    /// Metropolis does: at the same coupling the two agree on the mean plaquette.
    /// The comparison is distributional rather than bit-for-bit, because the two
    /// schedules consume the generator differently — Metropolis draws a link
    /// index per step and the checkerboard draws none — so identical seeds put
    /// them on different streams. The tolerance is set to swamp the residual
    /// Monte Carlo error at this sample count rather than to any physics.
    #[test]
    fn link_checkerboard_matches_the_metropolis_distribution() {
        fn mean_plaquette(kind: UpdaterKind) -> f64 {
            let mut run = config();
            run.updater = kind;
            run.beta = 0.5;
            run.thermalize = 500;
            run.sweeps_between = 4;

            let mut sampler = GaugeSampler::<3>::new(&run);
            let (lattice, model) = (sampler.lattice(), sampler.model());
            let n = 400;
            sampler
                .samples()
                .take(n)
                .map(|c| {
                    gauge_measure(&model, &lattice, &c).plaquette_sum
                        / lattice.n_plaquettes() as f64
                })
                .sum::<f64>()
                / n as f64
        }

        let metropolis = mean_plaquette(UpdaterKind::Metropolis);
        let checkerboard = mean_plaquette(UpdaterKind::LinkCheckerboard);

        assert!(
            (metropolis - checkerboard).abs() < 0.02,
            "mean plaquette: metropolis {metropolis}, link checkerboard {checkerboard}"
        );
    }
}
