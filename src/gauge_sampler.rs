//! Gauge sampler: the orchestrator that owns a warmed-up gauge run and streams
//! from it.
//!
//! [`GaugeSampler`] is to [`GaugeRunConfig`] what
//! [`IsingSampler`](crate::ising_sampler::IsingSampler) is to the Ising schema,
//! and it owns the same thing: the **phasing**. It assembles the pieces a run
//! needs, thermalizes once in its constructor, and then streams from its
//! warmed-up state. A bare [`Chain`] yields pre-equilibrium configurations; this
//! has already thermalized before it gives you anything.
//!
//! What it does *not* own is a backend choice, and that is the whole difference
//! between the two samplers. The GPU engine is written around a `D = 2` site
//! field and the checkerboard coloring it uses is a statement about sites, so
//! neither has a meaning for link variables; a gauge run is CPU and Metropolis,
//! which [`GaugeRunConfig::validate`] enforces. So there is no engine enum and no
//! wrapper over two chain types here — [`samples`](GaugeSampler::samples) returns
//! the [`Chain`] itself.
//!
//! It streams and keeps no history: the consumer decides what to retain. Because
//! the warmed-up state stays here, `samples` can be called again to draw more
//! after looking at the error bars, and the second batch continues the same chain
//! without re-thermalizing.
//!
//! Measurement stays entirely with the consumer, and the sampler's job is to hand
//! over what measuring needs: [`lattice`](GaugeSampler::lattice) and
//! [`model`](GaugeSampler::model) give owned copies, so a consumer reads them once
//! and then maps [`gauge_measure`](crate::observables::gauge_measure) or
//! [`wilson_rectangles`](crate::observables::wilson_rectangles) over the stream.
//! Which of those to call, and at what loop sizes, is not a run parameter — the
//! configurations are the same either way — which is why none of it appears in
//! the config.
//!
//! ```
//! # use plaquette::gauge_config::GaugeRunConfig;
//! # use plaquette::{GaugeSampler, gauge_measure};
//! # let run = GaugeRunConfig::parse("shape=[4,4,4]\nj=1.0\nbeta=0.75\nthermalize=10\nsweeps_between=1\nn_samples=5\nseed=1").unwrap();
//! let mut sampler = GaugeSampler::new(&run);
//! let (lattice, model) = (sampler.lattice(), sampler.model());
//! let plaquettes: Vec<f64> = sampler
//!     .samples()
//!     .take(5)
//!     .map(|c| gauge_measure(&model, &lattice, &c).plaquette_sum)
//!     .collect();
//! ```

use crate::chain::Chain;
use crate::configuration::Configuration;
use crate::gauge_config::GaugeRunConfig;
use crate::lattice::Lattice;
use crate::model::Z2Gauge;
use crate::rng::RandRng;
use crate::updater::Metropolis;

/// The dimension every gauge run is fixed at, matching [`GaugeRunConfig`].
const D: usize = 3;

/// Owns a gauge run's assembled pieces and its evolving state, thermalized and
/// ready to stream.
///
/// Fixed at `D = 3`, `Q = 2`, matching [`GaugeRunConfig`]. The pieces are held
/// loose — the configuration, the generator, and the updater separately — and a
/// transient [`Chain`] borrows them per call, which is the CPU half of what
/// [`IsingSampler`](crate::ising_sampler::IsingSampler) does behind its engine
/// enum. With one backend there is nothing to hide behind, so they sit here as
/// plain fields.
pub struct GaugeSampler {
    lattice: Lattice<D>,
    model: Z2Gauge,
    beta: f64,
    sweeps_between: usize,
    /// The only updater a gauge run has. Held as a field rather than built per
    /// call because [`Chain`] borrows it for the chain's lifetime.
    updater: Metropolis,
    /// The evolving configuration, lent to a transient [`Chain`] per call.
    state: Configuration<2>,
    rng: RandRng,
}

impl GaugeSampler {
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
    /// an updater other than Metropolis.
    pub fn new(config: &GaugeRunConfig) -> Self {
        let (lattice, model, mut rng, mut state, beta) = config.build();
        let updater = Metropolis;

        // Warm up a transient chain over the loose pieces, then stow them.
        Chain::new(&mut state, &lattice, &model, &updater, beta, &mut rng, 1)
            .advance(config.thermalize);

        GaugeSampler {
            lattice,
            model,
            beta,
            sweeps_between: config.sweeps_between,
            updater,
            state,
            rng,
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
    pub fn samples(&mut self) -> Chain<'_, 2, D, Z2Gauge, Metropolis, RandRng> {
        Chain::new(
            &mut self.state,
            &self.lattice,
            &self.model,
            &self.updater,
            self.beta,
            &mut self.rng,
            self.sweeps_between,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Start, UpdaterKind};
    use crate::configuration::Cell;
    use crate::observables::{gauge_measure, wilson_rectangles};

    fn config() -> GaugeRunConfig {
        GaugeRunConfig {
            shape: [4, 4, 4],
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

    /// The stream yields link configs of the run's lattice size, as many as
    /// asked for: 3 links per site on 64 sites.
    #[test]
    fn streams_link_configs_of_the_right_size() {
        let mut sampler = GaugeSampler::new(&config());
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

        let mut sampler = GaugeSampler::new(&run);
        let got: Vec<_> = sampler.samples().take(run.n_samples).collect();

        assert_eq!(got, expected);
    }

    /// Two samplers from the same config produce the same configurations, which
    /// is the reproducibility guarantee end to end rather than at `build` alone.
    #[test]
    fn is_reproducible_from_the_config() {
        let mut a = GaugeSampler::new(&config());
        let mut b = GaugeSampler::new(&config());
        assert_eq!(
            a.samples().take(4).collect::<Vec<_>>(),
            b.samples().take(4).collect::<Vec<_>>()
        );
    }

    /// Calling `samples()` twice continues the same chain without
    /// re-thermalizing: `n` then `m` equals one run of `n + m`.
    #[test]
    fn a_second_call_continues_the_same_chain() {
        let mut split = GaugeSampler::new(&config());
        let mut got: Vec<_> = split.samples().take(6).collect();
        got.extend(split.samples().take(4));

        let mut whole = GaugeSampler::new(&config());
        let expected: Vec<_> = whole.samples().take(10).collect();

        assert_eq!(got, expected);
    }

    /// Geometry off the sampler lets a consumer measure the stream without
    /// owning a second lattice. The plaquette sum is bounded by the plaquette
    /// count — 3 planes times 64 sites — which is the check that the field, the
    /// lattice, and the model handed over actually belong to each other.
    #[test]
    fn measures_the_stream_via_sampler_geometry() {
        let mut sampler = GaugeSampler::new(&config());
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

        let mut sampler = GaugeSampler::new(&run);
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

        let mut sampler = GaugeSampler::new(&run);
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
}
