//! Sampler: the orchestrator that owns a warmed-up run and streams from it.
//!
//! Where [`Chain`] is pure mechanism, `Sampler` owns the **phasing**. It
//! assembles the pieces a run needs, thermalizes once in its constructor, and
//! then lends chains from its warmed-up state. It is the blessed path: a bare
//! `Chain` yields pre-equilibrium states, a `Sampler` has already thermalized
//! before it gives you anything.
//!
//! It streams and keeps no history. [`samples`](Sampler::samples) lends a
//! [`Chain`] over the sampler's evolving state, and the consumer decides what to
//! retain — fold into running sums, flush batches to a file, or collect when the
//! run is small. The sampler holds only the chain position (state and RNG), so it
//! stays `O(L²)` however long the run goes.
//!
//! That also makes extend-and-recheck free: because the warmed-up state stays
//! here, `samples` can be called again to draw more after looking at the error
//! bars, and the second batch continues the same chain without re-thermalizing.
//! Two calls of `n` and `m` produce exactly what one call of `n + m` would.
//!
//! `Sampler` owns its pieces because `Chain` borrows all of them and so cannot be
//! returned from the function that builds them — which is why
//! [`RunConfig::build`] hands back loose parts. A `Sampler` cannot hold a `Chain`
//! as a field either (a struct cannot reference its own siblings), so `samples`
//! builds a transient chain and lends its fields to it. The warmup trajectory is
//! not exposed here; anyone wanting to watch an observable settle drives a
//! `Chain` directly.

use crate::chain::Chain;
use crate::config::{RunConfig, UpdaterKind};
use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::Ising;
use crate::rng::RandRng;
use crate::updater::{AnyUpdater, Checkerboard, Metropolis};

/// The concrete `Chain` a [`Sampler`] lends, named so
/// [`samples`](Sampler::samples)'s return type stays legible.
pub type SampleChain<'a> = Chain<'a, 2, 2, Ising, AnyUpdater, RandRng>;

/// Owns a run's assembled pieces and its evolving state, thermalized and ready
/// to stream.
///
/// Fixed at `D = 2`, `Q = 2`, matching [`RunConfig`]. The updater is an
/// [`AnyUpdater`] — the runtime choice among the built-in algorithms — so the
/// [`UpdaterKind`] read from a config file selects one without that type leaking
/// into `Sampler`'s or [`Chain`]'s signature.
pub struct Sampler {
    lattice: Lattice<2>,
    model: Ising,
    rng: RandRng,
    updater: AnyUpdater,
    /// The evolving chain state, lent to a transient [`Chain`] per call.
    state: Configuration<2>,
    beta: f64,
    sweeps_between: usize,
}

impl Sampler {
    /// Assemble a run from its config and thermalize it.
    ///
    /// Runs `config.thermalize` warmup sweeps and discards them, so the sampler
    /// is at equilibrium before it lends anything. [`RunConfig::build`] seeds the
    /// generator *before* drawing a [`Start::Hot`](crate::config::Start::Hot)
    /// configuration, which is what makes the whole run replay from the seed
    /// alone.
    ///
    /// `config.n_samples` is deliberately not read here: how many samples to draw
    /// is the consumer's `.take(n)`.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid, via `build`. Configs from
    /// [`load`](RunConfig::load) and [`parse`](RunConfig::parse) are already
    /// validated.
    pub fn new(config: &RunConfig) -> Self {
        let (lattice, model, mut rng, mut state, beta) = config.build();
        let updater = match config.updater {
            UpdaterKind::Metropolis => AnyUpdater::Metropolis(Metropolis),
            UpdaterKind::Checkerboard => AnyUpdater::Checkerboard(Checkerboard),
        };

        // `advance` is in sweeps and produces no snapshots, so the stride of 1 is
        // irrelevant and nothing is allocated. The chain drops at the end of the
        // statement, returning the borrows of `state` and `rng`.
        Chain::new(&mut state, &lattice, &model, &updater, beta, &mut rng, 1)
            .advance(config.thermalize);

        Sampler {
            lattice,
            model,
            rng,
            updater,
            state,
            beta,
            sweeps_between: config.sweeps_between,
        }
    }

    /// Lend a [`Chain`] over the warmed-up state.
    ///
    /// Bound it with `.take(n)` and route each yielded config wherever it should
    /// go; the sampler retains nothing, and calling this again continues the same
    /// chain. To measure the stream, pull the geometry off the chain first —
    /// [`Chain::lattice`] and [`Chain::action`] return `'a` borrows that survive
    /// the chain being consumed:
    ///
    /// ```
    /// # use plaquette::config::RunConfig;
    /// # use plaquette::{measure, Sampler};
    /// # let run = RunConfig::parse("shape=[8,8]\nj=1.0\nbeta=0.44\nthermalize=10\nsweeps_between=1\nn_samples=5\nseed=1").unwrap();
    /// let mut sampler = Sampler::new(&run);
    /// let chain = sampler.samples();
    /// let (lattice, model) = (chain.lattice(), chain.action());
    /// let energies: Vec<f64> = chain.take(5).map(|c| measure(model, lattice, &c).energy).collect();
    /// ```
    pub fn samples(&mut self) -> SampleChain<'_> {
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
    use crate::config::Start;
    use crate::observables::measure;

    fn config() -> RunConfig {
        RunConfig {
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
        let mut sampler = Sampler::new(&config());
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_sites() == 64));
    }

    /// A checkerboard-configured run streams too: the `UpdaterKind` from the
    /// config selects `AnyUpdater::Checkerboard`, which the sampler drives exactly
    /// like any other updater.
    #[test]
    fn streams_with_the_checkerboard_updater() {
        let mut run = config();
        run.updater = UpdaterKind::Checkerboard;

        let mut sampler = Sampler::new(&run);
        let configs: Vec<_> = sampler.samples().take(5).collect();

        assert_eq!(configs.len(), 5);
        assert!(configs.iter().all(|c| c.n_sites() == 64));
    }

    /// The geometry accessors let a consumer measure the stream without owning a
    /// second lattice.
    #[test]
    fn measures_the_stream_via_chain_accessors() {
        let mut sampler = Sampler::new(&config());
        let chain = sampler.samples();
        let (lattice, model) = (chain.lattice(), chain.action());
        let n_sites = lattice.n_sites() as f64;

        let energies: Vec<f64> = chain
            .take(3)
            .map(|c| measure(model, lattice, &c).energy)
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

        let mut sampler = Sampler::new(&run);
        let got: Vec<_> = sampler.samples().take(run.n_samples).collect();

        assert_eq!(got, expected);
    }

    /// Calling `samples()` twice continues the same chain without re-thermalizing:
    /// `n` then `m` equals one run of `n + m`.
    #[test]
    fn a_second_call_continues_the_same_chain() {
        let mut split = Sampler::new(&config());
        let mut got: Vec<_> = split.samples().take(6).collect();
        got.extend(split.samples().take(4));

        let mut whole = Sampler::new(&config());
        let expected: Vec<_> = whole.samples().take(10).collect();

        assert_eq!(got, expected);
    }
}
