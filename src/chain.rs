//! Chain: the Markov chain as a lazy [`Iterator`] over the [`Updater`] seam.
//!
//! A [`Chain`] is pure mechanism: it advances a configuration and yields
//! snapshots of it, and nothing else — no measuring, no statistics, no I/O, no
//! warmup. One step of the stream is `sweeps_between` sweeps, after which
//! [`next`](Iterator::next) hands back a clone of the current configuration. The
//! stride lives here because it defines what one step *is*; warmup is a phase of
//! a run, and does not.
//!
//! # A bare `Chain` yields pre-equilibrium configurations
//!
//! This is the one thing to know before using `Chain` directly. A Markov chain
//! starts in an arbitrary hot or cold configuration, and the early states are
//! biased by that start; they must be discarded before anything is measured, and
//! **`Chain` does not do it for you**. Warmup was deliberately kept out of the
//! constructor: fusing it in discarded the warmup states where nobody could
//! reach them, which made it impossible to *check* that thermalization worked —
//! and equilibration is something you verify by watching an observable settle,
//! not something you assume.
//!
//! The cost is that the invariant now lives in a convention rather than the
//! type. [`Sampler`](crate::sampler::Sampler) is the blessed path; reach for
//! `Chain` directly only when doing the phasing yourself.
//! [`advance`](Chain::advance) is the discarding form and allocates nothing;
//! pulling warmup configs through the iterator instead is how you keep the
//! trajectory to plot it. Successive yields stay correlated even in equilibrium,
//! which is what `sweeps_between` is for.
//!
//! # Borrows everything, owns nothing
//!
//! `Chain` borrows its configuration as well as its lattice, action, updater and
//! RNG. It is a transient view over state it does not own, so a caller keeps the
//! evolving configuration and can build a fresh chain over it once the previous
//! one is dropped — which is exactly how a warmup pass at stride 1 is followed by
//! a sampling pass at stride `sweeps_between`.
//!
//! `next` returns an owned `Configuration<Q>` rather than a borrow, because std
//! [`Iterator`] cannot lend out of `&mut self`. The clone is the product, not
//! waste: a consumer that only wants a scalar maps it away immediately and never
//! stores a config. The stream is open-ended — `next` always returns `Some` — so
//! callers bound a run with `.take(n)`.
//!
//! ```
//! # use plaquette::{Lattice, Configuration, Metropolis, measure};
//! # use plaquette::model::Ising;
//! # use plaquette::rng::RandRng;
//! # use plaquette::chain::Chain;
//! # let lat = Lattice::new([8, 8]);
//! # let model = Ising::new(1.0, 0.0);
//! # let mut rng = RandRng::seed_from_u64(0);
//! # let mut config = Configuration::<2>::hot(&lat, &mut rng);
//! # let (beta, thermalize, sweeps_between) = (1.0, 10, 2);
//! let updater = Metropolis;
//! let mut chain = Chain::new(&mut config, &lat, &model, &updater, beta, &mut rng, sweeps_between);
//!
//! // Warmup is the caller's job: advance past the approach to equilibrium,
//! // allocating nothing.
//! chain.advance(thermalize);
//!
//! // Only now measure, lazily, without ever storing the configs:
//! let samples: Vec<_> = chain.take(100).map(|c| measure(&model, &lat, &c)).collect();
//! ```

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::Action;
use crate::rng::Rng;
use crate::updater::Updater;

/// The Markov chain as a lazy iterator of [`Configuration`] snapshots.
///
/// Generic over the two seams `A: Action` and `U: Updater` and the generator
/// `R: Rng`, so the chain names the seams without naming a model or algorithm.
/// Construct with [`Chain::new`] (which runs **no** sweeps) and consume with any
/// [`Iterator`] adapter, bounding the open-ended stream with `.take(n)`.
///
/// Yields **pre-equilibrium** configurations from a fresh start — see the module
/// docs. The caller does the warmup.
pub struct Chain<'a, const Q: usize, const D: usize, A, U, R> {
    /// The evolving configuration — borrowed, and mutated in place by each sweep.
    config: &'a mut Configuration<Q>,
    /// Shared geometry: neighbor lookups for energy and updates.
    lattice: &'a Lattice<D>,
    /// The cost function pricing each proposed move.
    action: &'a A,
    /// The rule that advances the chain one sweep at a time.
    updater: &'a U,
    /// The random source driving site selection and accept/reject.
    rng: &'a mut R,
    /// Inverse temperature, passed through to the updater unchanged.
    beta: f64,
    /// Decorrelation sweeps run before each yield.
    sweeps_between: usize,
}

impl<'a, const Q: usize, const D: usize, A, U, R> Chain<'a, Q, D, A, U, R>
where
    A: Action<Q, D>,
    U: Updater<Q, D>,
    R: Rng,
{
    /// Build a chain over `config`. Runs no sweeps.
    ///
    /// Construction is a plain struct build, so the first
    /// [`next`](Iterator::next) yields whatever `config` becomes after
    /// `sweeps_between` sweeps — a pre-equilibrium state from a fresh start.
    /// `config` is borrowed mutably and advanced in place, so it outlives the
    /// chain and can be handed to another one.
    pub fn new(
        config: &'a mut Configuration<Q>,
        lattice: &'a Lattice<D>,
        action: &'a A,
        updater: &'a U,
        beta: f64,
        rng: &'a mut R,
        sweeps_between: usize,
    ) -> Self {
        Chain {
            config,
            lattice,
            action,
            updater,
            rng,
            beta,
            sweeps_between,
        }
    }

    /// The lattice this chain advances over.
    ///
    /// Returns the `'a` borrow the chain already holds, not one tied to `&self`,
    /// so a consumer can pull the geometry out *before* consuming the chain by
    /// value and use it to measure each yielded config.
    pub fn lattice(&self) -> &'a Lattice<D> {
        self.lattice
    }

    /// The action pricing this chain's moves — the model, for measurement.
    /// Returns the `'a` borrow, like [`lattice`](Chain::lattice).
    pub fn action(&self) -> &'a A {
        self.action
    }

    /// Advance the chain by `sweeps` sweeps **without producing any snapshot**.
    ///
    /// The unit is *sweeps*, not yields, so this ignores `sweeps_between`
    /// entirely. It is the primitive [`next`](Iterator::next) is built from, and
    /// the cheap way to discard the approach to equilibrium: pulling warmup
    /// configs through the iterator clones a whole configuration per pull only to
    /// free it, whereas `advance` allocates nothing.
    pub fn advance(&mut self, sweeps: usize) {
        for _ in 0..sweeps {
            self.sweep();
        }
    }

    /// One sweep of the borrowed config, in place — the single point that
    /// touches the [`Updater`] seam. `config` and `rng` are reborrowed out of
    /// their `&'a mut` fields, and those disjoint borrows alongside the shared
    /// `lattice`/`action` are what let this compile as one call.
    fn sweep(&mut self) {
        self.updater.sweep(
            &mut *self.config,
            self.lattice,
            self.action,
            self.beta,
            &mut *self.rng,
        );
    }
}

impl<const Q: usize, const D: usize, A, U, R> Iterator for Chain<'_, Q, D, A, U, R>
where
    A: Action<Q, D>,
    U: Updater<Q, D>,
    R: Rng,
{
    type Item = Configuration<Q>;

    /// Decorrelate with `sweeps_between` sweeps, then yield a clone of the current
    /// config. Always `Some`: the chain is open-ended, and callers bound it with
    /// `.take(n)`.
    fn next(&mut self) -> Option<Self::Item> {
        self.advance(self.sweeps_between);
        Some(self.config.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Ising;
    use crate::observables::{self, Sample};
    use crate::rng::RandRng;
    use crate::updater::Metropolis;
    use std::cell::Cell;

    /// An [`Updater`] that does nothing but count the sweeps it is asked to run.
    /// Its `sweep` is the whole implementation — the trait requires nothing else.
    /// The `Cell` gives interior mutability under the trait's `&self`, and reading
    /// it coexists with the chain's own `&updater`.
    struct CountingUpdater {
        sweeps: Cell<usize>,
    }

    impl<const D: usize> Updater<2, D> for CountingUpdater {
        fn sweep(
            &self,
            _config: &mut Configuration<2>,
            _lattice: &Lattice<D>,
            _action: &impl Action<2, D>,
            _beta: f64,
            _rng: &mut impl Rng,
        ) -> f64 {
            self.sweeps.set(self.sweeps.get() + 1);
            0.0
        }
    }

    /// Construction runs **zero** sweeps, and each `next()` then runs exactly
    /// `sweeps_between` of them.
    #[test]
    fn constructs_without_sweeping_and_decorrelates_each_next() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat);
        let updater = CountingUpdater {
            sweeps: Cell::new(0),
        };
        let mut rng = RandRng::seed_from_u64(0);

        let sweeps_between = 3;
        let mut chain = Chain::new(
            &mut config,
            &lat,
            &action,
            &updater,
            1.0,
            &mut rng,
            sweeps_between,
        );

        assert_eq!(updater.sweeps.get(), 0);
        assert!(chain.next().is_some());
        assert_eq!(updater.sweeps.get(), sweeps_between);
        assert!(chain.next().is_some());
        assert_eq!(updater.sweeps.get(), 2 * sweeps_between);
    }

    /// `.take(n)` yields exactly `n` configs, and the whole run performs
    /// `n * sweeps_between` sweeps — no warmup is folded in.
    #[test]
    fn take_n_yields_exactly_n_configs() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat);
        let updater = CountingUpdater {
            sweeps: Cell::new(0),
        };
        let mut rng = RandRng::seed_from_u64(0);

        let (sweeps_between, n) = (3, 10);
        let chain = Chain::new(
            &mut config,
            &lat,
            &action,
            &updater,
            1.0,
            &mut rng,
            sweeps_between,
        );

        let configs: Vec<_> = chain.take(n).collect();

        assert_eq!(configs.len(), n);
        assert_eq!(updater.sweeps.get(), n * sweeps_between);
    }

    /// Consumer pattern: measure lazily with `.map`, never storing configs. The
    /// closure captures `model` and `lattice` and turns each yielded config into
    /// a [`Sample`].
    #[test]
    fn take_map_measure_collects_n_samples() {
        let lat = Lattice::new([8, 8]);
        let model = Ising::new(1.0, 0.0);
        let updater = Metropolis;
        let mut rng = RandRng::seed_from_u64(7);
        let mut config = Configuration::<2>::hot(&lat, &mut rng);

        let n = 12;
        let chain = Chain::new(&mut config, &lat, &model, &updater, 1.0, &mut rng, 2);

        let samples: Vec<Sample> = chain
            .take(n)
            .map(|c| observables::measure(&model, &lat, &c))
            .collect();

        assert_eq!(samples.len(), n);
    }

    /// Consumer pattern: keep the configs themselves with `.collect`, each a full
    /// lattice snapshot.
    #[test]
    fn take_collect_gives_a_vec_of_configs() {
        let lat = Lattice::new([8, 8]);
        let model = Ising::new(1.0, 0.0);
        let updater = Metropolis;
        let mut rng = RandRng::seed_from_u64(7);
        let mut config = Configuration::<2>::hot(&lat, &mut rng);

        let n = 12;
        let chain = Chain::new(&mut config, &lat, &model, &updater, 1.0, &mut rng, 2);

        let configs: Vec<Configuration<2>> = chain.take(n).collect();

        assert_eq!(configs.len(), n);
        assert!(configs.iter().all(|c| c.n_sites() == lat.n_sites()));
    }

    /// End-to-end physics sanity: a low-temperature (`β = 1`, well below
    /// `β_c ≈ 0.44`) chain from a hot start trends toward alignment. Onsager's
    /// spontaneous magnetization there is `|m| ≈ 0.999`, so a conservative
    /// `> 0.5` threshold is a robust, seed-deterministic check that the chain
    /// equilibrates into the ordered phase.
    #[test]
    fn low_temperature_chain_trends_toward_alignment() {
        let lat = Lattice::new([16, 16]);
        let model = Ising::new(1.0, 0.0);
        let updater = Metropolis;
        let mut rng = RandRng::seed_from_u64(20260718);
        let mut config = Configuration::<2>::hot(&lat, &mut rng);

        let n = 50;
        let n_sites = lat.n_sites() as f64;
        let mut chain = Chain::new(&mut config, &lat, &model, &updater, 1.0, &mut rng, 1);

        // Discard the approach to equilibrium; measuring without this would
        // average in the disordered approach and fail.
        chain.advance(200);

        // Average |m| = |M| / N, sign-folded because the ordered phase picks
        // either well.
        let mean_abs_m = chain
            .take(n)
            .map(|c| (model.magnetization(&c) / n_sites).abs())
            .sum::<f64>()
            / n as f64;

        assert!(
            mean_abs_m > 0.5,
            "low-T chain should order: mean |m| = {mean_abs_m}"
        );
    }
}
