//! Random-number seam.
//!
//! [`Rng`] is a minimal generator interface — just the draws the sampler needs.
//! Downstream code depends on this trait, never on a concrete generator, so the
//! backend is swappable: [`RandRng`] wraps the `rand` ecosystem today, a
//! vendored PRNG could implement the same trait without touching any caller.

/// The random draws the sampler and initializers need. Implementors supply
/// `next_f64` and `next_below`; `coin` defaults in terms of `next_f64`.
pub trait Rng {
    /// A uniform value in `[0, 1)`.
    fn next_f64(&mut self) -> f64;

    /// A uniform integer in `0..n`.
    ///
    /// # Panics
    ///
    /// May panic if `n == 0` (there is no value to return).
    fn next_below(&mut self, n: usize) -> usize;

    /// A fair coin: `true` and `false` with equal probability.
    fn coin(&mut self) -> bool {
        self.next_f64() < 0.5
    }
}

/// An [`Rng`] backed by the `rand` crate's PCG64 generator — the only place the
/// `rand` ecosystem is named. Seeded explicitly, so a run is reproducible from a
/// `u64` that can be recorded as metadata.
pub struct RandRng(rand_pcg::Pcg64);

impl RandRng {
    /// Seed the generator from a `u64` for reproducible runs.
    pub fn seed_from_u64(seed: u64) -> Self {
        use rand::SeedableRng;
        RandRng(rand_pcg::Pcg64::seed_from_u64(seed))
    }
}

impl Rng for RandRng {
    fn next_f64(&mut self) -> f64 {
        use rand::RngExt as _;
        self.0.random::<f64>()
    }

    fn next_below(&mut self, n: usize) -> usize {
        use rand::RngExt as _;
        self.0.random_range(0..n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_stream() {
        let mut a = RandRng::seed_from_u64(42);
        let mut b = RandRng::seed_from_u64(42);
        for _ in 0..16 {
            assert_eq!(a.next_f64(), b.next_f64());
        }
    }

    #[test]
    fn floats_are_in_unit_interval() {
        let mut rng = RandRng::seed_from_u64(7);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
        }
    }

    #[test]
    fn next_below_stays_in_range() {
        let mut rng = RandRng::seed_from_u64(1);
        for _ in 0..1000 {
            assert!(rng.next_below(5) < 5);
        }
    }
}
