//! A single field variable valued in a finite set of `Q` states.
//!
//! The index is validated at construction, so a `State<Q>` can only ever hold an
//! in-range value. What the indices *mean* — Ising's `0 → +1, 1 → -1`, Potts
//! labels, `q`-th roots of unity — is the energy's concern, not this type's.

/// One value of a field valued in a finite set of `Q` states, stored as its
/// index in `0..Q`. `Q` must be in `1..=256` (the index is a single byte).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct State<const Q: usize>(u8);

impl<const Q: usize> State<Q> {
    /// The state with the given `index`, or `None` if `index >= Q`. The only
    /// constructor, so `index < Q` holds for every value that exists.
    pub fn new(index: usize) -> Option<Self> {
        debug_assert!((1..=256).contains(&Q), "Q must be in 1..=256");
        if index < Q {
            Some(State(index as u8))
        } else {
            None
        }
    }

    /// The state's index in `0..Q`.
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_range_index_constructs() {
        assert_eq!(State::<3>::new(0).map(|s| s.index()), Some(0));
        assert_eq!(State::<3>::new(2).map(|s| s.index()), Some(2));
    }

    #[test]
    fn out_of_range_index_is_rejected() {
        assert_eq!(State::<3>::new(3), None);
        assert_eq!(State::<2>::new(7), None);
    }
}
