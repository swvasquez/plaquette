//! Hypercubic-lattice geometry with periodic (toroidal) boundary conditions.
//!
//! Generic over the spatial dimension `D`. A `D`-dimensional lattice with
//! per-axis extents `shape = [L_0, ..., L_{D-1}]` indexes its sites row-major
//! (mixed-radix, lexicographic) with **axis 0 the fastest-varying**:
//!
//! ```text
//! site = x_0 + L_0 * (x_1 + L_1 * (x_2 + ...))
//! ```
//!
//! Every site has `2 * D` nearest neighbors — one forward (`+`) and one
//! backward (`−`) along each axis. They are precomputed once at construction so
//! the sampler hot path never re-derives geometry or takes a modulo per lookup.
//! Unequal extents give an anisotropic lattice (e.g. `N_t × N_s^d`).

/// Step direction along an axis: `Forward` is `+μ`, `Backward` is `−μ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Forward,
    Backward,
}

/// A `D`-dimensional hypercubic lattice with periodic boundaries.
///
/// The neighbor table is flattened with a stride of `2 * D`; within a site's row
/// column `2 * μ` is the `+μ` neighbor and column `2 * μ + 1` the `−μ` one.
#[derive(Debug, Clone)]
pub struct Lattice<const D: usize> {
    /// Extent along each axis; the number of sites is their product.
    shape: [usize; D],
    /// Flattened neighbor table, `n_sites * 2 * D` entries, stride `2 * D`.
    neighbors: Vec<usize>,
    // TODO: maybe cache axis_stride when we add coords/index conversion.
    // TODO: boundary conditions are hardcoded periodic; make swappable
    // (per-axis, at construction) — e.g. open, antiperiodic, twisted.
}

impl<const D: usize> Lattice<D> {
    /// Build a `D`-dimensional lattice with the given per-axis extents and
    /// precompute its neighbor table.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0` or if any extent is zero.
    pub fn new(shape: [usize; D]) -> Self {
        assert!(D > 0, "lattice dimension must be positive");
        assert!(
            shape.iter().all(|&l| l > 0),
            "every axis extent must be positive"
        );

        let n_sites: usize = shape.iter().product();
        let stride = 2 * D;

        // Mixed-radix place values: axis_stride[μ] is the linear-index step of a
        // one-unit move along axis μ (axis_stride[0] == 1).
        let mut axis_stride = [1usize; D];
        for mu in 1..D {
            axis_stride[mu] = axis_stride[mu - 1] * shape[mu - 1];
        }

        let mut neighbors = vec![0usize; n_sites * stride];
        for site in 0..n_sites {
            for mu in 0..D {
                let l = shape[mu];
                let s = axis_stride[mu];
                let coord = (site / s) % l;
                let base = site - coord * s;

                // Periodic wrap without a branch: add `l` before the modulo so
                // the backward step never underflows on usize.
                neighbors[site * stride + 2 * mu] = base + ((coord + 1) % l) * s;
                neighbors[site * stride + 2 * mu + 1] = base + ((coord + l - 1) % l) * s;
            }
        }

        Lattice { shape, neighbors }
    }

    /// The per-axis extents `[L_0, ..., L_{D-1}]`.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }

    /// Total number of sites, i.e. the product of the extents.
    pub fn n_sites(&self) -> usize {
        self.shape.iter().product()
    }

    /// The full `2 * D`-entry neighbor row for `site`, ordered `+0, −0, +1, ...`.
    pub fn neighbors(&self, site: usize) -> &[usize] {
        let stride = 2 * D;
        &self.neighbors[site * stride..site * stride + stride]
    }

    /// The single neighbor of `site` one step along `axis` in `dir`.
    pub fn neighbor(&self, site: usize, axis: usize, dir: Direction) -> usize {
        let offset = match dir {
            Direction::Forward => 0,
            Direction::Backward => 1,
        };
        self.neighbors[site * 2 * D + 2 * axis + offset]
    }

    /// The site reached from `site` by `delta` steps along `axis`, wrapping
    /// periodically.
    ///
    /// The general-`delta` companion to [`neighbor`](Lattice::neighbor):
    /// `shift(site, μ, 1)` is the forward neighbor and `shift(site, μ, 0)` is
    /// `site` itself. `delta` may exceed the axis extent; it is reduced modulo
    /// it.
    pub fn shift(&self, site: usize, axis: usize, delta: usize) -> usize {
        let s = self.axis_strides()[axis];
        let l = self.shape[axis];
        let coord = (site / s) % l;
        let base = site - coord * s; // strip the `axis` component, keep the rest
        base + ((coord + delta % l) % l) * s
    }

    /// Fold a coordinate tuple into its linear site index.
    pub fn index(&self, coords: [usize; D]) -> usize {
        let strides = self.axis_strides();
        let mut idx = 0;
        for mu in 0..D {
            debug_assert!(coords[mu] < self.shape[mu], "coordinate out of range");
            idx += coords[mu] * strides[mu];
        }
        idx
    }

    /// Expand a linear site index into its per-axis coordinates.
    pub fn coords(&self, site: usize) -> [usize; D] {
        let strides = self.axis_strides();
        let mut coords = [0usize; D];
        for mu in 0..D {
            coords[mu] = (site / strides[mu]) % self.shape[mu];
        }
        coords
    }

    // Mixed-radix place values; recomputed on demand (see TODO on `neighbors`).
    fn axis_strides(&self) -> [usize; D] {
        let mut strides = [1usize; D];
        for mu in 1..D {
            strides[mu] = strides[mu - 1] * self.shape[mu - 1];
        }
        strides
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_full_neighbor_table() {
        let lat = Lattice::new([4, 4]);
        assert_eq!(lat.shape, [4, 4]);
        // n_sites * 2 * D = 16 * 4.
        assert_eq!(lat.neighbors.len(), 64);
    }

    #[test]
    fn known_neighbors_on_3x3() {
        let lat = Lattice::new([3, 3]);
        // Stride 2*D = 4; columns [+x0, -x0, +x1, -x1].
        // Center site (1,1) = index 4: +x0=5, -x0=3, +x1=7, -x1=1.
        assert_eq!(&lat.neighbors[16..20], &[5, 3, 7, 1]);
        // Corner (0,0) = index 0 wraps: +x0=1, -x0=2, +x1=3, -x1=6.
        assert_eq!(&lat.neighbors[0..4], &[1, 2, 3, 6]);
    }

    #[test]
    fn generalizes_to_three_dimensions() {
        let lat = Lattice::new([2, 3, 4]);
        // 24 sites, stride 2*D = 6.
        assert_eq!(lat.neighbors.len(), 24 * 6);
        // Spot-check axis 2 at the origin; axis_stride = [1, 2, 6].
        assert_eq!(lat.neighbors[4], 6); // column 2*2 = +x2
        assert_eq!(lat.neighbors[5], 18); // column 2*2+1 = -x2, wrapping
    }

    #[test]
    #[should_panic]
    fn zero_extent_panics() {
        Lattice::new([3, 0]);
    }

    #[test]
    fn size_and_shape_accessors() {
        let lat = Lattice::new([2, 3, 4]);
        assert_eq!(lat.shape(), [2, 3, 4]);
        assert_eq!(lat.n_sites(), 24);
    }

    #[test]
    fn neighbor_accessors_agree_with_table() {
        let lat = Lattice::new([3, 3]);
        // Center site (1,1) = index 4: +x0=5, -x0=3, +x1=7, -x1=1.
        assert_eq!(lat.neighbors(4), &[5, 3, 7, 1]);
        assert_eq!(lat.neighbor(4, 0, Direction::Forward), 5);
        assert_eq!(lat.neighbor(4, 0, Direction::Backward), 3);
        assert_eq!(lat.neighbor(4, 1, Direction::Forward), 7);
        assert_eq!(lat.neighbor(4, 1, Direction::Backward), 1);
    }

    #[test]
    fn shift_wraps_periodically_along_an_axis() {
        let lat = Lattice::new([4, 4]);
        // Axis 0 (stride 1): step, full wrap, wrap+1, and the delta=0 identity.
        assert_eq!(lat.shift(0, 0, 1), 1);
        assert_eq!(lat.shift(0, 0, 4), 0); // full loop back to start
        assert_eq!(lat.shift(0, 0, 5), 1); // 5 mod 4 = 1
        assert_eq!(lat.shift(0, 0, 0), 0); // identity
        // Axis 1 (stride 4): steps move by whole rows.
        assert_eq!(lat.shift(0, 1, 1), 4);
        assert_eq!(lat.shift(0, 1, 3), 12);
        // delta = 1 must agree with the forward neighbor, for every site.
        for site in 0..lat.n_sites() {
            assert_eq!(
                lat.shift(site, 0, 1),
                lat.neighbor(site, 0, Direction::Forward)
            );
            assert_eq!(
                lat.shift(site, 1, 1),
                lat.neighbor(site, 1, Direction::Forward)
            );
        }
    }

    #[test]
    fn coords_and_index_are_inverse() {
        let lat = Lattice::new([2, 3, 4]);
        for site in 0..lat.n_sites() {
            assert_eq!(lat.index(lat.coords(site)), site);
        }
        // Spot-check a known coordinate: (1, 2, 3) with strides [1, 2, 6].
        assert_eq!(lat.index([1, 2, 3]), 1 + 2 * 2 + 3 * 6);
        assert_eq!(lat.coords(23), [1, 2, 3]);
    }
}
