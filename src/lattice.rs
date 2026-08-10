//! Hypercubic-lattice geometry with periodic (toroidal) boundary conditions,
//! generic over the spatial dimension `D`.
//!
//! Sites, links, and plaquettes each pack into a linear index, incidence
//! accessors follow the `a_bs(a)` naming rule, and staples are precomputed per
//! link. The packing orders, reserved vocabulary, and incidence contracts are
//! the labeled requirements `L0`–`L7` in `docs/lattice-implementation.md`,
//! which is the reference for this module.

/// The sign of a step along a direction: `Plus` is `+μ`, `Minus` is `−μ`.
/// Named apart from *direction* per `L0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    /// A step along `+μ`.
    Plus,
    /// A step along `−μ`.
    Minus,
}

/// A `D`-dimensional hypercubic lattice with periodic boundaries.
///
/// Two tables are precomputed and flattened with a fixed stride. The neighbor
/// table has stride `2 * D`; within a site's row, column `2 * μ` is the `+μ`
/// neighbor and column `2 * μ + 1` the `−μ` one. The staple table has stride
/// `6 * (D - 1)` per link, read as groups of three.
#[derive(Debug, Clone)]
pub struct Lattice<const D: usize> {
    /// Extent along each direction; the number of sites is their product.
    shape: [usize; D],
    /// Flattened neighbor table, `n_sites * 2 * D` entries, stride `2 * D`.
    neighbors: Vec<usize>,
    /// Flattened staple table, stride `6 * (D - 1)` per link, in groups of
    /// three per containing plaquette.
    staples: Vec<usize>,
    /// Coordinate-sum parity per site, `0` or `1` — the checkerboard color.
    /// A table because the checkerboard schedules read it twice per variable
    /// update, and computing it via [`site_coords`](Lattice::site_coords)
    /// measured at roughly a third of a whole CPU sweep.
    parities: Vec<u8>,
    // TODO: `dir_strides` is recomputed on every `site_index`, `site_coords`,
    // and `site_shift` call; cache it here if any of them lands in a hot loop.
    // TODO: boundary conditions are hardcoded periodic; make swappable
    // (per-direction, at construction) — e.g. open, antiperiodic, twisted.
}

impl<const D: usize> Lattice<D> {
    /// Build a `D`-dimensional lattice with the given per-direction extents and
    /// precompute its tables.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0`, if any extent is zero, or if the shape names more
    /// sites or table entries than a `usize` can address (see `table_len` for
    /// why that last one must be caught here).
    pub fn new(shape: [usize; D]) -> Self {
        assert!(D > 0, "lattice dimension must be positive");
        assert!(
            shape.iter().all(|&l| l > 0),
            "every extent must be positive"
        );

        let n_sites = shape
            .iter()
            .try_fold(1usize, |volume, &l| volume.checked_mul(l))
            .unwrap_or_else(|| {
                panic!(
                    "shape{shape:?} names more sites than a usize can count; \
                     reduce the extents or the dimension"
                )
            });
        let stride = Self::neighbor_stride();

        // Mixed-radix place values (L1); each is a partial product of the
        // extents, so all of them fit once `n_sites` does.
        let mut dir_stride = [1usize; D];
        for mu in 1..D {
            dir_stride[mu] = dir_stride[mu - 1] * shape[mu - 1];
        }

        let mut neighbors =
            vec![0usize; Self::table_len(n_sites, stride, &shape, "neighbor table")];
        // Filled in the same pass, which already decomposes the site index.
        let mut parities = vec![0u8; n_sites];
        for site in 0..n_sites {
            let mut coord_sum = 0usize;
            for mu in 0..D {
                let l = shape[mu];
                let s = dir_stride[mu];
                let coord = (site / s) % l;
                let base = site - coord * s;
                coord_sum += coord;

                // Periodic wrap without a branch: add `l` before the modulo so
                // the backward step never underflows on usize.
                neighbors[site * stride + 2 * mu] = base + ((coord + 1) % l) * s;
                neighbors[site * stride + 2 * mu + 1] = base + ((coord + l - 1) % l) * s;
            }
            parities[site] = (coord_sum % 2) as u8;
        }

        // The staple builder reads the neighbor table through `&self`, so
        // build the lattice first and fill the table in afterwards.
        let mut lattice = Lattice {
            shape,
            neighbors,
            staples: Vec::new(),
            parities,
        };
        lattice.staples = lattice.build_staples();
        lattice
    }

    /// A table of `count` rows `width` entries wide, or a panic naming the
    /// lattice if that length does not fit in a `usize`.
    ///
    /// The overflow this guards is silent in release builds: a wrapped length
    /// is *small*, so the allocation succeeds and the lattice comes out quietly
    /// empty with every other invariant intact. A panic rather than an error
    /// because it is the same kind of fault as a zero extent: a shape nobody
    /// could run, caught at the one door geometry comes through.
    fn table_len(count: usize, width: usize, shape: &[usize; D], what: &str) -> usize {
        count.checked_mul(width).unwrap_or_else(|| {
            panic!(
                "the {what} for shape{shape:?} in {D} dimensions needs \
                 {count} x {width} entries, more than a usize can address; \
                 reduce the extents or the dimension"
            )
        })
    }

    /// Invert the plaquette enumeration into the per-link staple table.
    ///
    /// Walking plaquettes in index order keeps the table consistent with
    /// [`plaquette_links`](Lattice::plaquette_links) by construction and makes
    /// groups land in increasing plaquette index (`L6`).
    fn build_staples(&self) -> Vec<usize> {
        let stride = Self::staple_stride();
        let n_links = Self::table_len(self.n_sites(), D, &self.shape, "link index space");
        let mut staples =
            vec![0usize; Self::table_len(n_links, stride, &self.shape, "staple table")];
        // Next free group slot within each link's row.
        let mut filled = vec![0usize; n_links];

        for plaquette in 0..self.n_plaquettes() {
            let links = self.plaquette_links(plaquette);
            for (i, &link) in links.iter().enumerate() {
                let base = link * stride + filled[link] * 3;
                let others = links.iter().enumerate().filter(|&(j, _)| j != i);
                for (slot, (_, &other)) in others.enumerate() {
                    staples[base + slot] = other;
                }
                filled[link] += 1;
            }
        }

        debug_assert!(
            filled
                .iter()
                .all(|&groups| groups == Self::plaquettes_per_link()),
            "every link must belong to exactly 2 * (D - 1) plaquettes"
        );
        staples
    }

    /// The per-direction extents `[L_0, ..., L_{D-1}]`.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }
}

/// Counting and naming the cells, per the packings `L1`–`L3` in
/// `docs/lattice-implementation.md`. All of this is arithmetic on `shape`;
/// none of it reads a table.
impl<const D: usize> Lattice<D> {
    /// Total number of sites, i.e. the product of the extents.
    pub fn n_sites(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total number of links: one forward edge per site per direction,
    /// `D * n_sites`.
    pub fn n_links(&self) -> usize {
        D * self.n_sites()
    }

    /// Total number of plaquettes: one per site per unordered direction pair,
    /// `C(D, 2) * n_sites`. Zero when `D < 2`, where no pair exists.
    pub fn n_plaquettes(&self) -> usize {
        Self::n_dir_pairs() * self.n_sites()
    }

    /// Fold a coordinate tuple into its linear site index.
    pub fn site_index(&self, coords: [usize; D]) -> usize {
        let strides = self.dir_strides();
        let mut idx = 0;
        for mu in 0..D {
            debug_assert!(coords[mu] < self.shape[mu], "coordinate out of range");
            idx += coords[mu] * strides[mu];
        }
        idx
    }

    /// Expand a linear site index into its per-direction coordinates; the
    /// inverse of [`site_index`](Lattice::site_index).
    pub fn site_coords(&self, site: usize) -> [usize; D] {
        let strides = self.dir_strides();
        let mut coords = [0usize; D];
        for mu in 0..D {
            coords[mu] = (site / strides[mu]) % self.shape[mu];
        }
        coords
    }

    /// Linear index of the link based at `coords` and running along `dir`, i.e.
    /// the forward edge from that site to its `+dir` neighbor.
    pub fn link_index(&self, coords: [usize; D], dir: usize) -> usize {
        self.site_link(self.site_index(coords), dir)
    }

    /// The site's checkerboard color: the parity of its coordinate sum, `0` or
    /// `1`. Two sites of the same color are never nearest neighbors — but only
    /// when every extent is even, since an odd wrap joins same-parity sites.
    pub fn site_parity(&self, site: usize) -> usize {
        self.parities[site] as usize
    }

    /// The base site of `link`, i.e. the end it points away from.
    pub fn link_site(&self, link: usize) -> usize {
        self.link_base(link).0
    }

    /// The direction `link` runs along. There is deliberately no accessor
    /// returning the site and direction together (`L0`).
    pub fn link_direction(&self, link: usize) -> usize {
        self.link_base(link).1
    }

    /// Linear index of the unit square based at `coords` and spanning
    /// directions `mu` and `nu`, which must satisfy `mu < nu`.
    pub fn plaquette_index(&self, coords: [usize; D], mu: usize, nu: usize) -> usize {
        self.plaquette_at(self.site_index(coords), mu, nu)
    }

    /// The base site of `plaquette`, the corner it extends away from.
    pub fn plaquette_site(&self, plaquette: usize) -> usize {
        self.plaquette_base(plaquette).0
    }

    /// The two directions `plaquette` spans, `(μ, ν)` with `μ < ν`.
    pub fn plaquette_directions(&self, plaquette: usize) -> [usize; 2] {
        let (_, mu, nu) = self.plaquette_base(plaquette);
        [mu, nu]
    }

    /// The link leaving `site` along `dir`, packed `site * D + dir` (`L2`):
    /// the form for a caller that already holds a site index, where
    /// [`link_index`](Lattice::link_index)'s coordinate round trip would cost
    /// a divide per axis.
    pub fn site_link(&self, site: usize, dir: usize) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(dir < D, "direction out of range");
        site * D + dir
    }

    /// Split a link index into its base site and direction.
    fn link_base(&self, link: usize) -> (usize, usize) {
        debug_assert!(link < self.n_links(), "link out of range");
        (link / D, link % D)
    }

    /// The plaquette at `site` spanning `(mu, nu)`, packed
    /// `site * C(D, 2) + pair` with `pair` lexicographic (`L3`).
    fn plaquette_at(&self, site: usize, mu: usize, nu: usize) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(mu < nu && nu < D, "direction pair must satisfy mu < nu < D");
        site * Self::n_dir_pairs() + Self::pair_index(mu, nu)
    }

    /// Split a plaquette index into its base site and direction pair.
    fn plaquette_base(&self, plaquette: usize) -> (usize, usize, usize) {
        debug_assert!(plaquette < self.n_plaquettes(), "plaquette out of range");
        let n_pairs = Self::n_dir_pairs();
        let (mu, nu) = Self::dir_pair(plaquette % n_pairs);
        (plaquette / n_pairs, mu, nu)
    }

    /// Number of unordered direction pairs `(μ, ν)` with `μ < ν`, i.e. of
    /// plaquettes anchored at each site.
    const fn n_dir_pairs() -> usize {
        D * D.saturating_sub(1) / 2
    }

    /// The direction pair `(μ, ν)`, `μ < ν`, at ordinal `pair` in lexicographic
    /// order, found by walking the group sizes `D - 1, D - 2, ...` (see `L3`).
    const fn dir_pair(pair: usize) -> (usize, usize) {
        debug_assert!(pair < Self::n_dir_pairs(), "direction pair out of range");
        let mut rest = pair;
        let mut mu = 0;
        loop {
            let in_group = D - 1 - mu; // pairs sharing this first direction
            if rest < in_group {
                return (mu, mu + 1 + rest);
            }
            rest -= in_group;
            mu += 1;
        }
    }

    /// The ordinal of `(mu, nu)`, `mu < nu`; the inverse of `dir_pair`. The
    /// formula is derived in `L3` of `docs/lattice-implementation.md`.
    const fn pair_index(mu: usize, nu: usize) -> usize {
        mu * (D - 1) - mu * mu.saturating_sub(1) / 2 + (nu - mu - 1)
    }

    // Mixed-radix place values; recomputed on demand (see TODO on `neighbors`).
    fn dir_strides(&self) -> [usize; D] {
        let mut strides = [1usize; D];
        for mu in 1..D {
            strides[mu] = strides[mu - 1] * self.shape[mu - 1];
        }
        strides
    }
}

/// Moving between sites: displacements along a direction, wrapping
/// periodically.
impl<const D: usize> Lattice<D> {
    /// The full `2 * D`-entry neighbor row for `site`, ordered `+0, −0, +1,
    /// ...`.
    pub fn site_neighbors(&self, site: usize) -> &[usize] {
        debug_assert!(site < self.n_sites(), "site out of range");
        let stride = 2 * D;
        &self.neighbors[site * stride..site * stride + stride]
    }

    /// The single neighbor of `site` one step along `dir`, on the `sign` side.
    ///
    /// The bounds check matters more here than elsewhere: a `dir` of `D` or
    /// more stays inside the table and reads the next site's row, returning a
    /// plausible site index rather than failing.
    pub fn site_neighbor(&self, site: usize, dir: usize, sign: Sign) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(dir < D, "direction out of range");
        let offset = match sign {
            Sign::Plus => 0,
            Sign::Minus => 1,
        };
        self.neighbors[site * 2 * D + 2 * dir + offset]
    }

    /// The site reached from `site` by `delta` steps along `dir`, wrapping
    /// periodically; `delta` may exceed the extent and is reduced modulo it.
    pub fn site_shift(&self, site: usize, dir: usize, delta: usize) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(dir < D, "direction out of range");
        let s = self.dir_strides()[dir];
        let l = self.shape[dir];
        let coord = (site / s) % l;
        let base = site - coord * s; // strip the `dir` component, keep the rest
        base + ((coord + delta % l) % l) * s
    }
}

/// Incidence between the cell kinds: `a_bs(a)` is the set of `b`-cells touching
/// the `a`-cell `a`. Why [`site_links`](Lattice::site_links) iterates while
/// [`link_plaquettes`](Lattice::link_plaquettes) collects is explained in the
/// Storage section of `docs/lattice-implementation.md`.
impl<const D: usize> Lattice<D> {
    /// The `2 * D` links touching `site`: for each direction the forward link
    /// leaving it and the backward link arriving at it, ordered `+0, −0, +1,
    /// ...` to match [`site_neighbors`](Lattice::site_neighbors). On an
    /// extent-1 direction the two coincide and the link is yielded twice.
    pub fn site_links(&self, site: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        debug_assert!(site < self.n_sites(), "site out of range");
        // The forward link is based here, the backward one at the neighbor it
        // arrives from.
        (0..2 * D).map(move |i| {
            let dir = i / 2;
            if i % 2 == 0 {
                self.site_link(site, dir)
            } else {
                self.site_link(self.site_neighbor(site, dir, Sign::Minus), dir)
            }
        })
    }

    /// The two endpoints of `link`, base site first.
    pub fn link_sites(&self, link: usize) -> [usize; 2] {
        let (site, dir) = self.link_base(link);
        [site, self.site_neighbor(site, dir, Sign::Plus)]
    }

    /// The `2 * (D - 1)` plaquettes containing `link`, in increasing index
    /// order — the same order as the groups of
    /// [`link_staples`](Lattice::link_staples), so the two can be zipped
    /// (`L6`). Allocates and sorts, so it belongs outside a hot path.
    ///
    /// On an extent-1 direction the same plaquette appears twice, which still
    /// lines up with the staple groups, since those are counted by slot rather
    /// than by distinct plaquette.
    pub fn link_plaquettes(&self, link: usize) -> Vec<usize> {
        let (site, dir) = self.link_base(link);
        let mut plaquettes = Vec::with_capacity(Self::plaquettes_per_link());
        for other in (0..D).filter(|&other| other != dir) {
            let (mu, nu) = if dir < other {
                (dir, other)
            } else {
                (other, dir)
            };
            let back = self.site_neighbor(site, other, Sign::Minus);
            plaquettes.push(self.plaquette_at(site, mu, nu));
            plaquettes.push(self.plaquette_at(back, mu, nu));
        }
        plaquettes.sort_unstable();
        plaquettes
    }

    /// The four links bounding `plaquette`, in the fixed loop order `[(s, μ),
    /// (s + μ̂, ν), (s + ν̂, μ), (s, ν)]` (`L4`).
    pub fn plaquette_links(&self, plaquette: usize) -> [usize; 4] {
        let (site, mu, nu) = self.plaquette_base(plaquette);
        let site_mu = self.site_neighbor(site, mu, Sign::Plus);
        let site_nu = self.site_neighbor(site, nu, Sign::Plus);
        [
            self.site_link(site, mu),
            self.site_link(site_mu, nu),
            self.site_link(site_nu, mu),
            self.site_link(site, nu),
        ]
    }

    /// The staples of `link`: for each of the `2 * (D - 1)` plaquettes
    /// containing it, that plaquette's three other links, flattened into `6 *
    /// (D - 1)` entries. Chunk it by three to recover the groups, which come in
    /// increasing plaquette index (`L6`); a single-link flip reads its energy
    /// change from this table alone (`L7`).
    pub fn link_staples(&self, link: usize) -> &[usize] {
        debug_assert!(link < self.n_links(), "link out of range");
        let stride = Self::staple_stride();
        &self.staples[link * stride..link * stride + stride]
    }

    /// Entries per link in the staple table: `2 * (D - 1)` groups of three.
    pub(crate) const fn staple_stride() -> usize {
        3 * Self::plaquettes_per_link()
    }

    /// Plaquettes containing a given link, `2 * (D - 1)`: one on each side of it
    /// in each of the other `D - 1` directions.
    pub(crate) const fn plaquettes_per_link() -> usize {
        2 * D.saturating_sub(1)
    }

    /// Entries per site in the neighbor table: one forward and one backward
    /// along each direction. Named because other layers upload and index rows
    /// of this width.
    pub(crate) const fn neighbor_stride() -> usize {
        2 * D
    }

    /// The links `path` crosses when walked from `base`, in traversal order.
    ///
    /// A forward step crosses the link based at the site it leaves; a backward
    /// step crosses the link based at the site it *arrives* at, since links are
    /// named by their forward end. Which way a link was crossed is not reported
    /// (see the Loops section of `docs/lattice-implementation.md`).
    pub fn loop_links<'a>(
        &'a self,
        base: usize,
        path: &'a Loop<D>,
    ) -> impl Iterator<Item = usize> + 'a {
        debug_assert!(base < self.n_sites(), "site out of range");
        path.steps().iter().scan(base, move |site, &(dir, sign)| {
            Some(match sign {
                Sign::Plus => {
                    let link = self.site_link(*site, dir);
                    *site = self.site_neighbor(*site, dir, Sign::Plus);
                    link
                }
                Sign::Minus => {
                    *site = self.site_neighbor(*site, dir, Sign::Minus);
                    self.site_link(*site, dir)
                }
            })
        })
    }
}

/// A closed path on the lattice: steps of direction and [`Sign`], with no base
/// site, closed when the net displacement vanishes modulo the extents (so
/// winding paths count). The Loops section of `docs/lattice-implementation.md`
/// explains both choices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop<const D: usize> {
    /// The steps in traversal order, each a direction and the sense of the move
    /// along it.
    steps: Vec<(usize, Sign)>,
}

impl<const D: usize> Loop<D> {
    /// Build the loop traced by `steps`, or `None` if some direction is out of
    /// range or the path does not close.
    ///
    /// The lattice is borrowed only to be measured, not kept, so a loop is
    /// valid for the extents it was built against: walking it on a lattice of
    /// different shape is meaningful only if it closes there too.
    pub fn new(lattice: &Lattice<D>, steps: &[(usize, Sign)]) -> Option<Self> {
        let shape = lattice.shape();

        let mut net = [0isize; D];
        for &(dir, sign) in steps {
            if dir >= D {
                return None;
            }
            net[dir] += match sign {
                Sign::Plus => 1,
                Sign::Minus => -1,
            };
        }

        let closes = (0..D).all(|mu| net[mu].rem_euclid(shape[mu] as isize) == 0);
        closes.then(|| Loop {
            steps: steps.to_vec(),
        })
    }

    /// The rectangle running `r` steps along `mu` and `t` along `nu` before
    /// retracing both, or `None` if the two directions are not distinct, one is
    /// out of range, or a side is long enough to wrap. Both refusals are
    /// degeneracies [`new`](Loop::new) would accept; the Loops section of
    /// `docs/lattice-implementation.md` explains why they are refused here.
    pub fn rectangle(
        lattice: &Lattice<D>,
        mu: usize,
        r: usize,
        nu: usize,
        t: usize,
    ) -> Option<Self> {
        let shape = lattice.shape();
        if mu == nu || mu >= D || nu >= D || r >= shape[mu] || t >= shape[nu] {
            return None;
        }

        let mut steps = Vec::with_capacity(2 * (r + t));
        for (dir, len, sign) in [
            (mu, r, Sign::Plus),
            (nu, t, Sign::Plus),
            (mu, r, Sign::Minus),
            (nu, t, Sign::Minus),
        ] {
            steps.extend(std::iter::repeat_n((dir, sign), len));
        }

        Loop::new(lattice, &steps)
    }

    /// The steps in traversal order.
    pub fn steps(&self) -> &[(usize, Sign)] {
        &self.steps
    }

    /// The number of links the path crosses, counted with multiplicity — the
    /// loop's perimeter.
    pub fn perimeter(&self) -> usize {
        self.steps.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn builds_full_neighbor_table() {
        let lat = Lattice::new([4, 4]);
        assert_eq!(lat.shape, [4, 4]);
        // n_sites * 2 * D = 16 * 4.
        assert_eq!(lat.neighbors.len(), 64);
    }

    /// The parity table agrees with the coordinate sum it caches, including on
    /// odd extents, where the wrap puts same-parity sites next to each other
    /// and the table must still report the honest value.
    #[test]
    fn parity_table_matches_the_coordinate_sum() {
        for lat in [Lattice::new([4, 6, 2]), Lattice::new([3, 5, 3])] {
            for site in 0..lat.n_sites() {
                let expected = lat.site_coords(site).iter().sum::<usize>() % 2;
                assert_eq!(lat.site_parity(site), expected, "site {site}");
            }
        }
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
        // Spot-check dir 2 at the origin; dir_stride = [1, 2, 6].
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
    fn link_and_plaquette_counts() {
        // 3D: 24 sites, 3 links each, C(3,2) = 3 plaquettes each.
        let lat = Lattice::new([2, 3, 4]);
        assert_eq!(lat.n_links(), 3 * 24);
        assert_eq!(lat.n_plaquettes(), 3 * 24);
        // 2D: C(2,2) = 1 plaquette per site, so half as many as links.
        let lat = Lattice::new([4, 4]);
        assert_eq!(lat.n_links(), 2 * 16);
        assert_eq!(lat.n_plaquettes(), 16);
        // 1D: links but no plaquettes, and no panic.
        let lat = Lattice::new([5]);
        assert_eq!(lat.n_links(), 5);
        assert_eq!(lat.n_plaquettes(), 0);
    }

    #[test]
    fn link_index_inverts_site_and_direction() {
        let lat = Lattice::new([2, 3, 4]);
        for link in 0..lat.n_links() {
            let site = lat.link_site(link);
            let dir = lat.link_direction(link);
            assert!(dir < 3);
            assert_eq!(lat.link_index(lat.site_coords(site), dir), link);
        }
        // Spot-check the packing: a site's links are contiguous, direction
        // fastest.
        assert_eq!(lat.link_index([0, 0, 0], 0), 0);
        assert_eq!(lat.link_index([0, 0, 0], 2), 2);
        assert_eq!(lat.link_index([1, 0, 0], 0), 3);
        // Link 7 is site 2, direction 1, and site 2 is (0, 1, 0) at strides
        // [1, 2, 6].
        assert_eq!(lat.link_site(7), 2);
        assert_eq!(lat.link_direction(7), 1);
        // The private packed form agrees with the public coordinate one.
        for site in 0..lat.n_sites() {
            for dir in 0..3 {
                assert_eq!(
                    lat.site_link(site, dir),
                    lat.link_index(lat.site_coords(site), dir)
                );
                assert_eq!(lat.link_base(lat.site_link(site, dir)), (site, dir));
            }
        }
    }

    #[test]
    fn hand_computed_plaquettes_on_3x3x3() {
        let lat = Lattice::new([3, 3, 3]);
        // Site strides [1, 3, 9]; pairs (0,1), (0,2), (1,2); links s * 3 + dir.
        // Plaquette 0 = site 0, pair (0,1): links (0,0), (1,1), (3,0), (0,1).
        assert_eq!(lat.plaquette_links(0), [0, 4, 9, 1]);
        // Plaquette 2 = site 0, pair (1,2): links (0,1), (3,2), (9,1), (0,2).
        assert_eq!(lat.plaquette_links(2), [1, 11, 28, 2]);
        // Site 4 = (1,1,0), pair (0,2): links (4,0), (5,2), (13,0), (4,2).
        assert_eq!(lat.plaquette_links(4 * 3 + 1), [12, 17, 39, 14]);
    }

    #[test]
    fn plaquette_links_are_distinct_and_close_a_loop() {
        let lat = Lattice::new([2, 3, 4]);
        for p in 0..lat.n_plaquettes() {
            let links = lat.plaquette_links(p);
            let unique: HashSet<usize> = links.iter().copied().collect();
            assert_eq!(unique.len(), 4, "plaquette {p} has a repeated link");

            // Each link runs from its base site to the forward neighbor along
            // its dir; a closed loop touches each of its four corners twice.
            let mut touches: HashMap<usize, usize> = HashMap::new();
            for &link in &links {
                let (site, dir) = lat.link_base(link);
                *touches.entry(site).or_default() += 1;
                *touches
                    .entry(lat.site_neighbor(site, dir, Sign::Plus))
                    .or_default() += 1;
            }
            assert_eq!(
                touches.len(),
                4,
                "plaquette {p} has {} corners",
                touches.len()
            );
            assert!(
                touches.values().all(|&n| n == 2),
                "plaquette {p} does not close: {touches:?}"
            );
        }
    }

    #[test]
    fn staple_groups_match_the_plaquettes() {
        let lat = Lattice::new([2, 3, 4]);
        let plaquettes: HashSet<[usize; 4]> = (0..lat.n_plaquettes())
            .map(|p| {
                let mut links = lat.plaquette_links(p);
                links.sort_unstable();
                links
            })
            .collect();

        let mut groups_seen = 0;
        for link in 0..lat.n_links() {
            let staples = lat.link_staples(link);
            // Stride 6 * (D - 1) = 12 in three dimensions, so four groups.
            assert_eq!(staples.len(), 12);
            for group in staples.chunks_exact(3) {
                let mut four = [link, group[0], group[1], group[2]];
                four.sort_unstable();
                assert!(
                    plaquettes.contains(&four),
                    "link {link} group {group:?} is not a plaquette"
                );
                groups_seen += 1;
            }
        }
        // Every plaquette contributes a group to each of its four links.
        assert_eq!(groups_seen, 4 * lat.n_plaquettes());
    }

    #[test]
    fn staple_groups_are_the_plaquettes_containing_the_link() {
        let lat = Lattice::new([3, 3, 3]);
        // Collect, for each link, the plaquettes that list it among their four.
        let mut containing: HashMap<usize, Vec<usize>> = HashMap::new();
        for p in 0..lat.n_plaquettes() {
            for link in lat.plaquette_links(p) {
                containing.entry(link).or_default().push(p);
            }
        }
        for link in 0..lat.n_links() {
            let plaquettes = &containing[&link];
            // 2 * (D - 1) = 4 in three dimensions.
            assert_eq!(plaquettes.len(), 4, "link {link} sits on {plaquettes:?}");
            // Groups come out in increasing plaquette index, each the matching
            // plaquette with `link` removed.
            for (group, &p) in lat.link_staples(link).chunks_exact(3).zip(plaquettes) {
                let expected: Vec<usize> = lat
                    .plaquette_links(p)
                    .into_iter()
                    .filter(|&other| other != link)
                    .collect();
                assert_eq!(group, expected, "link {link}, plaquette {p}");
            }
        }
    }

    #[test]
    fn two_dimensional_staple_table_has_one_group_per_link() {
        let lat = Lattice::new([4, 4]);
        // Stride 6 * (D - 1) = 6: two groups of three per link.
        for link in 0..lat.n_links() {
            assert_eq!(lat.link_staples(link).len(), 6);
        }
        // With one plane there is one plaquette per site, so plaquette index
        // and site index coincide. Link 0 is (site 0, dir 0): it is the (s, μ)
        // edge of the plaquette at site 0 and the (s + ν̂, μ) edge of the one a
        // step back along dir 1.
        let containing: HashSet<usize> = (0..lat.n_plaquettes())
            .filter(|&p| lat.plaquette_links(p).contains(&0))
            .collect();
        let back = lat.site_neighbor(0, 1, Sign::Minus);
        assert_eq!(containing, HashSet::from([0, back]));
    }

    #[test]
    fn neighbor_accessors_agree_with_table() {
        let lat = Lattice::new([3, 3]);
        // Center site (1,1) = index 4: +x0=5, -x0=3, +x1=7, -x1=1.
        assert_eq!(lat.site_neighbors(4), &[5, 3, 7, 1]);
        assert_eq!(lat.site_neighbor(4, 0, Sign::Plus), 5);
        assert_eq!(lat.site_neighbor(4, 0, Sign::Minus), 3);
        assert_eq!(lat.site_neighbor(4, 1, Sign::Plus), 7);
        assert_eq!(lat.site_neighbor(4, 1, Sign::Minus), 1);
    }

    // Gated because a `debug_assert!` is compiled out in release, and the test
    // would go red there for a reason that is not a defect. See the note on
    // check tiers in `model.rs`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "direction out of range")]
    fn out_of_range_direction_is_caught_not_folded_into_the_next_row() {
        // Without the check this lands in the next site's row and returns a
        // plausible site index, so nothing else would notice.
        let lat = Lattice::new([4, 4]);
        lat.site_neighbor(0, 2, Sign::Plus);
    }

    #[test]
    fn shift_wraps_periodically_along_a_direction() {
        let lat = Lattice::new([4, 4]);
        // Direction 0 (stride 1): step, full wrap, wrap+1, and the delta=0 identity.
        assert_eq!(lat.site_shift(0, 0, 1), 1);
        assert_eq!(lat.site_shift(0, 0, 4), 0); // full loop back to start
        assert_eq!(lat.site_shift(0, 0, 5), 1); // 5 mod 4 = 1
        assert_eq!(lat.site_shift(0, 0, 0), 0); // identity
        // Direction 1 (stride 4): steps move by whole rows.
        assert_eq!(lat.site_shift(0, 1, 1), 4);
        assert_eq!(lat.site_shift(0, 1, 3), 12);
        // delta = 1 must agree with the forward neighbor, for every site.
        for site in 0..lat.n_sites() {
            assert_eq!(
                lat.site_shift(site, 0, 1),
                lat.site_neighbor(site, 0, Sign::Plus)
            );
            assert_eq!(
                lat.site_shift(site, 1, 1),
                lat.site_neighbor(site, 1, Sign::Plus)
            );
        }
    }

    #[test]
    fn site_index_and_coords_are_inverse() {
        let lat = Lattice::new([2, 3, 4]);
        for site in 0..lat.n_sites() {
            assert_eq!(lat.site_index(lat.site_coords(site)), site);
        }
        // Spot-check a known coordinate: (1, 2, 3) with strides [1, 2, 6].
        assert_eq!(lat.site_index([1, 2, 3]), 1 + 2 * 2 + 3 * 6);
        assert_eq!(lat.site_coords(23), [1, 2, 3]);
    }

    #[test]
    fn plaquette_index_inverts_site_and_directions() {
        let lat = Lattice::new([2, 3, 4]);
        for p in 0..lat.n_plaquettes() {
            let [mu, nu] = lat.plaquette_directions(p);
            let coords = lat.site_coords(lat.plaquette_site(p));
            assert!(mu < nu && nu < 3);
            assert_eq!(lat.plaquette_index(coords, mu, nu), p);
        }
        // Pair ordinals are lexicographic: (0,1), (0,2), (1,2).
        assert_eq!(lat.plaquette_index([0, 0, 0], 0, 1), 0);
        assert_eq!(lat.plaquette_index([0, 0, 0], 0, 2), 1);
        assert_eq!(lat.plaquette_index([0, 0, 0], 1, 2), 2);
        assert_eq!(lat.plaquette_index([1, 0, 0], 0, 1), 3);
        // The private packed form agrees with the public coordinate one.
        for site in 0..lat.n_sites() {
            let p = lat.plaquette_at(site, 0, 2);
            assert_eq!(p, lat.plaquette_index(lat.site_coords(site), 0, 2));
            assert_eq!(lat.plaquette_base(p), (site, 0, 2));
        }
        // Four dimensions exercise the group-walking in both directions.
        let lat = Lattice::new([2, 2, 2, 2]);
        for p in 0..lat.n_plaquettes() {
            let [mu, nu] = lat.plaquette_directions(p);
            let coords = lat.site_coords(lat.plaquette_site(p));
            assert_eq!(lat.plaquette_index(coords, mu, nu), p);
        }
    }

    #[test]
    fn link_sites_are_the_endpoints() {
        let lat = Lattice::new([2, 3, 4]);
        for link in 0..lat.n_links() {
            let (site, dir) = lat.link_base(link);
            let [from, to] = lat.link_sites(link);
            assert_eq!(from, site);
            assert_eq!(to, lat.site_neighbor(site, dir, Sign::Plus));
        }
    }

    #[test]
    fn site_links_are_the_links_touching_the_site() {
        let lat = Lattice::new([2, 3, 4]);
        for site in 0..lat.n_sites() {
            let links = lat.site_links(site);
            assert_eq!(links.len(), 6);
            // Every one of them has this site as an endpoint, and the row is
            // ordered to match the neighbor row.
            for (i, link) in links.enumerate() {
                assert!(lat.link_sites(link).contains(&site), "site {site}");
                let expected = lat.site_neighbors(site)[i];
                assert!(lat.link_sites(link).contains(&expected));
            }
        }
    }

    #[test]
    fn a_unit_square_is_a_closed_loop() {
        let lat = Lattice::new([4, 4]);
        let square = [
            (0, Sign::Plus),
            (1, Sign::Plus),
            (0, Sign::Minus),
            (1, Sign::Minus),
        ];
        let path = Loop::new(&lat, &square).expect("a unit square closes");
        assert_eq!(path.steps(), &square);
        assert_eq!(path.perimeter(), 4);
    }

    #[test]
    fn an_open_path_is_rejected() {
        // Three sides of the square: it ends one step from where it began.
        let lat = Lattice::new([4, 4]);
        let open = [(0, Sign::Plus), (1, Sign::Plus), (0, Sign::Minus)];
        assert!(Loop::new(&lat, &open).is_none());
    }

    #[test]
    fn a_path_winding_the_torus_closes() {
        // L steps along one direction return to the base site the long way
        // round, which is closed on a torus and is the Polyakov loop's shape.
        let lat = Lattice::new([4, 6]);
        let wind: Vec<_> = (0..6).map(|_| (1, Sign::Plus)).collect();
        assert!(Loop::new(&lat, &wind).is_some());
        // One step short of a full wrap does not close.
        assert!(Loop::new(&lat, &wind[..5]).is_none());
        // Nor does a full wrap of the *other* direction's extent, which differs.
        let short: Vec<_> = (0..4).map(|_| (1, Sign::Plus)).collect();
        assert!(Loop::new(&lat, &short).is_none());
    }

    #[test]
    fn a_direction_out_of_range_is_rejected() {
        // Direction 2 does not exist in two dimensions, and the net
        // displacement in the directions that do exist would otherwise vanish.
        let lat = Lattice::new([4, 4]);
        let bad = [(2, Sign::Plus), (2, Sign::Minus)];
        assert!(Loop::new(&lat, &bad).is_none());
    }

    #[test]
    fn degenerate_paths_close() {
        // The displacement rule admits the empty path and an immediate retrace
        // rather than special-casing them; see the Loops section of the doc.
        let lat = Lattice::new([4, 4]);
        assert!(Loop::new(&lat, &[]).is_some());
        assert!(Loop::new(&lat, &[(0, Sign::Plus), (0, Sign::Minus)]).is_some());
    }

    #[test]
    fn a_rectangle_has_the_perimeter_it_should() {
        let lat = Lattice::new([8, 8, 8]);
        let rect = Loop::rectangle(&lat, 0, 3, 2, 1).unwrap();
        assert_eq!(rect.perimeter(), 2 * (3 + 1));
        // The smallest one is the unit square, step for step.
        assert_eq!(
            Loop::rectangle(&lat, 0, 1, 1, 1).unwrap().steps(),
            &[
                (0, Sign::Plus),
                (1, Sign::Plus),
                (0, Sign::Minus),
                (1, Sign::Minus),
            ]
        );
    }

    #[test]
    fn a_rectangle_needs_two_distinct_directions() {
        // `mu == nu` closes, so `new` would take it, but it measures nothing.
        let lat = Lattice::new([8, 8]);
        assert!(Loop::rectangle(&lat, 0, 2, 0, 3).is_none());
        assert!(Loop::rectangle(&lat, 0, 2, 5, 3).is_none());
    }

    #[test]
    fn a_rectangle_reaching_the_extent_is_refused() {
        // At the full extent the opposite sides land on the same links and
        // cancel, so the shape is no longer a rectangle.
        let lat = Lattice::new([4, 8]);
        assert!(Loop::rectangle(&lat, 0, 3, 1, 7).is_some());
        assert!(Loop::rectangle(&lat, 0, 4, 1, 1).is_none());
        assert!(Loop::rectangle(&lat, 0, 1, 1, 8).is_none());
    }

    #[test]
    fn walking_a_unit_square_reproduces_the_plaquette() {
        // The unit square must cross exactly `plaquette_links` in the L4 order;
        // any error in how a backward step picks its link shows up here.
        let lat = Lattice::new([2, 3, 4]);
        for (mu, nu) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let square = [
                (mu, Sign::Plus),
                (nu, Sign::Plus),
                (mu, Sign::Minus),
                (nu, Sign::Minus),
            ];
            let path = Loop::new(&lat, &square).expect("a unit square closes");
            for site in 0..lat.n_sites() {
                let walked: Vec<_> = lat.loop_links(site, &path).collect();
                let p = lat.plaquette_index(lat.site_coords(site), mu, nu);
                assert_eq!(walked, lat.plaquette_links(p), "site {site}, ({mu},{nu})");
            }
        }
    }

    #[test]
    fn walking_a_loop_returns_to_its_base_site() {
        // A closed walk leaves every site as often as it enters, so each
        // endpoint of the links crossed is touched an even number of times.
        let lat = Lattice::new([4, 4, 4]);
        let staircase = [
            (0, Sign::Plus),
            (1, Sign::Plus),
            (2, Sign::Plus),
            (0, Sign::Minus),
            (1, Sign::Minus),
            (2, Sign::Minus),
        ];
        let path = Loop::new(&lat, &staircase).expect("the staircase closes");

        for site in [0usize, 5, 21, 63] {
            let mut touches: HashMap<usize, usize> = HashMap::new();
            for link in lat.loop_links(site, &path) {
                for end in lat.link_sites(link) {
                    *touches.entry(end).or_default() += 1;
                }
            }
            assert!(
                touches.values().all(|&n| n % 2 == 0),
                "path from {site} does not close: {touches:?}"
            );
        }
    }

    #[test]
    fn a_winding_loop_crosses_a_whole_row_once() {
        // The Polyakov loop's shape: wrapping direction 1 crosses the `L_1`
        // links along that row, each exactly once, and comes back to the start.
        let lat = Lattice::new([4, 6]);
        let wind: Vec<_> = (0..6).map(|_| (1, Sign::Plus)).collect();
        let path = Loop::new(&lat, &wind).expect("a full wrap closes");

        let walked: Vec<_> = lat.loop_links(0, &path).collect();
        assert_eq!(walked.len(), 6);
        assert_eq!(walked.iter().collect::<HashSet<_>>().len(), 6);
        let expected: Vec<_> = (0..6).map(|r| lat.link_index([0, r], 1)).collect();
        assert_eq!(walked, expected);
    }

    #[test]
    fn a_backward_step_crosses_the_link_it_arrives_at() {
        // Links are named by their forward end, so stepping `−μ` out of a site
        // crosses the link based at the neighbor, not at the site left behind.
        let lat = Lattice::new([4, 4]);
        let there_and_back = [(0, Sign::Minus), (0, Sign::Plus)];
        let path = Loop::new(&lat, &there_and_back).expect("retracing closes");

        let back = lat.site_neighbor(5, 0, Sign::Minus);
        let walked: Vec<_> = lat.loop_links(5, &path).collect();
        assert_eq!(walked, vec![lat.site_link(back, 0), lat.site_link(back, 0)]);
    }

    #[test]
    fn link_plaquettes_match_the_staple_groups() {
        let lat = Lattice::new([2, 3, 4]);
        for link in 0..lat.n_links() {
            let plaquettes = lat.link_plaquettes(link);
            // 2 * (D - 1) = 4 in three dimensions, strictly increasing.
            assert_eq!(plaquettes.len(), 4);
            assert!(plaquettes.windows(2).all(|w| w[0] < w[1]));
            // Each listed plaquette really contains the link, and its staple
            // group is that plaquette minus the link.
            for (&p, group) in plaquettes
                .iter()
                .zip(lat.link_staples(link).chunks_exact(3))
            {
                let links = lat.plaquette_links(p);
                assert!(links.contains(&link), "link {link} not on plaquette {p}");
                let expected: Vec<usize> =
                    links.into_iter().filter(|&other| other != link).collect();
                assert_eq!(group, expected, "link {link}, plaquette {p}");
            }
        }
    }

    // --- Dimension sweep ---------------------------------------------------
    //
    // The same machinery as properties over a ladder of dimensions. Every
    // shape below is deliberately lopsided: a cubic one hides a transposed
    // stride, since every axis then carries the same place value.

    /// Counts follow from the shape and the dimension alone.
    fn counts_follow_the_shape<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        let n_sites: usize = shape.iter().product();
        assert_eq!(lat.n_sites(), n_sites, "{shape:?}");
        assert_eq!(lat.n_links(), D * n_sites, "{shape:?}");
        assert_eq!(
            lat.n_plaquettes(),
            D * D.saturating_sub(1) / 2 * n_sites,
            "{shape:?}"
        );
    }

    /// Every packing the lattice does is invertible: coordinates through a site
    /// index and back, and each cell kind's index through its parts and back.
    fn every_index_round_trips<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        for site in 0..lat.n_sites() {
            assert_eq!(lat.site_index(lat.site_coords(site)), site, "{shape:?}");
        }
        for link in 0..lat.n_links() {
            let (site, dir) = (lat.link_site(link), lat.link_direction(link));
            assert!(dir < D, "{shape:?}: direction {dir} out of range");
            assert_eq!(lat.site_link(site, dir), link, "{shape:?}");
        }
        for plaquette in 0..lat.n_plaquettes() {
            let site = lat.plaquette_site(plaquette);
            let [mu, nu] = lat.plaquette_directions(plaquette);
            assert!(mu < nu && nu < D, "{shape:?}: bad pair ({mu}, {nu})");
            assert_eq!(
                lat.plaquette_index(lat.site_coords(site), mu, nu),
                plaquette,
                "{shape:?}"
            );
        }
        // The direction-pair ordinal underneath the plaquette packing, which
        // nothing else exercises directly.
        for pair in 0..Lattice::<D>::n_dir_pairs() {
            let (mu, nu) = Lattice::<D>::dir_pair(pair);
            assert!(mu < nu, "{shape:?}");
            assert_eq!(Lattice::<D>::pair_index(mu, nu), pair, "{shape:?}");
        }
    }

    /// A step out and back along the same axis returns, in either order.
    fn neighbors_are_reciprocal<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        for site in 0..lat.n_sites() {
            for mu in 0..D {
                let up = lat.site_neighbor(site, mu, Sign::Plus);
                let down = lat.site_neighbor(site, mu, Sign::Minus);
                assert_eq!(lat.site_neighbor(up, mu, Sign::Minus), site, "{shape:?}");
                assert_eq!(lat.site_neighbor(down, mu, Sign::Plus), site, "{shape:?}");
            }
            // A displacement of zero is the identity, and one of `L` wraps back.
            for mu in 0..D {
                assert_eq!(lat.site_shift(site, mu, 0), site, "{shape:?}");
                assert_eq!(lat.site_shift(site, mu, shape[mu]), site, "{shape:?}");
            }
        }
    }

    /// A cell's incidence row lists distinct cells — which holds only at
    /// extents of two or more, all the shapes below use.
    fn incidence_rows_list_distinct_cells<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        for site in 0..lat.n_sites() {
            let distinct: HashSet<usize> = lat.site_links(site).collect();
            assert_eq!(
                distinct.len(),
                2 * D,
                "{shape:?}: site {site} repeats a link"
            );
        }
        for plaquette in 0..lat.n_plaquettes() {
            let links = lat.plaquette_links(plaquette);
            let distinct: HashSet<usize> = links.iter().copied().collect();
            assert_eq!(
                distinct.len(),
                4,
                "{shape:?}: plaquette {plaquette} repeats a link"
            );
        }
    }

    /// Boundary and coboundary describe the same incidence relation, read in
    /// opposite directions.
    fn incidence_agrees_in_both_directions<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);

        // Up from a site to its links, back down to the endpoints, one of
        // which must be the site we came from.
        for site in 0..lat.n_sites() {
            let links: Vec<usize> = lat.site_links(site).collect();
            assert_eq!(links.len(), 2 * D, "{shape:?}");
            for link in links {
                assert!(
                    lat.link_sites(link).contains(&site),
                    "{shape:?}: link {link} does not touch site {site}"
                );
            }
        }

        // Plaquettes against their links, both ways round.
        let mut from_plaquettes: HashSet<(usize, usize)> = HashSet::new();
        for plaquette in 0..lat.n_plaquettes() {
            for link in lat.plaquette_links(plaquette) {
                from_plaquettes.insert((link, plaquette));
            }
        }
        let mut from_links: HashSet<(usize, usize)> = HashSet::new();
        for link in 0..lat.n_links() {
            let plaquettes = lat.link_plaquettes(link);
            assert_eq!(
                plaquettes.len(),
                2 * D.saturating_sub(1),
                "{shape:?}: link {link}"
            );
            assert!(
                plaquettes.windows(2).all(|w| w[0] < w[1]),
                "{shape:?}: link_plaquettes must come back sorted"
            );
            for plaquette in plaquettes {
                from_links.insert((link, plaquette));
            }
        }
        assert_eq!(from_plaquettes, from_links, "{shape:?}");
    }

    /// Each staple group is some plaquette containing the link, minus the link,
    /// and the groups arrive in the order `link_plaquettes` reports.
    fn staples_are_plaquettes_minus_the_link<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        assert_eq!(Lattice::<D>::staple_stride(), 6 * D.saturating_sub(1));
        for link in 0..lat.n_links() {
            let plaquettes = lat.link_plaquettes(link);
            let groups = lat.link_staples(link).chunks_exact(3);
            assert_eq!(groups.len(), plaquettes.len(), "{shape:?}: link {link}");
            for (&plaquette, group) in plaquettes.iter().zip(groups) {
                let links = lat.plaquette_links(plaquette);
                assert!(
                    links.contains(&link),
                    "{shape:?}: link {link} not on {plaquette}"
                );
                let expected: Vec<usize> =
                    links.into_iter().filter(|&other| other != link).collect();
                assert_eq!(
                    group, expected,
                    "{shape:?}: link {link}, plaquette {plaquette}"
                );
            }
        }
    }

    /// A unit square walked as a path crosses exactly the plaquette's four
    /// links, in every plane the lattice has.
    fn unit_squares_close_in_every_plane<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        for plaquette in 0..lat.n_plaquettes() {
            let site = lat.plaquette_site(plaquette);
            let [mu, nu] = lat.plaquette_directions(plaquette);
            let path = Loop::new(
                &lat,
                &[
                    (mu, Sign::Plus),
                    (nu, Sign::Plus),
                    (mu, Sign::Minus),
                    (nu, Sign::Minus),
                ],
            )
            .expect("a unit square closes");
            let mut walked: Vec<usize> = lat.loop_links(site, &path).collect();
            walked.sort_unstable();
            let mut expected = lat.plaquette_links(plaquette).to_vec();
            expected.sort_unstable();
            assert_eq!(walked, expected, "{shape:?}: plaquette {plaquette}");
        }
    }

    /// Run every geometric property over one shape.
    fn geometry_sweep<const D: usize>(shape: [usize; D]) {
        counts_follow_the_shape(shape);
        every_index_round_trips(shape);
        neighbors_are_reciprocal(shape);
        incidence_rows_list_distinct_cells(shape);
        incidence_agrees_in_both_directions(shape);
        staples_are_plaquettes_minus_the_link(shape);
        unit_squares_close_in_every_plane(shape);
    }

    /// A shape too large to count is refused rather than wrapped: `[4096; 6]`
    /// is `2^72`, whose product wraps to *zero* in release builds, giving a
    /// silently empty lattice. Debug builds panic on the multiplication by
    /// themselves, which is why this asserts on the message.
    #[test]
    #[should_panic(expected = "more sites than a usize can count")]
    fn a_shape_too_large_to_count_is_refused() {
        Lattice::new([4096; 6]);
    }

    /// The same guard on the table lengths: `2^61` sites count fine, the
    /// neighbor table's `2^61 x 14` does not. The staple-table guard in
    /// `build_staples` cannot be reached in practice — the neighbor table
    /// always overflows or fails to allocate first — but stays because the two
    /// tables are sized in different functions and a reader of either should
    /// not have to check the other.
    #[test]
    #[should_panic(expected = "the neighbor table for shape")]
    fn a_table_too_large_to_address_is_refused() {
        Lattice::new([2usize.pow(55), 2, 2, 2, 2, 2, 2]);
    }

    /// The geometry holds as a set of properties at every dimension up to ten —
    /// a guess at the most anyone would plausibly run, not a bound the library
    /// states; `Lattice<12>` compiles.
    #[test]
    fn geometry_holds_in_every_dimension() {
        geometry_sweep([6]);
        geometry_sweep([4, 6]);
        geometry_sweep([2, 3, 4]);
        geometry_sweep([2, 3, 2, 3]);
        geometry_sweep([2, 2, 3, 2, 2]);
        geometry_sweep([2, 3, 2, 2, 3, 2]);
        geometry_sweep([2, 2, 2, 3, 2, 2, 2, 2]);
        geometry_sweep([2, 2, 2, 2, 3, 2, 2, 2, 2, 2]);
    }

    /// Neighboring sites carry opposite parity, which is the whole basis of the
    /// checkerboard colorings, and it holds under the periodic wrap only when
    /// every extent is even.
    #[test]
    fn parity_alternates_between_neighbors_in_every_dimension() {
        fn check<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            for site in 0..lat.n_sites() {
                for &neighbor in lat.site_neighbors(site) {
                    assert_ne!(
                        lat.site_parity(site),
                        lat.site_parity(neighbor),
                        "{shape:?}: sites {site} and {neighbor} share a color"
                    );
                }
            }
        }
        check([6]);
        check([4, 6]);
        check([2, 4, 6]);
        check([2, 4, 2, 4]);
        check([2, 2, 4, 2, 2]);
        check([2, 4, 2, 2, 4, 2]);
        check([2, 2, 2, 4, 2, 2, 2, 2]);
        check([2, 2, 2, 2, 4, 2, 2, 2, 2, 2]);
    }
}
