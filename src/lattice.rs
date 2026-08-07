//! Hypercubic-lattice geometry with periodic (toroidal) boundary conditions.
//!
//! Generic over the spatial dimension `D`. A `D`-dimensional lattice with
//! per-direction extents `shape = [L_0, ..., L_{D-1}]` indexes its sites
//! row-major (mixed-radix, lexicographic) with **direction 0 the
//! fastest-varying**:
//!
//! ```text
//! site = x_0 + L_0 * (x_1 + L_1 * (x_2 + ...))
//! ```
//!
//! Every site has `2 * D` nearest neighbors — one forward (`+`) and one
//! backward (`−`) along each direction. They are precomputed once at
//! construction so the sampler hot path never re-derives geometry or takes a
//! modulo per lookup. Unequal extents give an anisotropic lattice
//! (e.g. `N_t × N_s^d`).
//!
//! Gauge models put their variables on the edges rather than the sites, so the
//! lattice names three kinds of cell, each anchored at a base site and
//! extending in the positive directions: the site itself, the `D` forward links
//! leaving it, and the `C(D, 2)` unit squares (plaquettes) spanned by a pair of
//! directions there. Each kind gets a count `n_*`, an `*_index` that packs a
//! base position and however many directions that kind needs into a linear
//! index, and accessors giving back those parts one at a time
//! ([`link_site`](Lattice::link_site),
//! [`link_direction`](Lattice::link_direction), and the plaquette pair).
//!
//! Two words are reserved. `coords` means lattice coordinates and never a
//! packed site index; *direction* means μ and never the sense of a step, which
//! is what [`Sign`] carries. Nothing returns a site and a direction fused into
//! one value — the literature writes a link as `U_μ(n)` and never names that
//! pair, and an earlier revision that did fuse them made `coords` ambiguous.
//!
//! Incidence between the kinds follows one rule: `a_bs(a)` gives the `b`-cells
//! touching cell `a`, reading down the dimensions for the boundary
//! ([`link_sites`](Lattice::link_sites),
//! [`plaquette_links`](Lattice::plaquette_links)) and up for the cells that
//! contain it ([`site_links`](Lattice::site_links),
//! [`link_plaquettes`](Lattice::link_plaquettes)). All of these are cheap
//! enough for an inner loop, so which read a stored table and which recompute
//! is an implementation detail callers need not track; the sole exception is
//! `link_plaquettes`, which allocates and sorts to guarantee its order and so
//! belongs outside a hot path.
//!
//! The one exception to the naming rule is
//! [`link_staples`](Lattice::link_staples), which is not incidence but a
//! composition of it: the links a flip must multiply, precomputed because an
//! updater rederiving them per proposal would spend more time on geometry than
//! on the flip.

/// The sign of a step along a direction: `Plus` is `+μ`, `Minus` is `−μ`.
///
/// The literature reserves *direction* for μ itself, so the sign gets its own
/// name rather than competing for that word.
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
    /// Flattened staple table, stride `6 * (D - 1)` per link: for each of the
    /// `2 * (D - 1)` plaquettes containing a link, the three other links of
    /// that plaquette, in consecutive groups of three.
    staples: Vec<usize>,
    /// Coordinate-sum parity per site, `0` or `1` — the checkerboard color.
    ///
    /// A table rather than a computation because the checkerboard schedules ask
    /// for it twice per variable update, and deriving it needs
    /// [`site_coords`](Lattice::site_coords), whose divisions by runtime extents
    /// measured at roughly a third of a whole CPU sweep. One byte per site is
    /// nothing beside the neighbor table already here.
    parities: Vec<u8>,
    // TODO: `dir_strides` is recomputed on every `site_index`, `site_coords`,
    // and `site_shift` call; cache it here if any of them lands in a hot loop.
    // TODO: boundary conditions are hardcoded periodic; make swappable
    // (per-direction, at construction) — e.g. open, antiperiodic, twisted.
}

impl<const D: usize> Lattice<D> {
    /// Build a `D`-dimensional lattice with the given per-direction extents and
    /// precompute its neighbor table.
    ///
    /// # Panics
    ///
    /// Panics if `D == 0` or if any extent is zero.
    pub fn new(shape: [usize; D]) -> Self {
        assert!(D > 0, "lattice dimension must be positive");
        assert!(
            shape.iter().all(|&l| l > 0),
            "every extent must be positive"
        );

        let n_sites: usize = shape.iter().product();
        let stride = 2 * D;

        // Mixed-radix place values: dir_stride[μ] is the linear-index step of a
        // one-unit move along direction μ (dir_stride[0] == 1).
        let mut dir_stride = [1usize; D];
        for mu in 1..D {
            dir_stride[mu] = dir_stride[mu - 1] * shape[mu - 1];
        }

        let mut neighbors = vec![0usize; n_sites * stride];
        // Filled in the same pass, since decomposing the site index is most of
        // the cost of either table and this loop already does it.
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

        // The staple table is derived from the plaquette enumeration, which
        // reads the neighbor table through `&self`, so build the lattice first
        // and fill the table in afterwards.
        let mut lattice = Lattice {
            shape,
            neighbors,
            staples: Vec::new(),
            parities,
        };
        lattice.staples = lattice.build_staples();
        lattice
    }

    /// Invert the plaquette enumeration into the per-link staple table.
    ///
    /// Walking the plaquettes in index order and, for each of a plaquette's
    /// four links, recording the other three keeps the table consistent with
    /// [`plaquette_links`](Lattice::plaquette_links) by construction: a group
    /// is exactly some plaquette minus the link it hangs off. Groups land in
    /// increasing plaquette index because that is the order of the outer loop,
    /// which is what [`link_plaquettes`](Lattice::link_plaquettes) has to
    /// match.
    fn build_staples(&self) -> Vec<usize> {
        let stride = Self::staple_stride();
        let mut staples = vec![0usize; self.n_links() * stride];
        // How many groups each link has taken so far, i.e. where the next one
        // goes within its row.
        let mut filled = vec![0usize; self.n_links()];

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
                .all(|&groups| groups == 2 * D.saturating_sub(1)),
            "every link must belong to exactly 2 * (D - 1) plaquettes"
        );
        staples
    }

    /// The per-direction extents `[L_0, ..., L_{D-1}]`.
    pub fn shape(&self) -> [usize; D] {
        self.shape
    }
}

/// Counting and naming the cells. A cell of any kind is a base position plus an
/// orientation — none for a site, one direction for a link, an unordered pair
/// of directions for a plaquette — and each kind packs that into a linear index
/// the same way, orientation fastest, so a site's cells of a given kind are
/// contiguous.
///
/// Every `*_index` takes coordinates, and the parts come back separately rather
/// than as a tuple mixing a site with a direction. The packed forms that skip
/// the coordinate round trip (`site_link`, `link_base`, and their plaquette
/// counterparts) stay private until something outside proves it needs them. All
/// of this is arithmetic on `shape`; none of it reads a table.
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
    /// `1`.
    ///
    /// Two sites of the same color are never nearest neighbors on an even
    /// lattice, because a step along any axis changes one coordinate by one and
    /// so flips the parity. Read from a table built at construction, because the
    /// checkerboard schedules ask for it twice per variable update and computing
    /// it costs a division per axis.
    pub fn site_parity(&self, site: usize) -> usize {
        self.parities[site] as usize
    }

    /// The base site of `link`, i.e. the end it points away from.
    pub fn link_site(&self, link: usize) -> usize {
        self.link_base(link).0
    }

    /// The direction `link` runs along.
    ///
    /// There is no accessor returning the site and direction together: the
    /// literature never names that pair, writing the link as `U_μ(n)` and
    /// leaving the two parts separate, and fusing them here is what made
    /// `coords` ambiguous in an earlier revision.
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

    /// The link leaving `site` along `dir`, with the base site already packed.
    ///
    /// Packed `site * D + dir`, direction fastest, so a site's `D` links are
    /// contiguous just like its neighbor row. [`link_index`](Lattice::link_index)
    /// takes coordinates, which is the right interface for naming a link in a
    /// formula but the wrong one for a sweep that already holds a site index —
    /// going through coordinates and back costs a divide per axis each way to
    /// recover a number that is one multiply-add. So this is the form the
    /// incidence accessors, the table builder, and any schedule visiting one
    /// link per site per direction use.
    pub fn site_link(&self, site: usize, dir: usize) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(dir < D, "direction out of range");
        site * D + dir
    }

    /// Split a link index into its base site and direction; the shared helper
    /// behind [`link_site`](Lattice::link_site) and
    /// [`link_direction`](Lattice::link_direction).
    fn link_base(&self, link: usize) -> (usize, usize) {
        debug_assert!(link < self.n_links(), "link out of range");
        (link / D, link % D)
    }

    /// The plaquette at `site` spanning `(mu, nu)`, with the base site already
    /// packed.
    ///
    /// Packed `site * C(D, 2) + pair`, pair fastest, with `pair` the ordinal of
    /// `(μ, ν)` in lexicographic order — `(0,1), (0,2), (1,2)` in three
    /// dimensions.
    fn plaquette_at(&self, site: usize, mu: usize, nu: usize) -> usize {
        debug_assert!(site < self.n_sites(), "site out of range");
        debug_assert!(mu < nu && nu < D, "direction pair must satisfy mu < nu < D");
        site * Self::n_dir_pairs() + Self::pair_index(mu, nu)
    }

    /// Split a plaquette index into its base site and direction pair in one
    /// step.
    fn plaquette_base(&self, plaquette: usize) -> (usize, usize, usize) {
        debug_assert!(plaquette < self.n_plaquettes(), "plaquette out of range");
        let n_pairs = Self::n_dir_pairs();
        let (mu, nu) = Self::dir_pair(plaquette % n_pairs);
        (plaquette / n_pairs, mu, nu)
    }

    /// Number of unordered direction pairs `(μ, ν)` with `μ < ν`, i.e. the
    /// number of lattice planes, and hence of plaquettes anchored at each site.
    const fn n_dir_pairs() -> usize {
        D * D.saturating_sub(1) / 2
    }

    /// The direction pair `(μ, ν)`, `μ < ν`, at ordinal `pair` in lexicographic
    /// order.
    ///
    /// Pairs are grouped by their first direction: `μ = 0` accounts for the
    /// first `D - 1` ordinals, `μ = 1` the next `D - 2`, and so on, so walking
    /// those group sizes locates the pair without a table.
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

    /// The ordinal of `(mu, nu)`, `mu < nu`, in the same lexicographic order;
    /// the inverse of [`dir_pair`](Lattice::dir_pair).
    ///
    /// The pairs with a first direction below `mu` number
    /// `(D - 1) + ... + (D - mu) = mu * (D - 1) - mu * (mu - 1) / 2`, and
    /// `nu - mu - 1` more come before `(mu, nu)` within its own group.
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
    /// periodically.
    ///
    /// The general-`delta` companion to
    /// [`site_neighbor`](Lattice::site_neighbor): `site_shift(site, μ, 1)` is
    /// the forward neighbor and `site_shift(site, μ, 0)` is `site` itself.
    /// `delta` may exceed the extent along `dir`; it is reduced modulo it.
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

/// Incidence between the cell kinds. `a_bs(a)` is the set of `b`-cells touching
/// the `a`-cell `a`: going down a dimension gives its boundary, going up gives
/// the cells it bounds. A fixed-size answer comes back as an array, since `D`
/// is a const parameter and array lengths cannot depend on it. The two whose
/// size grows with `D` differ: [`site_links`](Lattice::site_links) yields its
/// links one at a time, because a gauge move walks them once per site per sweep
/// and an allocation that often would outweigh the work, while
/// [`link_plaquettes`](Lattice::link_plaquettes) collects, because its ordering
/// guarantee needs a sort and nothing on the energy path calls it.
impl<const D: usize> Lattice<D> {
    /// The `2 * D` links touching `site`: for each direction the forward link
    /// leaving it and the backward link arriving at it, ordered `+0, −0, +1,
    /// ...` to match [`site_neighbors`](Lattice::site_neighbors). On an
    /// extent-1 direction the two coincide and the link is yielded twice.
    ///
    /// The caller that asks most often is a gauge transformation — flip every
    /// link touching a site — which runs once per site per sweep and only walks
    /// the result, so the links are computed one at a time rather than
    /// collected. Being a map over `0..2 * D`, the iterator still knows its
    /// length.
    pub fn site_links(&self, site: usize) -> impl ExactSizeIterator<Item = usize> + '_ {
        debug_assert!(site < self.n_sites(), "site out of range");
        // Column `i` of the neighbor row: direction `i / 2`, forward when `i`
        // is even. The forward link is based here, the backward one at the
        // neighbor it arrives from.
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
    /// [`link_staples`](Lattice::link_staples), so the two can be zipped.
    ///
    /// For a link `(s, μ)` and each other direction `ν`, one plaquette is
    /// anchored at `s` itself and one a step back along `ν`. They do not arrive
    /// in index order, so the answer is collected and sorted; that ordering
    /// guarantee is the whole reason this one allocates where
    /// [`site_links`](Lattice::site_links) does not.
    ///
    /// On an extent-1 direction the two coincide and the same plaquette appears
    /// twice, which still lines up with the staple groups, since those are
    /// counted by slot rather than by distinct plaquette.
    pub fn link_plaquettes(&self, link: usize) -> Vec<usize> {
        let (site, dir) = self.link_base(link);
        let mut plaquettes = Vec::with_capacity(2 * D.saturating_sub(1));
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

    /// The four links bounding `plaquette`, in the fixed order `[(s, μ), (s +
    /// μ̂, ν), (s + ν̂, μ), (s, ν)]`.
    ///
    /// Derived on demand from the neighbor table rather than stored: it is read
    /// once per plaquette when the staple table is built and thereafter only by
    /// whole-lattice sweeps, never per proposed flip.
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
    /// increasing plaquette index.
    ///
    /// This is not incidence but a composition of it — link to plaquettes to
    /// their links, minus the link itself. The product of a plaquette's four
    /// link variables equals the variable on `link` times the product over the
    /// matching staple, which is what lets a single-link flip read its energy
    /// change from this table alone.
    pub fn link_staples(&self, link: usize) -> &[usize] {
        debug_assert!(link < self.n_links(), "link out of range");
        let stride = Self::staple_stride();
        &self.staples[link * stride..link * stride + stride]
    }

    /// Entries per link in the staple table: `2 * (D - 1)` groups of three.
    pub(crate) const fn staple_stride() -> usize {
        6 * D.saturating_sub(1)
    }

    /// The links `path` crosses when walked from `base`, in traversal order.
    ///
    /// A forward step crosses the link based at the site it leaves; a backward
    /// step crosses the link based at the site it *arrives* at, since links are
    /// named by their forward end. Which way a link was crossed is not reported:
    /// a `Z2` variable is its own inverse, so the sense of the step cannot
    /// change the product it feeds. The [`Loop`] keeps that information, so a
    /// model whose variables do not commute with their inverses could recover it
    /// without changing this.
    ///
    /// The links come one at a time rather than collected, because the caller
    /// that asks most often is a Wilson loop sweeping one shape across every
    /// site of the lattice, and a vector per base site would cost more than the
    /// walk.
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

/// A closed path on the lattice, named by the steps that trace it rather than by
/// the cells it crosses.
///
/// Each step is a direction and a [`Sign`], so consecutive steps join up by
/// construction and the path can never have a gap in it. What does have to be
/// checked is that it comes back to where it started: only a closed path has a
/// gauge-invariant link product, and an open one measures nothing. Making that a
/// type rather than a bare slice of steps is what moves the check to one place —
/// it runs once when the shape is built, not once at each site the shape is
/// later walked from.
///
/// The base site is deliberately not part of the shape. One rectangle walked
/// from every site in turn is a single translation class, and averaging over
/// that class is what makes a Wilson loop measurable at all, so the shape has to
/// be the thing that survives translation.
///
/// A path counts as closed when its net displacement vanishes *modulo the
/// extents*, not only when it literally retraces to its base site. On a torus a
/// path that winds a direction all the way round has also closed, and its
/// product is invariant like any other; ruling those out would rule out the
/// Polyakov loop with them.
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
    /// The lattice is borrowed only to be measured, since its extents are what
    /// decide when a winding path has closed, and is not kept afterwards. A loop
    /// is therefore valid for the extents it was built against: walking it on a
    /// lattice of different shape is meaningful only if it closes there too,
    /// which for a non-winding path it does.
    pub fn new(lattice: &Lattice<D>, steps: &[(usize, Sign)]) -> Option<Self> {
        let shape = lattice.shape();

        // Net displacement, one component per direction: `+μ` adds one to
        // component μ and `−μ` subtracts one.
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
    /// out of range, or a side is long enough to wrap.
    ///
    /// This is the one loop family the confinement measurement needs, and it is
    /// a constructor rather than something a caller assembles because the walk
    /// only means what it should when the four sides are in this order. The two
    /// directions have to differ: `mu == nu` traces a line out and back, which
    /// closes and would be accepted by [`new`](Loop::new), but crosses each of
    /// its links twice and so measures nothing.
    ///
    /// A side reaching the extent is refused rather than wrapped. Winding is
    /// what a Polyakov loop is for and is welcome in [`new`](Loop::new), but a
    /// rectangle that wraps has its two opposite sides land on the same links,
    /// which cancel; the shape stops being a rectangle while still looking like
    /// one.
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

    /// The parity table agrees with the definition it caches. Built alongside
    /// the neighbor table for speed, it would otherwise be free to drift from
    /// the coordinate sum it is supposed to be — including on odd extents, where
    /// the wrap puts same-parity sites next to each other and the table must
    /// still report the honest value.
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

    #[test]
    #[should_panic(expected = "direction out of range")]
    fn out_of_range_direction_is_caught_not_folded_into_the_next_row() {
        // Without the check this returns site 1's forward neighbor: the index
        // 0 * 2D + 2 * 2 lands in the next site's row rather than out of
        // bounds, so nothing else would notice.
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
        // Both are closed by the displacement rule and neither is useful: an
        // empty path crosses nothing, and a step retraced immediately crosses
        // the same link twice, whose product is 1 whatever the field is. The
        // rule admits them rather than growing a special case, since the
        // measurement they give is correct if uninformative.
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
        // `mu == nu` traces a line out and back. It closes, so `new` would take
        // it, but it crosses each link twice and measures nothing.
        let lat = Lattice::new([8, 8]);
        assert!(Loop::rectangle(&lat, 0, 2, 0, 3).is_none());
        assert!(Loop::rectangle(&lat, 0, 2, 5, 3).is_none());
    }

    #[test]
    fn a_rectangle_reaching_the_extent_is_refused() {
        // At the full extent the two opposite sides land on the same links and
        // cancel, so what is left is no longer a rectangle. Winding is fine in
        // general — `new` allows it — but not here.
        let lat = Lattice::new([4, 8]);
        assert!(Loop::rectangle(&lat, 0, 3, 1, 7).is_some());
        assert!(Loop::rectangle(&lat, 0, 4, 1, 1).is_none());
        assert!(Loop::rectangle(&lat, 0, 1, 1, 8).is_none());
    }

    #[test]
    fn walking_a_unit_square_reproduces_the_plaquette() {
        // The sharpest check available on the walk: the smallest closed path in
        // the (μ, ν) plane must cross exactly the links the lattice already
        // names as that plaquette's, in the same order — `plaquette_links`
        // returns `[(s, μ), (s + μ̂, ν), (s + ν̂, μ), (s, ν)]`, which is this
        // walk. Any error in how a backward step picks its link shows up here.
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
        // Every site the path visits is left as often as it is entered, so each
        // endpoint of the links crossed is touched an even number of times.
        // That is the same condition the plaquette test checks, and for `Z2` it
        // is exactly what makes the link product gauge invariant.
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
}
