//! Connected components of the lattice under a caller-supplied bond test.
//!
//! [`site_clusters`] walks every nearest-neighbor bond once, asks the caller
//! whether that bond is open, and returns the partition of the sites the open
//! bonds induce. There is no physics in it at all: no [`Action`](crate::Action),
//! no [`Configuration`](crate::Configuration), no `β`, no notion of a bond
//! *probability*. Whatever randomness a caller wants lives in its own closure,
//! which is what lets a stochastic updater and a deterministic percolation
//! measurement share this one walk.
//!
//! It sits at the root rather than inside [`updater`](crate::updater) for that
//! reason — the eventual consumers differ in their predicate and in nothing
//! else. [`ClusterUpdate`](crate::updater::ClusterUpdate) is the first;
//! percolation observables (mean cluster size, wrapping probability) would pass
//! another closure to the same function.
//!
//! [`grow_cluster`] is the single-cluster counterpart: it starts from a seed
//! site and walks outward only over bonds the predicate opens, touching work
//! proportional to the cluster it returns rather than to the lattice. It exists
//! for the Wolff updater, whose whole advantage on the CPU is that a move costs
//! the cluster and not the volume, and it carries the same no-physics contract
//! as [`site_clusters`].

use crate::lattice::Lattice;

/// Partition the sites of `lattice` into connected clusters under a
/// caller-supplied bond test, called exactly once per nearest-neighbor bond.
///
/// `joined` receives the bond's base site first and its forward neighbor along
/// one axis second, and returns whether the two are to be treated as connected.
/// The order is part of the contract because a caller may want it — a
/// non-symmetric predicate is meaningful even though the resulting relation is
/// symmetrized by the union.
///
/// A periodic lattice has `D * n_sites` bonds, one forward edge per site per
/// axis, and each is offered exactly once. That is the same walk
/// [`Potts`](crate::models::potts::Potts)'s energy takes, and getting it wrong
/// is a silent error rather than a loud one: offering each bond twice
/// would turn a bond probability `p` into `1 − (1−p)²` and quietly sample the
/// wrong distribution.
///
/// `joined` is [`FnMut`] so it can capture a generator, or a counter, by mutable
/// reference.
pub fn site_clusters<const D: usize>(
    lattice: &Lattice<D>,
    mut joined: impl FnMut(usize, usize) -> bool,
) -> SiteClusters {
    let n_sites = lattice.n_sites();
    let mut forest = Forest::new(n_sites);

    for site in 0..n_sites {
        // The neighbor row is ordered +0, −0, +1, −1, ...; taking the forward
        // columns only (every other entry) visits each bond once.
        for &partner in lattice.site_neighbors(site).iter().step_by(2) {
            if joined(site, partner) {
                forest.union(site, partner);
            }
        }
    }

    forest.into_clusters()
}

/// Grow one cluster outward from `seed` under a caller-supplied bond test,
/// returning its member sites in the order they were reached (`seed` first).
///
/// `joined` receives the site already inside the cluster first and the outside
/// candidate second, and returns whether the bond between them is open. It is
/// asked exactly once per bond that runs from a member to a site not yet in the
/// cluster at the moment the member is processed: a candidate that refuses one
/// bond may be offered again over a *different* bond from another member, but
/// never twice over the same one, and a bond whose far end has already joined
/// is not offered at all. That per-bond accounting is what a Wolff updater's
/// detailed balance rests on, so it is part of the contract rather than an
/// implementation detail — as is the walk order (members processed in the
/// order they joined, each offering its neighbor row in `+0, −0, +1, −1, …`
/// order), because a stochastic predicate's draws land in the stream in the
/// order the bonds are offered.
///
/// The two degeneracies of the periodic wrap are handled the way
/// [`site_clusters`] handles them: on an extent of one a site is its own
/// neighbor and the self-bond is never offered, and on an extent of two the
/// forward and backward neighbors coincide, so the same pair is offered twice —
/// two distinct bonds, either of which may open.
///
/// Work is proportional to the cluster's boundary-and-interior bond count, not
/// to the lattice, which is the reason this exists beside [`site_clusters`].
pub fn grow_cluster<const D: usize>(
    lattice: &Lattice<D>,
    seed: usize,
    mut joined: impl FnMut(usize, usize) -> bool,
) -> Vec<usize> {
    let mut member = vec![false; lattice.n_sites()];
    member[seed] = true;
    let mut cluster = vec![seed];

    // The frontier indexes into `cluster` rather than copying sites: everything
    // ever pushed is a member, and members are never removed.
    let mut next = 0;
    while next < cluster.len() {
        let site = cluster[next];
        next += 1;
        for &partner in lattice.site_neighbors(site) {
            if !member[partner] && joined(site, partner) {
                member[partner] = true;
                cluster.push(partner);
            }
        }
    }
    cluster
}

/// A partition of the sites into clusters, with labels compacted to
/// `0..n_clusters`.
///
/// The compaction is what makes [`labels`](SiteClusters::labels) index
/// [`sizes`](SiteClusters::sizes) directly, so a consumer that draws one value
/// per cluster can hold a flat array of them rather than a map keyed by whatever
/// site happened to become a root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiteClusters {
    /// One cluster label per site, in lattice index order, each below
    /// `sizes.len()`.
    labels: Vec<usize>,
    /// Site count per cluster, indexed by label. Sums to the site count.
    sizes: Vec<usize>,
}

impl SiteClusters {
    /// How many clusters the sites fall into — at least one on a non-empty
    /// lattice, and exactly `n_sites` when no bond is open.
    pub fn n_clusters(&self) -> usize {
        self.sizes.len()
    }

    /// One cluster label per site, in lattice index order.
    pub fn labels(&self) -> &[usize] {
        &self.labels
    }

    /// Site count per cluster, indexed by label.
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }
}

/// Disjoint sets over the sites: union by size, with path halving on lookup.
///
/// Kept private, and deliberately not a general-purpose structure. Its only job
/// is to survive one labeling pass and be consumed into a [`SiteClusters`].
struct Forest {
    /// Each site's parent; a root is its own parent.
    parent: Vec<usize>,
    /// Sites in the tree rooted at this entry, meaningful at roots only. Union
    /// by size keeps the trees shallow, which together with path halving is what
    /// holds the whole walk to effectively linear time.
    size: Vec<usize>,
}

impl Forest {
    fn new(n_sites: usize) -> Self {
        Forest {
            parent: (0..n_sites).collect(),
            size: vec![1; n_sites],
        }
    }

    /// The root of `site`'s tree, flattening the path walked on the way up.
    ///
    /// Path *halving* rather than full compression: it points each node at its
    /// grandparent in the same loop that walks up, so there is no second pass
    /// and no recursion — which matters because the recursive form would be
    /// bounded by the tree depth on a lattice that can hold millions of sites.
    fn find(&mut self, site: usize) -> usize {
        let mut current = site;
        while self.parent[current] != current {
            self.parent[current] = self.parent[self.parent[current]];
            current = self.parent[current];
        }
        current
    }

    /// Merge the trees holding `a` and `b`, hanging the smaller off the larger.
    fn union(&mut self, a: usize, b: usize) {
        let (mut big, mut small) = (self.find(a), self.find(b));
        if big == small {
            return;
        }
        if self.size[big] < self.size[small] {
            std::mem::swap(&mut big, &mut small);
        }
        self.parent[small] = big;
        self.size[big] += self.size[small];
    }

    /// Consume the forest into compacted labels, in a single pass over the
    /// sites: the first site of a cluster names it, and every later member finds
    /// the same root and so the same label.
    fn into_clusters(mut self) -> SiteClusters {
        /// No label has been handed out for this root yet. A sentinel rather
        /// than an `Option` because the table is one entry per site and this
        /// keeps it a plain `Vec<usize>`; `usize::MAX` is unreachable as a label
        /// since labels are bounded by the site count.
        const UNLABELED: usize = usize::MAX;

        let n_sites = self.parent.len();
        let mut label_of_root = vec![UNLABELED; n_sites];
        let mut labels = vec![0usize; n_sites];
        let mut sizes: Vec<usize> = Vec::new();

        for (site, slot) in labels.iter_mut().enumerate() {
            let root = self.find(site);
            let label = if label_of_root[root] == UNLABELED {
                let fresh = sizes.len();
                label_of_root[root] = fresh;
                sizes.push(0);
                fresh
            } else {
                label_of_root[root]
            };
            *slot = label;
            sizes[label] += 1;
        }

        SiteClusters { labels, sizes }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Sign;

    /// Every bond open leaves one cluster holding the whole lattice.
    #[test]
    fn joining_every_bond_gives_a_single_cluster() {
        let lat = Lattice::new([4, 6]);
        let clusters = site_clusters(&lat, |_, _| true);

        assert_eq!(clusters.n_clusters(), 1);
        assert_eq!(clusters.sizes(), &[lat.n_sites()]);
        assert!(clusters.labels().iter().all(|&l| l == 0));
    }

    /// No bond open leaves every site alone, which is the `beta = 0` limit of
    /// the cluster update and the case where it degenerates to independent
    /// resampling.
    #[test]
    fn joining_no_bond_gives_one_cluster_per_site() {
        let lat = Lattice::new([4, 6]);
        let clusters = site_clusters(&lat, |_, _| false);

        assert_eq!(clusters.n_clusters(), lat.n_sites());
        assert!(clusters.sizes().iter().all(|&s| s == 1));
        // Compaction with nothing merged is the identity on the site order.
        assert_eq!(clusters.labels(), (0..lat.n_sites()).collect::<Vec<_>>());
    }

    /// The partition invariants hold whatever the predicate does: the labels are
    /// in range, and the sizes account for every site exactly once.
    ///
    /// Run over lopsided shapes and up to four dimensions, since a walk that
    /// assumed four neighbors would pass on a square lattice and nowhere else.
    #[test]
    fn the_labels_and_sizes_partition_the_sites_in_every_dimension() {
        fn probe<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            // A deterministic, shape-independent scatter of open bonds — enough
            // structure to produce several clusters of different sizes without
            // needing a generator.
            let clusters = site_clusters(&lat, |i, j| (i + 3 * j).is_multiple_of(5));

            let n_clusters = clusters.n_clusters();
            assert!(n_clusters >= 1, "{shape:?}");
            assert_eq!(clusters.labels().len(), lat.n_sites(), "{shape:?}");
            assert_eq!(clusters.sizes().len(), n_clusters, "{shape:?}");
            assert!(
                clusters.labels().iter().all(|&l| l < n_clusters),
                "{shape:?}: a label sits outside 0..n_clusters"
            );
            assert_eq!(
                clusters.sizes().iter().sum::<usize>(),
                lat.n_sites(),
                "{shape:?}"
            );

            // `sizes` really is the tally of `labels`, not a count kept beside
            // it that could drift.
            let mut tally = vec![0usize; n_clusters];
            for &label in clusters.labels() {
                tally[label] += 1;
            }
            assert_eq!(tally, clusters.sizes(), "{shape:?}");
        }

        probe([6]);
        probe([4, 6]);
        probe([3, 4, 5]);
        probe([2, 3, 2, 3]);
    }

    /// The partition is exactly the connected components of the open bonds,
    /// checked against a flood fill that shares no code with the union-find.
    ///
    /// This is the module's actual contract, and the tests above only circle it:
    /// they say the labels are in range and the sizes add up, which a union-find
    /// that merged too much or too little would also manage. What is asserted
    /// here is that two sites carry the same label *if and only if* a path of
    /// open bonds joins them — the "only if" being the direction a missing path
    /// compression or a mis-taken root would break.
    ///
    /// The two labelings are compared outright rather than up to a permutation,
    /// because both number a cluster by the smallest site index in it: the
    /// compaction hands out a fresh label the first time it meets a root walking
    /// sites in order, and the flood fill does the same walking starts in order.
    ///
    /// The shapes include an extent of one, where a site's forward neighbor is
    /// itself and the bond is a self-loop, and an extent of two, where the
    /// forward and backward neighbors coincide and the pair is joined by two
    /// distinct bonds. Both are degeneracies of the periodic wrap rather than
    /// errors, and both have to come out of the union-find and the flood fill
    /// the same way.
    #[test]
    fn the_partition_is_the_connected_components_of_the_open_bonds() {
        fn probe<const D: usize>(shape: [usize; D], open_fraction: f64, seed: u64) {
            let lat = Lattice::new(shape);
            let n_sites = lat.n_sites();

            // Record what the predicate answered, so the reference is built from
            // the same open bonds without asking the same code for them.
            let mut rng = crate::rng::RandRng::seed_from_u64(seed);
            let mut open: Vec<(usize, usize)> = Vec::new();
            let clusters = site_clusters(&lat, |i, j| {
                let is_open = crate::rng::Rng::next_f64(&mut rng) < open_fraction;
                if is_open {
                    open.push((i, j));
                }
                is_open
            });

            let mut adjacency = vec![Vec::new(); n_sites];
            for &(i, j) in &open {
                adjacency[i].push(j);
                adjacency[j].push(i);
            }

            const UNVISITED: usize = usize::MAX;
            let mut reference = vec![UNVISITED; n_sites];
            let mut n_components = 0;
            for start in 0..n_sites {
                if reference[start] != UNVISITED {
                    continue;
                }
                reference[start] = n_components;
                let mut frontier = vec![start];
                while let Some(site) = frontier.pop() {
                    for &next in &adjacency[site] {
                        if reference[next] == UNVISITED {
                            reference[next] = n_components;
                            frontier.push(next);
                        }
                    }
                }
                n_components += 1;
            }

            assert_eq!(clusters.n_clusters(), n_components, "{shape:?}");
            assert_eq!(clusters.labels(), reference, "{shape:?}");
            // The fixture has to be a real partition rather than a degenerate
            // one, or the comparison above would hold vacuously.
            assert!(
                n_components > 1 && n_components < n_sites,
                "{shape:?}: {n_components} components over {n_sites} sites is \
                 too degenerate to test anything"
            );
        }

        // The open fraction is chosen per shape rather than held fixed, because
        // what makes a partition interesting depends on the dimension: a site
        // has `2D` neighbors, so the density at which the open bonds start
        // spanning the lattice falls as `D` rises. Roughly a third of the way
        // below each shape's percolation threshold gives clusters large enough
        // to have interior sites — which is what exercises a multi-step path to
        // the root — without one of them swallowing the box.
        probe([6], 0.5, 1);
        probe([4, 6], 0.4, 2);
        probe([3, 4, 5], 0.2, 3);
        probe([2, 3, 2, 3], 0.12, 4);
        // An extent of one: the forward neighbor along that axis is the site
        // itself, so the bond is a self-loop and must merge nothing.
        probe([1, 8], 0.5, 5);
        probe([4, 1, 3], 0.3, 6);
        // An extent of two: the forward and backward neighbors coincide, so one
        // pair of sites carries two distinct bonds and either may open.
        probe([2, 5], 0.3, 7);
        probe([2, 2, 3], 0.2, 8);
    }

    /// A hand-checked pattern: opening only the bonds along axis 0 on a `[4, 4]`
    /// lattice leaves four clusters of four, because the periodic wrap closes
    /// each row into a ring and nothing joins one row to the next.
    #[test]
    fn joining_one_axis_leaves_one_cluster_per_row() {
        let lat = Lattice::new([4, 4]);
        let clusters = site_clusters(&lat, |i, j| j == lat.site_neighbor(i, 0, Sign::Plus));

        assert_eq!(clusters.n_clusters(), 4);
        assert_eq!(clusters.sizes(), &[4, 4, 4, 4]);
        // Axis 0 is the fastest-varying index, so a row is four consecutive
        // sites and the labels come out in blocks.
        assert_eq!(
            clusters.labels(),
            &[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        );
    }

    /// Each bond is offered exactly once, `D * n_sites` in all.
    ///
    /// The sharpest test here, because offering a bond twice breaks nothing
    /// visible: the partition would still be a partition, and a caller opening
    /// bonds with probability `p` would silently get `1 − (1−p)²` instead.
    #[test]
    fn every_bond_is_offered_exactly_once() {
        fn probe<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            let mut offered: Vec<(usize, usize)> = Vec::new();
            site_clusters(&lat, |i, j| {
                offered.push((i, j));
                false
            });

            assert_eq!(offered.len(), D * lat.n_sites(), "{shape:?}");
            // The forward link a bond is named by; `n_links` counts exactly
            // these, so a duplicate offer would show up as a repeat here.
            let mut links: Vec<usize> = offered
                .iter()
                .map(|&(i, j)| {
                    let dir = (0..D)
                        .find(|&mu| lat.site_neighbor(i, mu, Sign::Plus) == j)
                        .expect("a bond runs forward along some axis");
                    lat.site_link(i, dir)
                })
                .collect();
            links.sort_unstable();
            links.dedup();
            assert_eq!(links.len(), lat.n_links(), "{shape:?}");
        }

        probe([6]);
        probe([4, 6]);
        probe([3, 4, 5]);
        probe([2, 3, 2, 3]);
    }

    /// The predicate may hold mutable state — which is the whole reason it is
    /// `FnMut`, since the updater's closure captures a generator.
    #[test]
    fn the_predicate_may_capture_mutable_state() {
        let lat = Lattice::new([4, 4]);
        let mut calls = 0usize;
        let clusters = site_clusters(&lat, |_, _| {
            calls += 1;
            calls.is_multiple_of(2)
        });

        assert_eq!(calls, 2 * lat.n_sites());
        assert!(clusters.n_clusters() < lat.n_sites());
    }

    /// A growth that opens every bond reaches the whole lattice, and one that
    /// opens none stays on the seed — the two ends the Wolff updater hits at
    /// large and zero `beta`.
    #[test]
    fn a_grown_cluster_spans_the_two_limits() {
        let lat = Lattice::new([4, 6]);

        let all = grow_cluster(&lat, 5, |_, _| true);
        assert_eq!(all.len(), lat.n_sites());
        assert_eq!(all[0], 5, "the seed is the first member");
        let mut sorted = all.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..lat.n_sites()).collect::<Vec<_>>());

        assert_eq!(grow_cluster(&lat, 5, |_, _| false), vec![5]);
    }

    /// Under a deterministic symmetric predicate the grown cluster is exactly
    /// the [`site_clusters`] component holding the seed — the two walks name
    /// the same partition cell, reached from opposite ends.
    #[test]
    fn a_grown_cluster_is_the_seeds_connected_component() {
        fn probe<const D: usize>(shape: [usize; D]) {
            let lat = Lattice::new(shape);
            // Symmetric in (i, j), so growth (offered from the inside out) and
            // the forward-bond walk of `site_clusters` agree on which bonds
            // are open.
            let open = |i: usize, j: usize| (i + j).is_multiple_of(3);
            let clusters = site_clusters(&lat, open);

            for seed in 0..lat.n_sites() {
                let mut grown = grow_cluster(&lat, seed, open);
                grown.sort_unstable();
                let label = clusters.labels()[seed];
                let component: Vec<usize> = (0..lat.n_sites())
                    .filter(|&s| clusters.labels()[s] == label)
                    .collect();
                assert_eq!(grown, component, "{shape:?}, seed {seed}");
            }
        }

        probe([6]);
        probe([4, 6]);
        probe([3, 4, 5]);
        // The wrap degeneracies: a self-loop on an extent of one, a doubled
        // bond on an extent of two.
        probe([1, 8]);
        probe([2, 5]);
    }

    /// Each bond from a member to a then-outside site is offered exactly once,
    /// and a bond whose far end already joined is not offered at all. Offering
    /// one twice would turn the Wolff add-probability `p` into `1 − (1−p)²` —
    /// the same silent error `every_bond_is_offered_exactly_once` guards
    /// against for the full decomposition.
    #[test]
    fn a_growth_offers_each_bond_at_most_once() {
        let lat = Lattice::new([4, 6]);
        let mut offered: Vec<(usize, usize)> = Vec::new();
        // Open enough bonds that the cluster has interior sites and refused
        // candidates that later join over another bond.
        grow_cluster(&lat, 0, |i, j| {
            offered.push((i, j));
            (i * 7 + j) % 2 == 0
        });

        let mut bonds: Vec<(usize, usize)> =
            offered.iter().map(|&(i, j)| (i.min(j), i.max(j))).collect();
        let total = bonds.len();
        bonds.sort_unstable();
        bonds.dedup();
        // On extents above two, one unordered pair is one bond; a duplicate
        // offer would survive the dedup as a shrunken list.
        assert_eq!(bonds.len(), total, "some bond was offered twice");
    }
}
