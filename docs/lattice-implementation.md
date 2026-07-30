# Lattice geometry

This file covers what `Lattice<D>` in `src/lattice.rs` names and how it names it:
the cells of a periodic hypercubic lattice, the integer index each one carries,
and the incidence relations between them. It is pure geometry — no configuration,
no action, no sampler — but the packings it fixes are the contract every model
built on top reads, so they are stated here as labelled requirements `L1`–`L7`
that code comments and tests can cite. The physics that motivates the link and
plaquette machinery is in [`z2-gauge.md`](z2-gauge.md).

## Cells

The lattice is a cubical complex: a $D$-dimensional grid of $N$ sites with
periodic boundaries, together with the edges and square faces they span. Read it
by dimension and there are three kinds of cell in play, and each one is fixed by
the same two pieces of data — a base site, and an orientation saying which
directions it extends in from that base. A site extends in no direction, so its
orientation is empty; a link extends along one direction $\mu$; a plaquette spans
an unordered pair $(\mu, \nu)$ with $\mu < \nu$. Every cell extends in the
*positive* directions only, so each is owned by exactly one site and the counts
come out as $N$, $DN$, and $\binom{D}{2}N$.

That uniformity is what the naming scheme reflects. Each kind gets a count, a
function packing a base position and however many directions it needs into a
linear index, and accessors for the parts:

| cell | count | index | parts |
|---|---|---|---|
| site | `n_sites` | `site_index(coords)` | `site_coords(s)` |
| link | `n_links` | `link_index(coords, mu)` | `link_site(l)`, `link_direction(l)` |
| plaquette | `n_plaquettes` | `plaquette_index(coords, mu, nu)` | `plaquette_site(p)`, `plaquette_directions(p)` |

**L0. `coords` means lattice coordinates and *direction* means $\mu$, with no
second use of either word.** An earlier revision had `link_coords` return
`(site, axis)` — a packed site index and a direction, neither of them
coordinates — and had `Direction` name the forward/backward sign while $\mu$ was
called an axis. Both collisions are the same mistake: a word doing two jobs.
Hence the sign enum is `Sign::Plus`/`Sign::Minus`, and no accessor returns a
site and a direction fused into one value. That last point follows the
literature, which writes a link as $U_\mu(n)$ and never names the pair — see
`L2`.

**L1. Sites are packed mixed-radix with direction 0 fastest.** A coordinate tuple
$(x_0, \dots, x_{D-1})$ maps to

$$s = x_0 + L_0\,(x_1 + L_1\,(x_2 + \cdots)),$$

so a step along direction $\mu$ moves the index by the place value
$L_0 L_1 \cdots L_{\mu-1}$. Unequal extents are allowed and give an anisotropic
lattice. Everything downstream — the neighbour table, the checkerboard colouring,
the GPU buffer layout — assumes this order, so changing it silently changes what
those tables mean.

**L2. Links are packed `site * D + direction`, direction fastest.** The link
$(s, \mu)$ is the forward edge from $s$ to its $+\mu$ neighbour, and a site's $D$
links occupy one contiguous run of the index space, mirroring the neighbour
table's per-site row. Contiguity is the point: an updater walking the links of a
site reads a cache line rather than striding.

Fusing the site and direction into a single integer is a deliberate departure
from the field. Grid, QDP++, and Chroma keep them apart, storing the gauge field
as $D$ site-indexed fields addressed `u[mu][site]`, which is why the literature
never needs a name for the pair. We fuse them because a link then looks like a
site to everything upstream: a gauge configuration is a `Configuration` of
`n_links` variables, and the existing sampler, updater, and observables carry it
unchanged. The cost is that the packing is an implementation detail masquerading
as geometry, so it is kept private — `link_at(site, dir)` and `link_base(link)`
do the arithmetic, the public `link_index` takes coordinates, and the parts come
back separately through `link_site` and `link_direction`.

**L3. Plaquettes are packed `site * C(D,2) + pair`, pair fastest, with `pair`
lexicographic in $(\mu, \nu)$.** In three dimensions that means the pairs
$(0,1), (0,2), (1,2)$ take ordinals $0, 1, 2$, so plaquette $3s + 1$ is the
$xz$-face at site $s$. The ordinal is computed from the pair, and back, without a
lookup table: the pairs whose first direction is below $\mu$ number
$\mu(D-1) - \mu(\mu-1)/2$, and $\nu - \mu - 1$ more precede $(\mu, \nu)$ inside
its own group. `n_plaquettes` is zero when $D < 2$, where no pair exists, and that
case must not panic. As with links, the packed form lives in a private
`plaquette_at(site, mu, nu)`, the public entry point takes coordinates, and the
parts come back through `plaquette_site` and `plaquette_directions`.

## Incidence

Naming the cells is half the geometry; the other half is which cells touch which.
One rule covers it: `a_bs(a)` returns the $b$-cells incident to the $a$-cell `a`,
so reading down a dimension gives a cell's boundary and reading up gives the cells
that contain it.

| cell | boundary (down) | contained in (up) |
|---|---|---|
| site | — | `site_links(s)` → $2D$ |
| link | `link_sites(l)` → 2 | `link_plaquettes(l)` → $2(D-1)$ |
| plaquette | `plaquette_links(p)` → 4 | — |

Site-to-site adjacency sits outside that table because it does not change
dimension: `site_neighbors`, `site_neighbor`, and `site_shift` are displacements
within the site set, all wrapping periodically. `site_neighbor` takes a direction
and a `Sign`, so `site_neighbor(s, mu, Sign::Minus)` is the site at $s - \hat\mu$.

**L4. A plaquette's four links come back in the fixed order
$[(s, \mu), (s + \hat\mu, \nu), (s + \hat\nu, \mu), (s, \nu)]$.** This traverses
the boundary as a loop — out along $\mu$, up along $\nu$, back along $\mu$, down
along $\nu$ — with the last two links traversed backwards. For $\mathbb{Z}_2$ the
direction is immaterial, since each variable is its own inverse, but for any
larger gauge group the ordered product $U_\mu(s) U_\nu(s+\hat\mu)
U_\mu(s+\hat\nu)^{-1} U_\nu(s)^{-1}$ is the plaquette, and it needs exactly this
order. Fixing it now costs nothing and keeps the door open.

A plaquette is *anchored* at its base site rather than centred on it: the base is
one of the four corners, and the square extends from there along the two positive
directions, so its corners are $s$, $s + \hat\mu$, $s + \hat\nu$, and
$s + \hat\mu + \hat\nu$. That one-to-one ownership by a corner is what makes the
packing in `L3` exact — a centred convention would have no site to assign each
plaquette to.

**L5. Each of a plaquette's four corner sites is touched by exactly two of its
four links.** This is what makes the plaquette product gauge invariant: a gauge
transformation multiplies each link by the signs at its two ends, so each corner's
sign enters the product twice and squares away. It is also a cheap and sharp test
of the enumeration, since a mis-stepped neighbour breaks the closure immediately.
With any extent equal to 1 the loop degenerates — a link and its periodic image
coincide, so the four links are no longer distinct — which is legitimate but worth
knowing before trusting a small test lattice.

## Staples

One accessor deliberately breaks the naming rule, because it is not incidence but
a composition of two steps of it: from a link, up to the plaquettes containing it,
then down to their links, minus the link itself. Lattice gauge theory already
calls that object the *staple* — delete one side of a square and the three that
remain form a U, two legs and a crossbar — and `link_staples(l)` returns them as a
flat row of $6(D-1)$ entries, one group of three per containing plaquette. In
three dimensions a link therefore has four staples, fanned around it like the
vanes of a paddle wheel.

The table is the gauge counterpart of the neighbour table, and the two are
instances of one construction: when the variables live on $k$-cells and the
action sums over $(k+1)$-cells, what an update needs is, for each $k$-cell, the
other $k$-cells of every $(k+1)$-cell containing it. That gives $2(D-k)$ groups of
$2k+1$. At $k = 0$ — Ising, variables on sites, energy on bonds — it degenerates
to $2D$ groups of one, which is exactly the neighbour row. At $k = 1$ it is
$2(D-1)$ groups of three, hence the stride $6(D-1)$. Nothing here builds the
$k = 2$ case, and the literature has no name for it, but the shape is fixed:
$2(D-2)$ groups of five.

**L6. Staple group $g$ of link $\ell$ is the $g$-th plaquette containing $\ell$
with $\ell$ removed, and groups run in increasing plaquette index.** The table is
built by walking plaquettes in index order and, for each of a plaquette's four
links, appending the other three, so consistency with `plaquette_links` holds by
construction rather than by agreement between two enumerations. The ordering
guarantee is what lets `link_plaquettes(l)` be zipped against the groups: the two
answer the same question, one by index and one by contents.

**L7. A single-link flip reads its energy change from this table alone.** The
plaquette product factorises as $\sigma_\square = \sigma_\ell \prod_{k \in g}
\sigma_k$ for the group $g$ belonging to that plaquette, and flipping $\sigma_\ell$
negates every plaquette containing it and no other, so

$$\Delta E = 2 J\, \sigma_\ell \sum_{g} \prod_{k \in g} \sigma_k,$$

summed over the link's $2(D-1)$ groups. This is the reason the table exists.
Rederiving the geometry per proposed flip — unpacking the link, looping over
directions, taking neighbours forward and backward — would cost more than the
arithmetic it feeds, and a sweep proposes a flip on every link. No updater reads
the table yet; it is built for the one that will.

## Storage

Two tables are materialised at construction, both flat `Vec<usize>` with a fixed
stride: the neighbour table, $2D$ entries per site ordered $+0, -0, +1, \dots$,
and the staple table, $6(D-1)$ entries per link. Nothing else is stored — sites,
links, and plaquettes have no representation beyond their integer index, and the
`*_index` functions and the accessors that recover a cell's site and directions
are arithmetic on `shape`. Which accessors read a table and which recompute is
therefore not something a caller needs to track: with the single exception noted
below, every one of them is constant-time and safe in an inner loop.

Two accessors return more entries than an array can hold, since $D$ is a const
parameter and array lengths cannot depend on it, and they resolve that
differently. `site_links` yields its $2D$ links one at a time and allocates
nothing: the caller that asks most often is the gauge transformation, which
flips every link at a site and so runs once per site per sweep, and building a
list that often would cost more than the flips it enables. `link_plaquettes`
collects into a `Vec`, because its contract is that the plaquettes come back in
increasing index — which they do not do naturally, so they must be gathered and
sorted — and nothing on the energy path calls it, a flip reading `link_staples`
instead.

The memory cost is dominated by the staple table: $6(D-1)$ words per link is 12
words per link in three dimensions, or 36 words per site, which at $64^3$ is
roughly 75 MB of `usize`. Narrowing the indices to `u32` would halve it and is the
obvious move if the GPU path wants these tables resident, but nothing forces it
yet.

## Status

As of writing, this doc reflects `src/lattice.rs` with periodic boundaries
hardcoded in every direction; the type carries a TODO for making them swappable
(open, antiperiodic, twisted), which would change what the neighbour table means
without changing any packing above. The plaquette and staple machinery is
geometry only — no `Configuration`, `State`, or `Action` is involved — and is
covered by tests for the counts, the index round-trips (through four dimensions),
hand-computed plaquettes on a $3^3$ lattice, loop closure, and agreement between
the staple groups and the plaquette enumeration in two and three dimensions.

The vocabulary settled late and the earlier revisions are worth knowing about,
since the tests and comments were rewritten with it: `Direction` became `Sign`
once *direction* was reserved for $\mu$, and the `link_coords` and
`plaquette_coords` accessors were removed in favour of separate `link_site` /
`link_direction` and `plaquette_site` / `plaquette_directions`, so that no public
function returns a site fused with a direction.
