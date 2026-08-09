# Z2 gauge theory

The Z2 gauge theory is a lattice of two-state variables with an interaction built
from products around the smallest closed loops. It shares the Ising model's
alphabet — every variable is a $\pm 1$ — but moves those variables off the sites
and onto the links between them, and scores a configuration by the elementary
square faces of the lattice rather than by nearest-neighbor bonds. That shift is
the whole character of the model: it introduces a local symmetry, a redundancy in
the description that no site model has, and everything downstream — which
configurations count as the same, which quantities are meaningful to measure —
follows from it.

## Lattice and links

The lattice is a $D$-dimensional hypercubic grid of $N$ sites with periodic
boundaries, the same toroidal geometry the Ising model uses. What differs is where
the degrees of freedom sit. Instead of one variable per site, a variable lives on
each *link* — each edge joining a site to its nearest neighbor. A site has one
forward link along each of the $D$ axes, so the lattice carries $DN$ links in all,
and a link is named by its base site $i$ together with the axis $\mu$ it points
along, written $\ell = (i, \mu)$.

The links in turn bound the elementary square faces, called *plaquettes*. A
plaquette is fixed by a base site and a pair of distinct axes $(\mu, \nu)$: it is
the unit square in the $\mu\nu$-plane whose four sides are the links $(i, \mu)$,
$(i + \hat\mu, \nu)$, $(i + \hat\nu, \mu)$, and $(i, \nu)$. There are
$\binom{D}{2}$ such planes, so each site anchors that many plaquettes and the
lattice has $\binom{D}{2} N$ of them. Two dimensions is the floor: below it no
pair of distinct axes exists, so there are no plaquettes and no theory. Three
dimensions is the one where the plaquette count $3N$ happens to equal the link
count, a coincidence of $\binom{3}{2} = 3$ rather than anything structural.

## The link variable

On each link sits a single two-state variable,

$$\sigma_\ell = \pm 1,$$

so a configuration is a choice of $\sigma_\ell$ on all $DN$ links, and there are
$2^{DN}$ of them. The variable is an element of the group $\mathbb{Z}_2 = \{+1,
-1\}$ under multiplication, and reading it as a group element rather than a spin is
what generalizes: replacing $\mathbb{Z}_2$ by a larger group is the path from this
model toward the continuous gauge theories that are the longer-term aim.

## The plaquette energy

The energy scores a configuration by its plaquettes. To each plaquette belongs the
product of the four link variables around its boundary,

$$\sigma_\square = \prod_{\ell \in \partial\square} \sigma_\ell,$$

itself a $\pm 1$, and the energy sums this over every plaquette with a coupling
$J$,

$$H = -J \sum_{\square} \sigma_\square.$$

This is the Wilson action specialised to $\mathbb{Z}_2$. With $J > 0$ the energy
is lowest when every plaquette product is $+1$, the analogue of the ferromagnetic
alignment that lowers the Ising energy, and the Boltzmann weight $e^{-\beta H}$
carries the coupling in the same combination $\beta J$ that governs the Ising
model. The single structural change from Ising is the unit of interaction: a
four-link product around a face in place of a two-spin product across a bond.

## Gauge invariance

The plaquette energy has a local symmetry that the Ising energy does not. A *gauge
transformation* chooses a sign $\eta_i = \pm 1$ independently at every site and
flips each link by the signs at its two ends,

$$\sigma_{(i,\mu)} \;\longrightarrow\; \eta_i \, \sigma_{(i,\mu)} \, \eta_{i+\hat\mu}.$$

Around any plaquette each corner site is shared by exactly two of the four links,
so its $\eta$ enters the plaquette product twice and squares to one. Every corner
cancels this way, $\sigma_\square$ is left unchanged, and therefore so is the whole
energy. The transformation is *local* — an independent choice at each of the $N$
sites, $2^N$ transformations in all — and it maps a configuration to a physically
identical one. Configurations related by a gauge transformation are not distinct
states but the same state described differently, and the $2^{DN}$ configurations
collapse into far fewer gauge-inequivalent classes.

This redundancy is the defining feature of a gauge theory and the sharpest break
from the Ising model, where every configuration is a distinct physical state. It
also dictates what can be measured. Because the symmetry is exact and local,
Elitzur's theorem forces the average of any gauge-dependent quantity to vanish:
the link variable itself averages to zero, so there is no local order parameter
and nothing plays the role the magnetization plays for Ising. Only gauge-invariant
quantities — ones built, like the plaquette, from links multiplied around closed
loops — survive averaging.

## Observables and phases

The gauge-invariant observable that distinguishes the phases is the *Wilson loop*,
the product of the link variables around a closed rectangular loop $C$,

$$W(C) = \prod_{\ell \in C} \sigma_\ell.$$

It is gauge invariant for the same reason the plaquette is — every site the loop
visits is touched by two of its links — and the plaquette is just its smallest
case. How its average $\langle W(C) \rangle$ falls off with the size of the loop
separates the two phases of the theory. At small $\beta$ (strong coupling) it
decays with the *area* the loop encloses, the confined phase; at large $\beta$
(weak coupling) it decays only with the loop's *perimeter*, the deconfined phase.
Whether the crossover between the two laws is a genuine transition depends on the
dimension. In two it is not: the theory confines at every coupling, and the area
law holds exactly, which is what makes two dimensions the check the measurement is
pinned against below. From three up there is a transition at a finite critical
coupling.

In three dimensions that transition is tied to the Ising model by a duality:
three-dimensional $\mathbb{Z}_2$ gauge theory is dual to the three-dimensional
Ising model, with the gauge coupling and the Ising temperature related by the
duality map, so the gauge transition sits at the image of the Ising critical point
(near $\beta_c \approx 0.76$). The two models are the same physics read on dual
lattices — which is why a framework that already samples the Ising model is the
natural place to build this one. Four dimensions is dual to *itself*, and the
self-duality fixes its transition exactly at $\beta_c = \tfrac12 \ln(1 +
\sqrt2) \approx 0.4407$; unlike the three-dimensional one it is first order, so
the two phases coexist there and a short run near it stays in whichever it
started in.

## Measuring the loops

A single loop product is a $\pm 1$ and says nothing on its own, so what the code
measures is an average over a whole class of loops at once. `Z2Gauge::wilson_rectangles`
returns a table whose entry $(r, t)$ is the mean of $W(C)$ over every $r \times t$
rectangle on the lattice: based at every site, in every plane, and with the two
side lengths assigned to the plane's two directions both ways round. Every member
of that class has the same expectation, so averaging them estimates it with far
less noise, and the table comes out symmetric in $r$ and $t$ by construction.
Sides run from $0$ to a requested maximum, capped at half the shortest extent,
because a wider rectangle wraps the torus far enough to meet itself; row and
column zero hold the $1.0$ that a zero-width rectangle gives. One table comes out
per configuration, and $\langle W(r,t) \rangle$ is the chain average of one entry
across that series.

What the average contains is clearest in its logarithm, which splits into the area
the loop encloses, its perimeter, and a constant,

$$-\log \langle W(R,T) \rangle = \sigma R T + \mu (R + T) + c,$$

where $\sigma$ is the string tension — the coefficient of the area law that the
confined phase is named for — and $\mu$ absorbs the self-energy carried by the
loop's edges. Only the first term is wanted, and the *Creutz ratio* isolates it by
combining four loops whose sides differ by one,

$$\chi(R,T) = -\log \frac{\langle W(R,T) \rangle \, \langle W(R-1,T-1) \rangle}
{\langle W(R-1,T) \rangle \, \langle W(R,T-1) \rangle}.$$

The perimeters cancel, $(R+T) + (R+T-2) - 2(R+T-1) = 0$, the constants cancel in
pairs, and the areas leave $RT + (R-1)(T-1) - (R-1)T - R(T-1) = 1$, so
$\chi(R,T) = \sigma$ up to the short-distance corrections the decomposition
neglects at small loops — which is why `statistics::creutz_ratio` reports one value
per $(R,T)$ and leaves reading the plateau to the caller. It lives in statistics
rather than beside the measurement because it is a nonlinear function of four
ensemble averages rather than the average of anything measured per configuration:
its error has to be propagated through that function, which is what the blocked
jackknife it shares with the fluctuation quantities does. When the ratio inside the
logarithm is not positive — a run too noisy to resolve $\chi$ at that size — it
returns `NaN` rather than a clamped stand-in.

Two dimensions is the exact check on both halves of this. Fixing the gauge there
leaves one free variable per plaquette, so the plaquettes are independent, each
averages to $\tanh(\beta J)$, and a rectangle — the product of the plaquettes it
encloses — averages to

$$\langle W(R,T) \rangle = \tanh(\beta J)^{RT}.$$

The measurement is tested against that directly, and since the Creutz combination
of areas is $1$ at every size, the ratio must return $-\log \tanh(\beta J)$ for
*any* $R$ and $T$, which is what pins the estimator.

The one remaining loop is the one no rectangle can deform into.
`Z2Gauge::polyakov_loop` multiplies the links down a straight line wrapping the
torus along a chosen direction and averages that over the lines, one per position
across the direction. Multiplying every link along that direction in a single slice
is a further symmetry of the energy — each plaquette crosses the slice with two of
its links or with none — and it flips every one of these products at once, so the
signed average vanishes in the confined phase by symmetry and is nonzero only when
that symmetry breaks. Keeping the sign is what makes it the order parameter for
deconfinement.
