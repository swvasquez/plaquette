# The q-state Potts model

The Potts model is a lattice of $q$-valued labels with a nearest-neighbor
interaction that rewards agreement. It generalizes the Ising model by widening
the alphabet a site can carry from two values to $q$, and in doing so it drops
the one thing the Ising alphabet had that mattered: an ordering. A Potts label is
a name, not a number, and everything that follows — what the energy can look at,
what an order parameter can be built from, why there is no external field — comes
from that.

## Lattice and labels

The lattice is a $D$-dimensional hypercubic grid of $N$ sites with periodic
boundaries, the same toroidal geometry the Ising model uses, and the interaction
acts across the same nearest-neighbor bonds — one forward and one backward along
each axis, $2D$ per site, $DN$ bonds in all. On each site sits a single label

$$s_i \in \{0, 1, \dots, q-1\},$$

so a configuration is a choice of label at every site and there are $q^N$ of them.

The set is unordered. Nothing distinguishes label $0$ from label $2$, and the
integers are an indexing convenience rather than a value: any statement about a
configuration that survives a *relabelling* — the same permutation $\pi$ of the
$q$ labels applied at every site — is physical, and any statement that does not is
an artifact of how the labels happened to be numbered. That symmetry group is
$S_q$, and it is the exact analogue of the Ising model's $\mathbb{Z}_2$ spin flip,
which is the $q = 2$ case of it. In the code this is why `Potts` never reads a
state through the private `decode` map that `Ising` and `Z2Gauge` share: it
compares two `State<Q>` values for equality and never turns one into a number.

## The agreement energy

The energy scores a configuration bond by bond, asking only whether the two
labels across a bond are the same,

$$H = -J \sum_{\langle i,j \rangle} \delta(s_i, s_j),$$

where the sum runs over each nearest-neighbor bond once and $\delta$ is $1$ when
the labels agree and $0$ otherwise. A second term, one energy offset per label,
is optional and described below; it is zero unless a run asks for it, and every
exact result quoted here assumes it is. With $J > 0$ the model is ferromagnetic:
agreeing neighbors lower the energy, so the ground states are the $q$ uniform
configurations, each with energy $-J D N$, and they are related to one another by
relabelling exactly as the two Ising ground states are related by a spin flip.

A single-site update changes only the bonds incident to that site, so its cost is

$$\Delta E = -J \left[ n(s') - n(s) \right],$$

where $n(\cdot)$ counts how many of the site's $2D$ neighbors carry a given label
and $s'$ is the proposed label. Both counts are integers, which is why
`Potts::energy_delta` accumulates in an integer and scales once at the end — the
incremental energy then agrees with the difference of two from-scratch energies
exactly rather than up to rounding. Note that the two counts are *not*
complements of one another once $q > 2$: a neighbor may carry a third label and
enter neither. That is the first place a delta written for two states quietly
fails, and it is what the three-state unit test in `models/potts` is there to catch.

## The per-label offset

There is a second term, off by default. Ising's $h$ couples to every spin the
same way because $+1$ and $-1$ are values one dial can pull between; a Potts
label is a name, so "turn up the field" means nothing until you say *which*
label you are favouring. The term is therefore a function on the label set
rather than a number,

$$
h : \{0, \dots, q-1\} \to \mathbb{R},
\qquad
H_{\text{offset}} = -\sum_i h_{s_i} = -\sum_\alpha h_\alpha N_\alpha,
$$

where $N_\alpha$ counts the sites carrying label $\alpha$. Each site pays the
entry matching its own label and nothing else, so unlike the coupling term this
one reads no geometry at all — which is what makes it, read as $-\sum_\alpha
h_\alpha N_\alpha$, literally a chemical potential per label. It is uniform in
space: every red site anywhere on the lattice contributes the same number.

Adding a constant to every entry shifts $H$ by that constant times $N$ for
*every* configuration alike, so it cancels out of every energy difference and
every Boltzmann ratio and changes nothing the chain does. Only the differences
between entries carry content, which means $q$ numbers hold $q-1$ of it, and the
usual choice is a single entry offset with the rest left at zero. That is the
smallest thing that is not nothing, and it breaks the relabelling symmetry down
to permutations of the remaining $q-1$ labels rather than destroying it outright.

Breaking that symmetry is the point of having the term. At $h = 0$ the $q$
ordered ground states are exactly degenerate, so on any finite lattice a long
enough chain visits all of them and every label-specific average washes out —
the symmetry is never really broken. Switching on a small $h$, taking the volume
to infinity, and only then sending $h$ back to zero is how spontaneous symmetry
breaking is defined at all, and it cannot be stated without the term. The second
use is the phase diagram: $h$ is its other axis, and at $q > 4$ in two dimensions
turning it up weakens the first-order transition until the line ends at a
critical point, the same structure as a liquid–gas line.

At $q = 2$ this reduces to Ising's field. Reading the labels as $\pm 1$, Ising's
$-h\sum_i s_i$ gives $+h$ to an up spin and $-h$ to a down one, so the pair
$[h, -h]$ is the same term written per label — which is what
`the_offsets_reproduce_an_ising_field_at_two_states` pins.

## Relation to the Ising model

At $q = 2$ this is the Ising model with the coupling halved. Reading the two
labels as $\sigma_i = \pm 1$ gives the identity

$$\delta(s_i, s_j) = \frac{1 + \sigma_i \sigma_j}{2},$$

so a Potts bond at coupling $2J$ is a zero-field Ising bond at coupling $J$ plus
the constant $-J$. Summed over the lattice the two energies differ by $-J D N$, a
constant that cancels in every energy *difference* — so the two models assign the
same $\Delta E$ to the same move, accept it with the same probability, and sample
the same distribution. The critical couplings agree under the same map: the
square-lattice Potts value below is $\ln(1 + \sqrt2) \approx 0.8814$ at $q = 2$,
which is twice the Ising $\tfrac12\ln(1+\sqrt2) \approx 0.4407$.

Both models are kept rather than one wrapping the other. `Ising` carries a
uniform external field that plain Potts has no counterpart for, its observables
are built on reading states as signs, and its GPU kernel updates a site by
flipping a bit where the Potts kernel has to draw among alternatives, which is a
faster specialized path. What the equivalence buys instead is a check: two
independently written actions must agree on every move, which
`potts_at_two_states_prices_moves_like_ising` pins at the action level and
`tests/potts_e2e.rs` pins again through the whole sampler stack.

## The order parameter

The Ising magnetization $M = \sum_i s_i$ has no counterpart here, and not merely
no convenient one. A sum needs its summands to be numbers, and relabelling would
change any such sum while leaving the physics untouched, so a quantity built that
way would report the labelling rather than the state. What does survive
relabelling is the *imbalance between the populations*: let $f_\alpha$ be the
fraction of sites carrying label $\alpha$ and take the largest,
$f_{\max} = \max_\alpha f_\alpha$. A permutation permutes the $f_\alpha$ and
leaves their maximum alone.

The conventional order parameter rescales $f_{\max}$ so that both ends come out
clean,

$$m = \frac{q\, f_{\max} - 1}{q - 1}.$$

In the disordered phase every label holds about $1/q$ of the sites, so
$f_{\max} \to 1/q$ and $m \to 0$; in a fully ordered configuration one label holds
all of them, $f_{\max} = 1$ and $m = 1$. On a finite lattice the disordered value
is not exactly zero — $f_{\max}$ is the maximum of $q$ fluctuating fractions and
so sits a little above $1/q$ — which is a finite-size effect that shrinks as $N$
grows, not a bias in the estimator.

One consequence is worth stating plainly, because it differs from the Ising side.
`Ising::magnetization` keeps its sign, which is what lets `statistics` recover
both $\langle m^2 \rangle$ and $\langle |m| \rangle$ from one series.
`Potts::order_parameter` is non-negative by construction and has no sign to keep:
it already stands where $\langle |m| \rangle$ does, and nothing downstream can
undo the fold.

## Two conventions for it

The definition above is not the only one in use, and the other is what most
published Potts curves are, so `PottsSample` carries both. The alternative comes
from a different construction rather than a different normalization of the same
one: place the $q$ labels at the vertices of a regular $(q-1)$-dimensional
simplex, giving unit vectors $e_\alpha$ with $\sum_\alpha e_\alpha = 0$ and
$e_\alpha \cdot e_\beta = -1/(q-1)$ for $\alpha \neq \beta$, and take the length
of their average over the sites. Expanding that square,

$$
\left| \frac{1}{N} \sum_i e_{s_i} \right|^2
= \frac{q \sum_\alpha f_\alpha^2 - 1}{q - 1},
$$

so the vectors cancel out entirely and what is left is again a function of the
populations alone. That is why `Potts::simplex_order_parameter` costs the same
scan as `Potts::order_parameter` and needs no geometry: both are reductions of
one array of label counts, one taking its maximum and the other its sum of
squares, and both are invariant under relabelling for the same reason — a
permutation reorders that array and neither reduction reads the order.

The two agree at both ends, at zero for an equal split and one for a uniform
configuration, which is what their normalizations are chosen for. They disagree
in between: half the sites on one label and half on another gives $1/4$ under the
most-populated reading and $1/2$ under the vector one, so a comparison against a
published number has to know which convention that number used. At $q = 2$ the
simplex degenerates to an interval, the two definitions become the same function,
and both reduce identically to $|M|/N$ of the corresponding Ising field — which
is how the pair is tied back to a quantity that was already tested.

The vector form is also the one behind the Binder caveat below. It is the
definition with $q-1$ components, and it is those components that make the
scalar normalization the wrong one.

## Phases and the exact critical coupling

The model orders at low temperature and disorders at high, and in two dimensions
the transition sits at a coupling known exactly. The square lattice is self-dual
under a map exchanging strong and weak coupling, and the fixed point of that map
gives

$$\beta_c J = \ln\!\left(1 + \sqrt{q}\right),$$

which is $\approx 1.0050$ at $q = 3$ — the value `examples/potts/potts.toml` and
`tests/potts_e2e.rs` are placed relative to. Self-duality locates the transition
but does not by itself say what kind it is, and the two questions have different
answers as $q$ grows: in two dimensions the transition is continuous for
$q \le 4$ and first order for $q > 4$. At $q = 3$, then, a short run near $\beta_c$
sees critical slowing down but not phase coexistence, which is why the end-to-end
tests can sit reasonably close to the transition; a $q = 5$ run could not, for the
same reason the four-dimensional gauge tests keep well clear of their own first
order transition.

Duality gives one more exact number, and it is the only one a sampled chain here
can be read against directly. Differentiating the duality relation at the
self-dual point fixes the internal energy there,

$$
\frac{E_c}{N} = -J\left(1 + \frac{1}{\sqrt{q}}\right),
$$

per site on the square lattice, where each site carries two forward bonds. At
$q = 2$ this is checkable against Onsager independently: the exact critical Ising
energy is $-\sqrt2 J_I$ per site, and pushing it through the constant offset
between the two models lands on exactly $-J(1 + 1/\sqrt2)$. This is what
`the_critical_energy_matches_the_exact_duality_value` asserts, and it is the
Potts counterpart of the exact area law the gauge suite is checked against.

Reading it accurately is harder than stating it, and the two obstacles push
against each other. A finite lattice is more ordered than an infinite one at the
same coupling, so the measured energy sits *below* the exact value and
approaches it from that side as $L$ grows; measured at $q = 3$ on this code the
gap runs $0.038, 0.028, 0.025, 0.012$ at $L = 16, 24, 32, 48$, close to a $1/L$
decay. Slower decay than that is expected asymptotically — the singular term
falls only as $L^{-(1-\alpha)/\nu} = L^{-4/5}$ at $q = 3$ — but over this range
an analytic $1/L$ background dominates it, so the fitted exponent should not be
read as either. Either way the convergence is slow.

Growing the lattice to close that gap runs into the other obstacle. Sitting
exactly on $\beta_c$ is where a local update decorrelates most slowly, and the
autocorrelation grows roughly as $L^2$, so past about $L = 32$ the seed-to-seed
scatter of an affordable run has caught up with the ground a larger box would
gain. `the_critical_energy_sits_just_below_the_exact_duality_value` therefore
uses a small box and a tolerance sized to a known correction rather than a large
box and a tight one, and asserts the *sign* of the gap alongside its size —
noise does not keep a sign, so that is what distinguishes a finite-size
correction from a wrong number. Escaping the trade altogether is what the cluster
algorithms below are for.

Above two dimensions there is no closed form, and the transition becomes first
order for smaller $q$ — in three dimensions already at $q = 3$. Nothing in the
code depends on this; it is the reason a three-dimensional test here asserts a
contrast between two couplings rather than a number.

## Measuring correlations

The two-point function needs the same treatment the order parameter did: a
product of labels is meaningless, so what is correlated is *agreement*. The raw
agreement fraction does not decay to zero, though — two sites far enough apart to
be independent still agree with probability $1/q$ — so the floor is subtracted,

$$C_r = \langle \delta(s_x, s_{x+r}) \rangle - \frac{1}{q},$$

leaving a quantity that falls to zero at large separation in the disordered phase
exactly as the Ising $\langle s_x s_{x+r} \rangle$ does, and from which a
correlation length can be read. Unlike the Ising estimator this is already
*connected*, so nothing is left for a downstream subtraction to do. The anchor at
$r = 0$ is $1 - 1/q$ rather than $1$, since a site always agrees with itself.
`Potts::correlator` returns one row per axis over displacements
$r = 0, \dots, L_\mu/2$, the same non-redundant half `Ising::correlator` stores.

## Reductions and the Binder convention

Most of `statistics` is model-agnostic and takes a Potts series unchanged.
`reduce` and `specific_heat` read the energy series directly; `susceptibility`
takes the order-parameter series, and since it computes the variance of the
magnitude it never sees a sign to begin with.

`binder_cumulant` is the one that needs a word. It computes

$$U_4 = 1 - \frac{\langle m^4 \rangle}{3 \langle m^2 \rangle^2},$$

and the $3$ is a convention rather than a universal constant. It is calibrated so
that a *signed* order parameter, symmetric about zero and near-Gaussian in the
disordered phase, satisfies $\langle m^4 \rangle = 3\langle m^2 \rangle^2$ and
gives $U_4 = 0$ there, while sharp ordering gives
$\langle m^4 \rangle = \langle m^2 \rangle^2$ and $U_4 = 2/3$ at the other end.
The Potts order parameter above is non-negative and does not average to zero on a
finite lattice, so its ordered limit is still $2/3$ but its disordered value is a
size-dependent number rather than the Ising anchor. This does not stop the
cumulant doing its job — locating $\beta_c$ by where curves for different $L$
cross needs only a dimensionless ratio of moments, not particular endpoints — but
the endpoints should not be read as though they were Ising's.

## Cluster updates

The algorithms Potts is best known for are not implemented. Swendsen–Wang and
Wolff build a cluster of like-labelled sites by opening each agreeing bond with
probability $1 - e^{-\beta J}$ and then relabel the whole cluster at once, which
near criticality moves the configuration much further per sweep than any
single-site scheme can — the correlation time grows far more slowly with the
lattice size, where local Metropolis suffers the full critical slowing down.

They are absent because they are a different *shape* of move rather than a
different schedule for the same one. Everything here — `Metropolis`,
`SiteCheckerboard`, and the GPU kernel alike — is built from a single-variable
propose-and-accept step over a set of variables chosen in some order, whereas a
cluster move constructs its own set stochastically, changes an unbounded number
of variables at once, and is accepted with probability one. That is a question
about what the `Updater` seam should be, not an addition behind the one that
exists, and it is worth answering when a second cluster algorithm is on the table
rather than guessed at from the first.
