# The Metropolis algorithm

The Metropolis algorithm samples states approximately according to the Boltzmann
distribution, without ever computing that distribution directly. It does so by
building a Markov chain — a rule that turns the current configuration into the
next one — arranged so that, run long enough, the configurations it visits appear
with approximately their Boltzmann frequencies.

## Target distribution

The distribution we want to sample is the Boltzmann distribution at inverse
temperature $\beta$,

$$P(C) = \frac{1}{Z}\, e^{-\beta H(C)}, \qquad Z = \sum_{C} e^{-\beta H(C)},$$

where $H(C)$ is the energy of a configuration $C$ and $Z$ normalizes the weights.
The obstacle is $Z$: the sum runs over a state space far too large to enumerate,
so $P(C)$ cannot be evaluated for any single configuration. What can be evaluated
is a *ratio* of two weights,

$$\frac{P(C')}{P(C)} = e^{-\beta\,(H(C') - H(C))} = e^{-\beta\,\Delta E},$$

in which $Z$ cancels and only the energy difference $\Delta E$ survives. The whole
algorithm is built to need nothing more than this ratio.

## Algorithm

One step of the chain takes the current configuration $C$ to the next by a
propose-then-accept-or-reject move:

1. Propose a candidate $C'$ by making a small, reversible change to $C$ — one
   drawn from a rule that is as likely to propose $C'$ from $C$ as $C$ from $C'$.
2. Compute the energy change $\Delta E = H(C') - H(C)$.
3. Accept $C'$ with probability

$$A(C \to C') = \min\!\left(1,\; e^{-\beta\,\Delta E}\right),$$

   and otherwise reject it and keep $C$ as the next configuration.

A downhill move ($\Delta E \le 0$) is always accepted; an uphill move is accepted
with probability $e^{-\beta\,\Delta E}$, less than one but never zero. Rejecting
still produces a step — the chain stays where it is and that repeated
configuration counts as the next sample. Repeating the step generates the chain.

## Correctness

The move is engineered so that the Boltzmann distribution is unchanged by it: feed
in configurations already distributed as $P$, apply one step, and they are still
distributed as $P$. The condition that guarantees this is *detailed balance*,
which asks that in equilibrium each transition and its reverse carry equal
probability flow,

$$P(C)\, W(C \to C') = P(C')\, W(C' \to C),$$

where $W$ is the probability of stepping from one configuration to the other. With
a symmetric proposal the proposal probabilities cancel, and detailed balance
reduces to a condition on the acceptance alone,

$$\frac{A(C \to C')}{A(C' \to C)} = \frac{P(C')}{P(C)} = e^{-\beta\,\Delta E}.$$

The Metropolis acceptance rule satisfies this identity exactly: for an uphill move
one direction accepts with probability $e^{-\beta\,\Delta E}$ and the reverse
downhill move accepts with probability one, and their ratio is $e^{-\beta\,\Delta
E}$ as required. Because the rule depends only on $\Delta E$, the unknown $Z$
never appears.

Detailed balance makes $P$ a fixed point of the update, but a fixed point is only
reached if the chain can get anywhere. The moves must also be *ergodic*: from any
configuration, some sequence of them must be able to reach any other. A single
kind of move need not manage this alone — a local change to one site goes almost
nowhere by itself — so ergodicity is a property of the whole *set* of moves taken
together, one for each site, applied in sequence so that between them they can
reach the entire space. A chain whose moves satisfy detailed balance but cannot
between them explore everywhere settles into the wrong distribution. Given both
conditions, the chain has a unique equilibrium equal to $P$.

## Sampling

Reaching that equilibrium is a limit, approached by running the moves in sequence
rather than delivered by any single step, so sampling means running the chain for
many steps and reading configurations off it as it goes. How that reading is done
follows from one feature of the sequence: consecutive configurations differ by at
most one accepted move, so they are strongly correlated, not independent draws.
Keeping every step would pile up near-duplicates and overstate how much the
samples say. Instead we sample at spaced intervals — retaining one configuration
every so many steps and discarding those in between — so the kept configurations
are far enough apart to be nearly independent. Taken together, the spaced samples
from a long enough run look like independent draws from the Boltzmann
distribution, and averaging an observable over them estimates its Boltzmann
average.

## Kernels, schedules, models, backends

Before the checkerboard sections, it is worth naming the factoring the code is
built around, because the derivations below slot into it. A *kernel* is what
happens at one variable once it has been chosen — the propose-and-accept rule
above, or the heat bath of `docs/heat-bath.md` — and it talks to the *model*
only through the price of a move, `Action::energy_delta` on the CPU and a WGSL
`energy_delta` on the GPU, receiving a bare variable index either way. A
*schedule* is which variables are chosen in what order — uniformly at random,
or the checkerboards below — and it talks only to the lattice. A *backend* is
where the composition runs: sequentially on the CPU, where any order is valid,
or as parallel color passes on the GPU, where the checkerboard's independence
argument is load-bearing. `LocalUpdate` in `src/updater.rs` is the CPU
composition and `device::assemble_shader` the GPU one; the correctness
arguments in this file attach to the schedule and kernel separately, which is
why they hold for every model that meets the seam.

## Checkerboard on sites

The sampling above runs its moves in sequence, each seeing what the last one left,
and that is what we want to escape on a GPU, where the point is to update many sites
at once. A naïve parallel sweep is wrong, though: two neighboring sites updated
simultaneously each price their move against the other's old value, double-counting
the bond between them and breaking detailed balance. The checkerboard schedule fixes
this by making the sites updated together never interact. It colors each site by
the parity of its coordinate sum, $\left(\sum_\mu x_\mu\right) \bmod 2$; a step
along any axis flips the parity, so a site's neighbors are all the opposite color.
A whole color then contains no neighboring pairs, and can be updated at once
against a fixed background of the other color. A sweep is two passes — all of
color 0, then all of color 1 — with a barrier between them so the second reads the
first's new values.

This still samples $P$ because it only reorders the single-site moves. Each move
leaves $P$ invariant on its own, and invariance composes,

$$P T_1 = P, \quad P T_2 = P \;\implies\; P\,(T_1 T_2) = P,$$

for any order, so a sweep in color order preserves the Boltzmann distribution just
as a random one does. Updating a color in parallel is the same product, since
within a color the simultaneous pass equals doing its sites in sequence. On one
processor the schedule is thus reordered Metropolis, differing only in
autocorrelation — which is what makes the sequential version the reference a
parallel one is checked against.

Two conditions bind the parallel case alone. The independence within a color needs
even extents: under periodic boundaries an odd extent wraps a site onto a
same-color neighbor, restoring the double-counting. And matching a parallel run
bit-for-bit needs a random source keyed by each site's coordinates rather than a
stream advanced in visiting order. Run sequentially neither applies, so the CPU
checkerboard is correct on any lattice.

Nothing in that argument mentions how many values a site can take. What it uses
is that the interaction is nearest-neighbor and that the lattice is bipartite, so
that a two-coloring exists in which no site interacts with another of its own
color; the acceptance step then leaves $P$ invariant whatever a single move does
to one site. The schedule therefore carries from Ising to the $q$-state Potts
model unchanged, and the checkerboard `Schedule` of `LocalUpdate` implements
it once for any $q$ rather than once per model. What does generalize is the *proposal*. At two states the
alternative is determined and the move is a flip; at $q$ states there are $q-1$
alternatives, so the proposal draws one uniformly among them. That keeps it
symmetric — every other label is offered with probability $1/(q-1)$ from any
label — which is the condition the acceptance rule needs to drop the proposal
probabilities above, and it reduces to the flip at $q = 2$.

One thing the coloring does not carry is the cluster update. A checkerboard is a
device for making single-site moves independent of one another, and a cluster
move has no single-site moves to separate — its whole point is that the sites it
changes are chosen together, precisely because they interact. So there is no
checkerboard variant of Swendsen–Wang, and the parallelism a device run gets
from it comes from somewhere else entirely; `docs/swendsen-wang.md` says where.

## Checkerboard on links

A gauge model poses the same problem and needs a different coloring. Its variables
sit on links rather than sites, and its unit of interaction is the plaquette — four
links around a face — so what two variables updated together must avoid is not
sharing a bond but sharing a plaquette. Base-site parity alone does not deliver
that. The four links of the plaquette based at $i$ in the $\mu\nu$-plane are
$(i,\mu)$, $(i+\hat\mu,\nu)$, $(i+\hat\nu,\mu)$ and $(i,\nu)$, and the first and
last share a base site as well as a plaquette, so any rule that reads only the site
would place them in the same color.

Splitting first by direction removes that case. Fix a direction $\mu$ and let a pass
update only $\mu$-links; every link of another direction is then frozen, and the
only collisions left are between two $\mu$-links lying in one plaquette. A plaquette
in the $\mu\nu$-plane holds exactly two of them, $(i,\mu)$ and $(i+\hat\nu,\mu)$,
whose base sites differ by a single step along some $\nu \neq \mu$. A step along any
axis flips the coordinate-sum parity, so those two always carry opposite base-site
parity, and coloring a link by the pair

$$\text{color}(i,\mu) = \left(\mu,\ \Big(\textstyle\sum_\nu x_\nu\Big) \bmod 2\right)$$

separates them. Each of the $2D$ colors is then collision-free, and a sweep is $2D$
passes rather than the site version's two — six in three dimensions.

The staples confirm the count is right. Updating $(i,\mu)$ prices the move against
its $2(D-1)$ staples, each of three links: two of direction $\nu \neq \mu$, frozen
because the pass touches only $\mu$; and one further $\mu$-link based at $i \pm
\hat\nu$, which the parity split has already placed in the other color. Nothing a
move reads can change while that move is being made.

Correctness carries over unchanged, and for the same reason. The schedule only
reorders single-link Metropolis moves, each of which leaves $P$ invariant on its
own, and invariance composes in any order — so the color order samples $P$ exactly
as a random order does, and a parallel pass over one color equals doing that
color's links in sequence. The even-extent condition transfers too: an odd extent
along any direction wraps a base site onto a same-parity one, putting two links of a
shared plaquette back in the same color. Run sequentially none of this binds, so
the CPU version is a valid schedule on any lattice and serves as the reference a
parallel gauge sweep is checked against.
