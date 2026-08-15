# The Wolff single-cluster algorithm

This file is a self-contained account of the Wolff algorithm: what one move is,
why it is exact without an accept/reject step, where its efficiency comes from,
and what it costs in bookkeeping. It is written algorithm-first, as the quick
reference for the update itself; how the crate realizes it is collected in one
short section at the end, and the Fortuin–Kasteleyn machinery it shares with
Swendsen–Wang is derived once in `docs/swendsen-wang.md` and cited rather than
repeated.

Throughout, $H$ is the energy, $\beta$ the inverse temperature, $N$ the number
of sites, $q$ the number of states a site can take, and $\Delta$ the per-bond
energy gap $E(\text{disagree}) - E(\text{agree})$.

## Algorithm

One Wolff move updates one cluster, built and consumed in the same pass:

1. Pick a seed site $x$ uniformly at random and put it in the cluster.
2. For each bond from a cluster site to a site not yet in the cluster: if the
   two sites *disagree*, do nothing; if they agree, add the outside site with
   probability

   $$
   p = 1 - e^{-\beta \Delta}.
   $$

3. Repeat step 2 for every newly added site until no more sites join.
4. Relabel every cluster site to one of the other $q - 1$ labels, drawn
   uniformly. The move is never rejected.

At $q = 2$ step 4 is the flip. The add probability is the same $p$ the
Swendsen–Wang bond step uses, and it does the same job: it is exactly the rate
at which the joint spin–bond model of the Fortuin–Kasteleyn representation
occupies a bond between agreeing sites.

Four details of step 2 carry the correctness, so they are requirements rather
than choices.

**W1 — the add probability is $p = 1 - e^{-\beta \Delta}$, with $\Delta$ the
model's bond gap.** The derivation below consumes one factor of
$1 - p = e^{-\beta \Delta}$ per boundary bond, and the cancellation it rests on
holds for no other value. The gap is the model's own convention to state: the
Potts delta convention gives $\Delta = J$ while the Ising $\pm 1$ convention
gives $\Delta = 2J$, and this factor of two is the likeliest way for an
implementation to silently simulate a different temperature —
`docs/swendsen-wang.md` (SW3) makes the same point for the bond step it shares.

**W2 — each bond is tested at most once per move.** A bond from a cluster site
to an outside site is tested at the moment the cluster side is processed, and
never again: an outside site that refused one bond may be offered membership
over a *different* bond from another cluster site, but a retest of the same
bond would turn $p$ into $1 - (1-p)^2$, which is a different model. Bonds whose
far end has already joined are not tested at all — both ends move together, so
nothing depends on them.

**W3 — the seed is a uniformly random site, not a uniformly random cluster.**
Seeding at a site picks a cluster with probability proportional to its size.
That bias is part of the algorithm twice over: it appears in the detailed
balance ratio below (the seed choice contributes $|C|/N$ from either side, and
cancels only because both sides carry it), and it is the source of the
efficiency — the move preferentially lands in exactly the large correlated
regions that decorrelate slowest.

**W4 — the new label must differ from the old one.** Uniform over the other
$q - 1$ labels is symmetric between old and new, which the cancellation below
uses; drawing from all $q$ would also be valid (a lazier chain), but the forced
change is the algorithm, and at $q = 2$ it is what makes the move deterministic
given the cluster.

## Why nothing is rejected

Consider the move from $s$ to $s'$ that grows a cluster $C$ from seed $x$ and
relabels it from $a$ to $b$. The reverse move grows the same $C$ from the same
$x$ in $s'$ and relabels $b$ to $a$. Compare the two proposal probabilities
factor by factor. The seed choice is $1/N$ either way. The label choice is
$1/(q-1)$ either way, by W4. Every bond test *interior* to $C$ sees the same
agreement pattern in $s$ and in $s'$ — the cluster is monochromatic in both —
so its success and failure factors are identical on the two sides. The only
asymmetry is the boundary: bonds from $C$ to sites outside it.

A boundary bond to a site labeled $a$ agrees with the cluster in $s$, so the
forward growth tested it and it failed, costing a factor $1 - p$; in $s'$ it
disagrees and is never tested. A boundary bond to a site labeled $b$ is the
mirror image, costing $1 - p$ in the reverse growth only. Boundary bonds to any
other label are tested by neither. With $m_a$ and $m_b$ the two counts,

$$
\frac{P(s \to s')}{P(s' \to s)} = \frac{(1-p)^{m_a}}{(1-p)^{m_b}}.
$$

The energy difference lives on the same bonds and nowhere else: each
$a$-boundary bond goes from agreeing to disagreeing (cost $+\Delta$), each
$b$-boundary bond the reverse ($-\Delta$), and every other bond in the lattice
is unchanged, so $H(s') - H(s) = (m_a - m_b)\,\Delta$. Since
$1 - p = e^{-\beta \Delta}$,

$$
\frac{P(s \to s')}{P(s' \to s)} = e^{-\beta\,(m_a - m_b)\,\Delta}
= e^{-\beta\,[H(s') - H(s)]},
$$

which is detailed balance with acceptance one: the proposal ratio *is* the
Boltzmann ratio, so there is nothing left to accept against. This is the same
fact `docs/swendsen-wang.md` derives from the joint weight — the cluster
boundary is chosen to sit where the energy change costs nothing on average —
reached here by direct bookkeeping because the single-cluster move never
constructs the full bond configuration.

The same move can be read inside the Fortuin–Kasteleyn picture, and the
equivalence is worth stating because the GPU takes it literally.

**W5 — relabeling the cluster that contains a uniformly random site, in a full
Swendsen–Wang bond decomposition, is the same move.** Given the bonds, the
label conditional is uniform and independent per cluster (SW-derivation in
`docs/swendsen-wang.md`); picking the cluster under a uniform random site is
exactly W3's size-biased choice, and a forced change is a symmetric doubly
stochastic kernel on that cluster's label, which preserves the uniform
conditional. So growing one cluster (touching only it) and filtering one
cluster out of the full decomposition (touching everything) sample identical
transitions, and differ only in cost. The CPU does the first; the device does
the second, because the full decomposition is the shape of work it is good at.

## Where the efficiency comes from

Near a continuous transition a typical configuration is a landscape of
correlated patches at every scale up to the correlation length, and a local
update moves patch boundaries by diffusion — the dynamic exponent $z \approx 2$
story told in `docs/swendsen-wang.md`. A cluster built with the add probability
$p$ at the critical coupling *is* one of those patches, statistically: the move
flips a physical degree of freedom whole, at every scale, and the measured
$\tau_{\mathrm{int}}$ grows with lattice size far slower than a local update's
$z \approx 2$ — around $z \approx 0.25$ for Wolff in the two-dimensional Ising
model, a little below the multi-cluster move's measured exponent there.

The size bias of W3 is what sharpens the single-cluster move past the
multi-cluster one. Swendsen–Wang spends most of its relabeling work on the
many small clusters, which carry little of the slow dynamics; Wolff's seed
lands in a cluster with probability proportional to its size, so the work goes
where the large-scale structure is. The cost side matches: the CPU growth
touches work proportional to the cluster it flips, so the expected cost per
move is $\langle |C| \rangle$ and the algorithm self-scales — tiny cheap moves
at high temperature, lattice-sized moves near and below the transition.

Away from criticality it remains exact and stops being special. At high
temperature clusters are single sites and the move degenerates to an expensive
local flip; deep in the ordered phase the cluster is nearly the whole lattice,
so each move costs a volume and mostly toggles the global label — valuable
precisely when tunneling between ordered states matters, wasteful otherwise.
The practical guidance is the usual one: Wolff (or any cluster move) is the
tool for the neighborhood of a continuous transition, and local updates are
cheaper everywhere else.

## What a sweep is

**W6 — one Wolff sweep is one cluster move.** Every other updater's sweep is
sized to the lattice — $N$ local attempts, or one full decomposition — while a
Wolff sweep touches $|C|$ sites, a random quantity that swings from one site to
the whole box with temperature. The conventional repair, counting Monte Carlo
time in flipped sites and calling $N/\langle |C| \rangle$ moves a sweep, needs
a running estimate of $\langle |C| \rangle$, i.e. chain state the updater seam
deliberately does not hold; so the unit here is the move, stated plainly
instead of hidden in a heuristic. Two consequences follow. Warmup and
decorrelation counts for a Wolff run must be set several times larger than a
Swendsen–Wang run's, since each unit does less. And an autocorrelation time
measured in sweeps is not comparable across updaters — the point
`docs/swendsen-wang.md` (SW6) already makes for local-versus-cluster, sharpened
here because even two cluster updates now disagree about the unit. What is
comparable is the cost of one independent sample: wall-clock per sweep times
$2\tau_{\mathrm{int}}$.

## What it does not reach

The construction inherits every limit of the bond representation, and they are
argued in `docs/swendsen-wang.md` rather than here: the model's energy must be
invariant under relabeling (an external field or per-label offset breaks the
uniform conditional — SW5), the bond gap must be non-negative (a frustrated
antiferromagnet has no probability to sample — SW4), and a plaquette energy
has no pairwise bond graph at all, which is why the gauge model runs no
cluster update of either kind.

One composition of the two axes is invalid on its own terms, and it is the
mirror image of Wolff rather than Wolff itself: forcing *every* cluster to
change at $q = 2$. There the forced change is the deterministic flip, so the
move is a function of the bond draw alone — and a configuration that agrees
everywhere is one cluster under any draw, so it flips whole and flips straight
back, trapping the chain in a two-cycle between the ordered states. The seeded
extent is immune (the seed varies), and $q \ge 3$ is immune (the forced change
has real choices); the crate refuses the one bad pairing at construction.

## In the code

Wolff is `ClusterUpdate::wolff` in `src/updater.rs` — the `Extent::Seeded`,
`Relabel::ForcedChange` composition of the same two axes whose
`Extent::All`/`Relabel::Redraw` pairing is Swendsen–Wang — and reaches the
model only through `BondAction::bond_energy_gap` (W1) and
`relabel_invariant`. The CPU growth is `cluster::grow_cluster`, whose
offered-once contract is W2 and whose tests pin it. The device chain
`GpuClusterChain` (in `src/gpu_cluster.rs`) runs the W5 form: the shared bond
and labeling stages of the Swendsen–Wang shader, then a relabel stage that
keeps only the seed site's cluster and applies the forced change; it pays for
the full decomposition per move, so on the GPU Wolff buys agreement with the
CPU updater rather than speed. In a run config the rule is `updater = "wolff"`,
with the same no-schedule, any-shape, symmetric-model rules as
`"swendsen_wang"` and the W6 note about sweep counts.

## References

- U. Wolff, *Collective Monte Carlo updating for spin systems*, Phys. Rev.
  Lett. **62**, 361 (1989) — the algorithm and the boundary-bond argument.
- R. H. Swendsen and J.-S. Wang, *Nonuniversal critical dynamics in Monte
  Carlo simulations*, Phys. Rev. Lett. **58**, 86 (1987) — the multi-cluster
  predecessor.
- C. M. Fortuin and P. W. Kasteleyn, *On the random-cluster model I*, Physica
  **57**, 536 (1972) — the joint representation behind W5.
