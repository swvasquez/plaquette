# The Swendsen–Wang cluster algorithm

This file covers the cluster update the crate runs for the $q$-state Potts model
and for Ising: why a local update becomes useless near a continuous transition,
how the Fortuin–Kasteleyn bond representation turns the spin model into a
percolation problem, why the resulting move never has to be rejected, and which
models the construction does not reach. The code it describes is
`SwendsenWang` in `src/updater.rs`, the model-side seam `BondAction` in
`src/action.rs`, the labeling in `src/cluster.rs`, and the device backend in
`src/models/potts/gpu_cluster.rs`. `docs/metropolis.md` is the companion piece
for the local updates, and everything here assumes its account of detailed
balance.

Throughout, $H$ is the energy, $\beta$ the inverse temperature, $N$ the number
of sites, $D$ the dimension, and $q$ the number of states a site can take.

## Critical slowing down

The reason to build any of this is that a single-site update stops working
exactly where the interesting physics is. Near a continuous transition the
correlation length $\xi$ grows until it is cut off by the box, and a typical
configuration is not a fine-grained mixture but a landscape of large patches,
each holding one label over a region of linear size $\xi$. A Metropolis move
changes one site, and a site in the interior of a patch is surrounded by
agreeing neighbors, so changing it costs energy and is almost always rejected.
Only sites on a patch boundary move at all, which means the configuration
evolves by boundary diffusion — the patches wander, and the time for one to be
replaced scales like the time for a random walk to cross it.

The consequence is written as a dynamic exponent. The integrated
autocorrelation time at criticality grows with the linear size as

$$\tau_{\mathrm{int}} \sim L^{z},$$

and boundary diffusion gives $z \approx 2$ for a local update, close to the
random-walk value the picture above suggests. A cluster update flips a whole
patch at once and reaches $z$ well below one — near $1/2$ in two dimensions at
small $q$. The ratio therefore grows as roughly $L^{3/2}$, which is a factor of
several hundred by $L = 64$. Prefactors differ between the two algorithms and a
cluster sweep costs more per sweep than a local one, so the realized gain is
smaller than that; the point is that the gap grows without bound as the lattice
does, and no amount of tuning a local update closes it.

That is what `examples/potts/potts.toml` means when it says raising the lattice
size collapses the reported `n_eff`. The `tests/potts_e2e.rs` critical-energy
run has the same problem and works around it by staying small, which
`docs/potts.md` explains at length.

## The bond representation

The construction rests on rewriting one bond's Boltzmann factor as a mixture.
Take the Potts energy, $H = -J \sum_{\langle ij \rangle} \delta(s_i, s_j)$, so
that the weight of a configuration factors over bonds,

$$e^{-\beta H} = \prod_{\langle ij \rangle} e^{\beta J\, \delta(s_i, s_j)}.$$

Each factor takes only two values, $e^{\beta J}$ when the labels agree and $1$
when they do not, so it can be written as a constant times an affine function of
the indicator. Setting

$$p = 1 - e^{-\beta J}$$

gives the identity the whole algorithm turns on,

$$e^{\beta J\, \delta(s_i, s_j)} = e^{\beta J}\Big[(1 - p) + p\, \delta(s_i, s_j)\Big],$$

which is checked by evaluating both sides at $\delta = 1$ and at $\delta = 0$.
The bracket is a sum of two non-negative terms, so it can be read as a sum over
a two-valued auxiliary variable rather than as a number. Give every bond a
variable $n_b \in \{0, 1\}$ — call the bond *occupied* when $n_b = 1$ — and
expand the product over bonds. Dropping the constant $e^{\beta J}$ per bond,
which cancels from every ratio,

$$
W(s, n) \;\propto\; \prod_b (1-p)^{\,1 - n_b}\,\big[p\, \delta_b\big]^{\,n_b},
$$

where $\delta_b$ is the agreement indicator of bond $b$. This joint weight is
zero unless every occupied bond joins two agreeing sites, which is the only
constraint linking the two variables.

Summing $W$ over the bond variables reassembles the product above and returns
the Potts weight, so the spin marginal of the joint model is the model we wanted.
Summing instead over the spins at fixed $n$ gives something else: the occupied
bonds partition the sites into connected clusters, every site of a cluster must
carry the same label, and each cluster is otherwise free, so

$$
\sum_s W(s, n) \;\propto\; p^{|n|}\,(1-p)^{\,N_b - |n|}\; q^{\,C(n)},
$$

with $|n|$ the number of occupied bonds, $N_b = DN$ the number of bonds, and
$C(n)$ the number of clusters. That is the random-cluster model, and the factor
$q^{C(n)}$ is the only place $q$ appears — which is why the same construction
runs at any number of states.

## Why nothing is rejected

The algorithm is the pair of conditional distributions of that joint weight,
sampled alternately. Read $W(s,n)$ as a function of $n$ at fixed $s$: the bonds
are independent, a bond whose endpoints disagree is forced empty, and a bond
whose endpoints agree is occupied with probability $p$. Read it as a function of
$s$ at fixed $n$: the weight does not depend on *which* label a cluster carries,
only that its sites agree, so each cluster takes a label drawn uniformly and
independently from all $q$. Those two steps are exactly the two halves of a
sweep, and each is a Gibbs step — a draw from an exact conditional of $W$ — so
each leaves $W$ invariant on its own. Their composition therefore leaves $W$
invariant, and the spin marginal of $W$ is the Boltzmann distribution.

A Gibbs step has no acceptance test, which is the whole answer, but it is worth
seeing the same fact as a Metropolis ratio because that is the form
`docs/metropolis.md` sets up. Suppose the bond step produced $n$ from $s$, and
the relabel step produces $s'$. Every occupied bond lies inside a cluster, and a
cluster is monochromatic in $s$ and again in $s'$, so every occupied bond joins
agreeing sites in both. The bond weights in $W$ read only the agreement
indicator, so $W(s, n) = W(s', n)$. And the probability of proposing $n$ is the
same from either configuration, since the same bonds agree in both. The forward
and reverse flows are therefore equal term by term,

$$\frac{W(s', n)\, P(n \mid s')}{W(s, n)\, P(n \mid s)} = 1,$$

so the Metropolis acceptance $\min(1, \cdot)$ is one and the move is always
taken. The uphill moves a local update accepts only with probability
$e^{-\beta \Delta E}$ are being made for free, and that is the source of the
speedup: the algorithm pays for a large energy change by having chosen the
cluster boundary to sit where the change costs nothing on average.

**SW1 — every bond is offered exactly once per sweep.** The bond variables are
independent given $s$, and there are $DN$ of them on a periodic lattice: one
forward edge per site per axis. Offering a bond twice would open it with
probability $1 - (1-p)^2$ instead of $p$, which is a different model at a
different temperature and fails nothing visibly. `cluster::site_clusters` walks
the forward neighbor columns only, the same walk `Potts::energy` takes, and
`every_bond_is_offered_exactly_once` pins it.

**SW2 — a bond is opened only between agreeing sites.** The joint weight is zero
on a configuration with an occupied bond across a disagreeing pair, so such a
bond is not merely improbable but forbidden. The CPU updater short-circuits on
the agreement test before drawing, which also fixes where the draws land in the
stream; the device kernel evaluates both and combines them, since a counter-based
draw costs nothing to skip.

## The bond gap and the two conventions

The derivation above used the Potts delta convention, where an agreeing bond
scores $-J$ and a disagreeing one scores $0$. The identity only ever saw the
*difference* between those two energies, so the general statement is in terms of
that difference rather than a coupling.

**SW3 — the bond probability is $p = 1 - e^{-\beta \Delta}$, where $\Delta$ is
the per-bond gap $E(\text{disagree}) - E(\text{agree})$.** For Potts,
$\Delta = J$. For Ising, whose bond scores $-J$ aligned and $+J$ anti-aligned,
$\Delta = 2J$. Same physics, different bookkeeping — and the factor of two is by
some distance the likeliest place for a cluster implementation to go quietly
wrong, because a run at the wrong $p$ still equilibrates, still looks healthy,
and simply reports the physics of a different temperature. It is the reason
`BondAction::bond_energy_gap` returns a gap rather than a coupling, so that the
convention is stated once by each model and never inferred by the algorithm.
This is the same factor of two the $q = 2$ correspondence in `docs/potts.md`
carries: a Potts model at coupling $2J$ is a zero-field Ising model at $J$, and
the two gaps agree exactly there, which
`the_two_conventions_give_the_same_bond_probability` pins.

**SW4 — the gap must be non-negative.** The identity requires
$0 \le p \le 1$, hence $\beta \Delta \ge 0$. A ferromagnetic coupling satisfies
it and $\beta = 0$ or $J = 0$ sits at the boundary, where $p = 0$ and the sweep
degenerates to redrawing every site independently — which is correct, since that
*is* the infinite-temperature model. An antiferromagnetic coupling does not: it
would give a negative $p$, no bond would ever open, and the chain would sample
the infinite-temperature model while reporting the coupling it was handed.
`SwendsenWang::for_model` refuses it rather than letting that happen.

## Redrawing rather than flipping

The conditional $P(s \mid n)$ draws each cluster's label uniformly from all $q$,
including whichever label the cluster already carried, and the implementation
does exactly that. The textbook presentation at $q = 2$ instead says "flip each
cluster with probability one half", and the two are the same move: drawing
uniformly from two labels keeps the current one half the time and changes it the
other half. The redraw is the form that generalizes, since at $q > 2$ there is no
flip to speak of — and it is also the form the derivation produces, because
nothing in $W$ distinguishes the label a cluster is leaving from the one it is
arriving at.

The device kernel gets the same effect without any communication by keying its
draw on the cluster's *root* — the site index the labeling converged to — rather
than on the drawing site. Every member of a cluster reads the same key, so every
member draws the same label, and the relabel stage stays one independent thread
per site.

## What the construction does not reach

**SW5 — the model's energy must be invariant under relabeling.** The step that
needs it is $P(s \mid n)$: a cluster's label is drawn uniformly because $W$ is
indifferent to which label it takes. An external field, or a per-label offset,
is precisely a term that is not indifferent. Under Potts's
$H_{\text{offset}} = -\sum_\alpha h_\alpha N_\alpha$ a cluster $C$ carrying
label $\alpha$ picks up a factor $e^{\beta h_\alpha |C|}$, so the conditional
becomes a draw weighted by cluster *size* rather than a uniform one. That is
still exact, but it is unusable: the weight is exponential in the size of the
cluster, and the percolating cluster near criticality holds a finite fraction of
the lattice, so it is pinned to the favored label with probability
indistinguishable from one and overflows a float on the way there.

The standard repair is the ghost spin. Add one extra site connected to every
real site, give those connections the field as their coupling, and the field
term becomes an ordinary bond term in a lattice with one more vertex; clusters
that end up connected to the ghost are held at its label and the rest are drawn
uniformly as before. It works, and it is not implemented here. Adding it would
mean a second lattice geometry, a bond kind the neighbor table does not
describe, and a cluster labeling over $N + 1$ vertices — a change to what a
lattice *is*, made for a term no run in this crate currently uses in anger. So
the crate refuses instead, at two places: `SwendsenWang::for_model` panics, and
each run config's `validate` reports it as a load-time error. That pairing — a
graceful message on the path a config file takes and a backstop panic for a
caller who skipped it — is the one `check_dimension` and `shape_array` already
use.

The Z2 gauge model is out of reach for a structural reason rather than a
symmetry one. Its energy is a product over plaquettes of four link variables,
and the identity in the bond representation applies to a factor that reads *two*
variables — that is what lets an occupied bond mean "these two sites are in the
same cluster" and what makes the resulting object a graph. Expanding a
four-variable factor the same way gives an occupied *plaquette* joining four
links at once, which is a hypergraph rather than a connectivity problem, and the
clusters it would produce are not what a gauge-invariant move relabels. So
`Z2Gauge` has no `BondAction` implementation, and its module docs say why so the
absence is not read as an oversight. The config layer needs no new rule for this:
both cluster kinds name `Cell::Site`, and the gauge schema already refuses
anything that colors sites.

Frustrated antiferromagnets fail differently again, and it is worth separating
the two. For a *bipartite* Ising antiferromagnet the fix is trivial — flip every
spin on one sublattice and the model is ferromagnetic — so nothing is really
wrong there beyond the sign in SW4. Frustration is when no such map exists: on a
triangular lattice, or for the antiferromagnetic Potts model at $q > 2$, the
bond weights cannot all be made non-negative by any relabeling, the expansion
produces terms of both signs, and there is no probability to sample. That is a
sign problem, not a missing feature, and it is why SW4 refuses rather than
compensating.

## What a sweep is, and what that costs the statistics

One Swendsen–Wang sweep is one full pass: place every bond, label every cluster,
redraw every cluster's label. It touches every site exactly once, which is what
makes it the same *unit* of Monte Carlo time a Metropolis sweep is, but the two
are not the same *move*. A Metropolis sweep attempts $N$ independent local
changes; a cluster sweep makes one global one.

**SW6 — an autocorrelation time in sweeps is not comparable between the two
algorithms.** `reduce` in `src/statistics.rs` reports $\tau_{\mathrm{int}}$ in
units of whatever a sample is, and a sample is `sweeps_between` sweeps, so the
number it returns for a cluster run and for a Metropolis run are counts of
different things done at different cost. A cluster sweep is $\Theta(DN)$ like a
local sweep, but with a larger constant: the labeling is a union-find pass over
every bond, and `Updater::sweep`'s contract to return the realized $\Delta E$
costs two more full energy scans, since a cluster move does not price itself one
site at a time. What is comparable across algorithms is the cost of one
independent sample — wall-clock per sweep times $2\tau_{\mathrm{int}}$ — and that
is the quantity a comparison should quote. The end-to-end test asserting the
speedup measures $\tau_{\mathrm{int}}$ in sweeps and says so, which is a fair
statement about the *dynamics* and deliberately not a benchmark.

The two extra energy scans are worth defending rather than removing. They exist
because `Updater::sweep` returns a net realized $\Delta E$, and every other
updater accumulates that as it goes. Widening `BondAction` so a cluster move
could accumulate its own would put an incremental-energy method on a trait whose
whole purpose is to describe a *static* property of the model, to save a constant
factor on a pass that is already linear in the volume. Keeping the seam uniform
is worth more.

## On the device

The GPU backend runs the same four steps and differs only in how the second and
third are done. Placing bonds is one thread per bond, and relabeling is one
thread per site keyed on the cluster root, so both are embarrassingly parallel.
Labeling the connected components is neither.

The labeling is done by iterated propagation with pointer jumping. Every site
starts holding its own index; each pass lowers a site's value to the smallest it
can see among itself, its bonded neighbors, and one step past whichever of those
won. Every value a site can take is a site index inside its own cluster, so no
pass can merge clusters that are not connected, and the values only decrease, so
the iteration terminates. At the fixed point every bonded pair holds the same
value, hence the value is constant across a cluster and — being itself a site of
that cluster — equal to its own entry. The pointer jump is what makes the number
of passes logarithmic in the lattice size rather than proportional to a cluster's
diameter, which on a percolating cluster would be the whole box.

**SW7 — the labeling must reach its fixed point before the relabel step runs.**
Relabeling a partial labeling would split one cluster across two draws, which
breaks $P(s \mid n)$ and samples the wrong distribution without failing. The
convergence test cannot live inside a dispatch: a compute dispatch has no global
barrier, so a workgroup cannot learn whether another workgroup moved a value, and
convergence is only observable *between* dispatches. That is why the host is in
the loop — it runs a batch of passes, reads a one-word flag back, and dispatches
again if anything moved. A pass reporting no change means the state it started
from was already the fixed point: only one thread writes a given site, so a pass
cannot miss a move it should have made itself, and a move another thread has yet
to make is caught by the next pass. Running past convergence is harmless, which
is what lets several passes share one round-trip. If a run exceeds a pass cap
derived from the lattice size, the chain panics rather than proceeding on a
partial labeling.

There is no even-extent requirement here, and that is the one place a reader is
likely to expect this backend to behave like the checkerboard one and be wrong.
The even extents in `docs/metropolis.md` protect a *coloring*: an odd extent
wraps a site onto a same-color neighbor and a parallel color pass stops being
independent. A cluster update has no coloring — it labels a graph, and the graph
is whatever the bonds make of it — so a device cluster run is correct on any
shape, which `a_cluster_updater_accepts_odd_extents_even_on_the_gpu` states and
`runs_on_odd_extents` exercises.

The host and device paths consume randomness differently: the host draws from a
stream in bond order, and the device keys a counter on `(seed, bond, sweep)`.
The two are therefore equal in distribution and not bit-for-bit, exactly as the
checkerboard backends already are with their CPU reference, and the tests
comparing them are distributional.

## What is not built

Wolff's single-cluster variant grows one cluster from a random seed site and
relabels only that, which samples the same joint distribution with a different
choice of what to update per move and is usually somewhat faster. It needs
exactly the data `BondAction` already supplies and exactly the lattice walk
`cluster::site_clusters` already takes, so nothing here should have to change
when it arrives — but nothing here is built for it either.

Two generalizations of the bond step are left out deliberately, and both are
noted where they would go. Per-bond couplings would make the gap a function of
the bond rather than a scalar, which is what a disordered model needs; and an
agreement test other than label equality — an angle threshold, or a reflection —
is what a clock or O($n$) model needs. Each is a real extension of this
construction and each is a guess until a model in this crate asks for it.

Percolation observables are the other absence. The mean cluster size and the
wrapping probability are functions of the partition alone, which is why
`cluster::site_clusters` returns a `SiteClusters` carrying sizes rather than
writing labels back into a configuration, and why it takes a bare predicate
instead of a model. They are not implemented; the module is shaped so that they
could be.

## References

- R. H. Swendsen and J.-S. Wang, *Nonuniversal critical dynamics in Monte Carlo
  simulations*, Phys. Rev. Lett. **58**, 86 (1987) — the algorithm.
- C. M. Fortuin and P. W. Kasteleyn, *On the random-cluster model I*, Physica
  **57**, 536 (1972) — the bond representation the algorithm samples.
- U. Wolff, *Collective Monte Carlo updating for spin systems*, Phys. Rev. Lett.
  **62**, 361 (1989) — the single-cluster variant.
- Y. Komura and Y. Okabe, *GPU-based Swendsen–Wang multi-cluster algorithm for
  the simulation of two-dimensional classical spin systems*, Comput. Phys.
  Commun. **183**, 1155 (2012) — the device construction, including the
  label-propagation scheme.
