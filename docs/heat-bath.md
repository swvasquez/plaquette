# The heat bath algorithm

The heat bath update is the second single-variable kernel, beside the
Metropolis move derived in `docs/metropolis.md`. This file states the update as
the definite procedure the code implements — `heat_bath_step` in
`src/updater.rs` and the `heat_bath.wgsl` kernel fragment under `src/wgsl/` — with the
single-variable step and the parallel checkerboard sweep both given in full,
and comparisons, cost accounting, and limits gathered at the end as context.

## Algorithm

A heat bath step freezes every variable but one and redraws that variable
directly from its conditional Boltzmann distribution given the frozen rest.
There is no proposal and no rejection; the update lands somewhere every time.

**HB1 — the conditional weights are exponentials of `energy_delta`.**
Everything in the energy that does not touch the chosen variable cancels from
the conditional, so measuring each candidate state $s$ against the current
configuration leaves

$$
P(s \mid \text{rest}) = \frac{e^{-\beta\,\Delta E_s}}{\sum_{s'=0}^{Q-1} e^{-\beta\,\Delta E_{s'}}},
$$

where $\Delta E_s$ is the energy change of poking the variable to $s$ — exactly
what `Action::energy_delta` returns — and the current state enters at
$\Delta E = 0$, weight one. The kernel therefore needs nothing from a model
beyond the `Action` seam in `src/action.rs`, and it is grade-neutral in the
same way `energy_delta` is: sites for Ising and Potts, links for Z2 gauge.

Concretely, one step at a variable is:

1. Evaluate $\Delta E_s$ for each of the $Q$ candidate states via
   `Action::energy_delta(lattice, config, var, s)`. The current state's entry
   is zero by construction and is written down rather than computed, so this is
   $Q - 1$ calls.
2. Subtract the minimum $\Delta E$ from every entry and exponentiate:
   $w_s = e^{-\beta(\Delta E_s - \min_{s'} \Delta E_{s'})}$. The shift is part
   of the correct procedure, not a cosmetic one (HB2).
3. Draw one uniform, scale it by $\sum_s w_s$, and walk the cumulative sum of
   the weights until it exceeds the draw; the variable becomes that state.
   Walking the unnormalized sum is CDF inversion with the division folded into
   the draw, and the last state serves as the structural fallback — it takes
   every draw the earlier states do not claim — so rounding in the total cannot
   leave the walk without an answer.

The step consumes exactly one uniform per variable regardless of the outcome,
so the draw count is configuration-independent.

**HB2 — offset the exponents by the minimum $\Delta E$ before
exponentiating.** A strongly downhill candidate at large $\beta$ makes
$e^{-\beta\,\Delta E}$ overflow while the current state sits at weight one.
Subtracting the minimum shifts the largest exponent to zero and cancels in the
normalization, so it changes nothing mathematically and everything numerically.
(At $Q = 2$ the two-state ratio could be rearranged so both limits saturate
safely, but every kernel here — CPU and GPU alike — subtracts the minimum,
so the two-state models run the same arithmetic the general case does.)

## Schedules

The kernel does not care in what order variables are visited: any sequential
order is valid on any lattice, and it composes with the same schedules
`docs/metropolis.md` derives. That composition is literal in the code: a
`LocalUpdate` in `src/updater.rs` pairs `Kernel::HeatBath` with either
schedule, so the heat bath under the random schedule differs from Metropolis
in the kernel and in nothing else, and the checkerboard schedule is how the
GPU runs it — a config naming `updater = "heat_bath"` with
`schedule = "checkerboard"` and `backend = "gpu"` selects the coloring the
Metropolis chains already use with the heat bath kernel in its place.

**HB3 — a parallel pass may only update mutually non-interacting variables.**
The conditional in HB1 is defined against a neighborhood that is actually
frozen. Two interacting variables updated together would each condition on a
state the other is changing, so a parallel pass must be restricted to a set of
variables no two of which interact. The checkerboard sweep below is the
schedule that delivers this.

## Checkerboard

The coloring is the one `docs/metropolis.md` derives for the Metropolis
kernel, restated here so the parallel heat bath sweep can be followed end to
end. On a site field a site's color is the parity of its coordinate sum,
$\left(\sum_\mu x_\mu\right) \bmod 2$, giving two colors; a step along any
axis flips the parity, so every neighbor of a site carries the other color.
On a link field parity alone is not enough, because two of a plaquette's four
links share a base site. The color there is the pair (direction, base-site
parity), giving $2D$ colors: fixing a direction freezes every link of the
other directions, and the two same-direction links of a plaquette sit on
bases one step apart, which carry opposite parity. Either way, within one
color no two variables interact, so everything a variable's update reads
belongs to another color and is untouched for the whole pass. That is exactly
the frozen neighborhood HB1's conditional is defined against, which is how
the coloring meets HB3 — and why the coloring the Metropolis checkerboard
needs is the coloring the heat bath needs. The requirement comes from the
schedule, not from the kernel.

One parallel sweep is then:

1. Take the colors in a fixed order — color 0 then color 1 on a site field;
   each direction in turn, even bases then odd, on a link field.
2. For each color, run one pass with one thread per site. On a link field a
   thread owns the one link its site has in the pass's direction. A thread
   whose site does not carry the pass's color returns at once.
3. Each remaining thread runs the single-variable step of the algorithm
   above, unchanged: it reads its frozen neighborhood — neighbor labels on a
   site field, the link's staples on a link field — prices the $Q$
   candidates, draws its one uniform, and writes the outcome. There is no
   accept-or-reject path, so every thread does the same work; that absence is
   the one real difference from the Metropolis kernel under this coloring.
4. Finish the pass before starting the next, so each pass reads the writes of
   the passes before it. All colors once — two passes on a site field, $2D$
   on a link field — make one sweep, one update per variable.

Two requirements bind this sweep and neither binds a sequential one.

**HB4 — every extent must be even.** Under periodic boundaries an odd extent
wraps two same-color interacting variables next to each other, and the
neighborhood a thread conditions on is then not frozen: HB3 fails without
anything failing visibly. Sequential order carries no such requirement, which
is why the CPU random-variable schedule runs on any lattice.

**HB5 — the pass randomness is counter-based, keyed on
`(seed, variable, sweep)`.** Each thread's uniform comes from `keyed_uniform`
in `src/wgsl/rng.wgsl` rather than from a stream, so a sweep's
result does not depend on the order threads happen to run in. Every pass of a
sweep shares the one sweep counter, which is safe because the passes' variable
sets are disjoint — no key is ever used twice. The heat bath makes the budget
exact: one uniform per variable per sweep, unconditionally, so the keying is
spent completely and identically whatever the configuration.

## Comparisons

The step satisfies detailed balance with respect to the conditional. It is
also the Metropolis–Hastings special case where the proposal is the target
conditional itself, so the acceptance probability is identically one — which
is why the acceptance rate stops being a meaningful diagnostic here. The
statistics literature calls the same move Gibbs sampling.

Against Metropolis the trade is arithmetic. A heat bath step prices $Q - 1$
candidates where a Metropolis step prices one, and the draw can return the
value the variable already had, where the Metropolis proposal in this crate is
a forced change to one of the other $Q - 1$ states. At $Q = 2$ the cost gap
disappears and the heat bath is the better default; the update is then the
classic Glauber dynamics. At large $Q$ the accounting can go the other way.

Against Swendsen–Wang the trade is applicability. Swendsen–Wang needs
relabeling invariance and a pairwise bond graph, and so refuses Ising with a
field, Potts with per-label offsets, and Z2 gauge entirely. The heat bath
needs no symmetry of the model at all: a field or offset term simply enters
$\Delta E_s$ and tilts the conditional, which is why every model in the crate
qualifies.

## Limits

The general limitation is easy to state backwards, so state it carefully. The
obstruction to a heat bath is sampling, not computing. Metropolis needs only
ratios of weights, and a ratio is always available. The heat bath needs an
actual draw from the normalized conditional, which requires a constructive
sampling method for that distribution. For a discrete $Q$-state variable this
is trivial — normalizing $Q$ numbers and inverting the CDF is all it takes,
and that is why every model here qualifies. For continuous link groups no
general recipe exists: exact draws need a construction built for the group,
like the SU(2) heat baths of Creutz and of Kennedy and Pendleton, and where
none is known Metropolis is the fallback.
