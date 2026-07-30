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

where $H(C)$ is the energy of a configuration $C$ and $Z$ normalises the weights.
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
