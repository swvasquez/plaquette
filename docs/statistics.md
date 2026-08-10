# Estimates and error bars

A Monte Carlo run yields a series of measurements, and the point of this layer is
to turn that series into a number with an honest uncertainty attached. Everything
here takes a slice of scalars the consumer has already collected and never looks
at a configuration, which is the division of labour between it and the
observables layer: one measures, the other reduces.

The whole difficulty is that the samples are not independent. A chain moves by
local updates, so consecutive configurations differ hardly at all, and a series of
$N$ of them carries far less information than $N$ independent draws would. The
textbook standard error $\sigma/\sqrt{N}$ is therefore wrong — too small, often by
a large factor — and reporting it would produce error bars that shrink with run
length while the mean itself wanders outside them.

## Autocorrelation and the primary reduction

The correction is expressed through the integrated autocorrelation time. Writing
$\rho(t)$ for the normalised autocorrelation of the series at lag $t$,

$$\tau_{\mathrm{int}} = \frac{1}{2} + \sum_{t \ge 1} \rho(t),$$

the variance of the sample mean is inflated by a factor $2\tau_{\mathrm{int}}$
relative to the independent case, so the standard error becomes

$$\mathrm{stderr} = \sqrt{\frac{2\,\tau_{\mathrm{int}}\, C(0)}{N}},
\qquad
N_{\mathrm{eff}} = \frac{N}{2\,\tau_{\mathrm{int}}}.$$

An independent series has $\tau_{\mathrm{int}} = 1/2$, recovering the textbook
formula, and $\tau_{\mathrm{int}}$ is clamped below at that value throughout,
since a smaller number would claim the samples are anti-correlated in a way local
updates do not produce.

The sum cannot be taken to $t = N$. At large lag $\rho(t)$ is pure noise, and
summing that noise adds variance without adding signal, so the estimate has to be
truncated — but truncating too early biases $\tau_{\mathrm{int}}$ downward and
understates the error. The resolution used here is Sokal's automatic windowing:
accumulate $\tau(W) = 1/2 + \sum_{t=1}^{W} \rho(t)$ and stop at the smallest
$W \ge 1$ satisfying $W \ge c\,\tau(W)$, with $c = 5$. The window is thus chosen
by the data rather than fixed in advance, sitting far enough out to capture the
tail and close enough in to exclude the noise. The autocovariance is normalised by
$N$ rather than $N - t$, the biased convention the windowing criterion assumes.

## When an error bar cannot be believed

Windowing has a failure mode that is silent rather than loud. If the run is short
compared with its own correlation time, the criterion $W \ge c\,\tau(W)$ is met
almost immediately and $\tau_{\mathrm{int}}$ saturates near $N/5$ — not because
the chain decorrelates that fast, but because a run of that length cannot perceive
anything slower. The reported $\tau_{\mathrm{int}}$ is then a ceiling rather than
a measurement, and every error derived from it is too optimistic with nothing in
the output to say so.

This is why the diagnostics travel attached to every mean instead of being
available on request. The usual lattice requirement that a run be at least fifty
correlation times long is exactly $N_{\mathrm{eff}} > 25$, which is the threshold
`MIN_EFFECTIVE_SAMPLES` encodes and `is_reliable` tests. Falling below it is not a
signal to choose a different estimator; it means the run was too short or the
updater too slow, and the fix is more sweeps or a better algorithm.

## Derived quantities

A fluctuation such as the specific heat is not the average of anything measured
per configuration — it is a function of two moments of the energy series. That
distinction matters twice. An autocorrelation time is meaningless for it, since
there is no time series of the quantity itself to correlate, which is why derived
results carry a block count instead of $\tau_{\mathrm{int}}$ and
$N_{\mathrm{eff}}$. And its error cannot be averaged; it has to be propagated
through the function, which is what the jackknife does.

The estimator is a blocked jackknife. The series is cut into $n_b$ contiguous
blocks of length $\ell$, each block is left out in turn, the quantity is recomputed
from the remaining data, and the spread of those recomputed values gives the error:

$$\mathrm{stderr} = \sqrt{\frac{n_b - 1}{n_b} \sum_b
\left(\theta_{(b)} - \bar\theta\right)^2}.$$

Blocking is what handles the correlation: within a block the samples are
correlated, but blocks longer than the correlation time are nearly independent of
each other, so the scatter across them is a fair estimate of the uncertainty.
Every estimator here is a function of additive moments, so leaving a block out is
a subtraction from a precomputed total rather than a rescan — one pass per column,
then constant work per block. A genuinely non-additive estimator would need the
slice-based form back.

The block length is the one physics decision the caller makes. A combination of
moments decorrelates only as fast as its slowest ingredient, so the length must
exceed the correlation time of *every* series entering the estimator; taking the
maximum $\tau_{\mathrm{int}}$ across the columns and blocking by
$\lceil 2\tau \rceil$ is the rule used here. Erring long merely widens the error
bar, while erring short understates it — the dangerous direction, and the reason
the maximum is taken rather than an average. A tighter alternative exists, the
linearised Gamma method, which projects each sample onto the estimator's gradient
and can justify shorter blocks; it is not implemented.

Fewer than two whole blocks leaves nothing to measure a scatter from, and that
case returns `NaN` rather than zero. Zero would read as perfect certainty, which
is the opposite of what a two-sample run means.

## Numerical care

Two of the estimators are variances, and a variance computed as
$\langle x^2 \rangle - \langle x \rangle^2$ differences two large and nearly equal
numbers. The features are therefore centred by the global mean before the jackknife
runs, so that every leave-one-out first moment sits near zero. The estimator's
value is unchanged by the shift — a variance does not care where the origin is —
but the cancellation disappears.

One consequence is worth stating because it looks like an inconsistency. Blocks are
sized from the raw series, while the jackknife runs on the centred features. That
is deliberate: the correlation time of the raw integrand is what the block length
must clear, and the centred square has a different one, so folding the two together
would understate the error without failing any obvious test.

## The estimators

The specific heat and the susceptibility are variances of the energy and the order
parameter respectively, scaled by $\beta$ and the lattice volume in the
conventional way. The Binder cumulant is a ratio of the fourth moment to the square
of the second, and its normalising constant depends on how many components the
order parameter has — the familiar $1 - \langle m^4\rangle / 3\langle m^2\rangle^2$
is the scalar convention, and a Potts order parameter needs a different one.

The Creutz ratio is the exception that motivates keeping the jackknife general. It
is not a fluctuation at all but a nonlinear combination of four separate ensemble
averages of Wilson loops, taken so that perimeter and constant terms cancel and the
string tension survives. Its error has to travel through that logarithm and those
four means, which is precisely what a jackknife over the joint series does and what
no per-quantity error propagation would.
