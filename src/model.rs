//! Model: the value *semantics* the lower layers withhold, and everything that
//! reads them — the energy, and the config-derived observables.
//!
//! Everything below this module is meaning-free: a [`State<Q>`] is an index, a
//! [`Configuration`] a flat array of indices, a [`Lattice`] geometry alone. The
//! model is the first place that says what the indices *are* — that `{0, 1}`
//! map to the values `{+1, −1}`. That is the private `decode` map, read by the
//! energies and by observables like [`Ising::magnetization`], and nothing
//! outside this module sees a `±1`. [`Potts`] says the opposite about its own
//! indices: they are labels standing for nothing, compared for equality and
//! never decoded, which is what leaves it free to be generic over `Q` while the
//! other two are pinned at the two states `decode` knows.
//!
//! [`Action`] is the energy seam the updater depends on: energies only, never
//! variable values. It owns just the physics parameters, borrowing the lattice
//! and configuration per call, and keeps no running energy. Value-semantic
//! observables stay inherent methods on the concrete model rather than trait
//! methods, so the trait stays energy-only. The three models differ in where
//! their variables sit and in what one term of the energy scores: [`Ising`]
//! scores bonds between site variables as a product of signs, [`Z2Gauge`] scores
//! plaquettes built from link variables, and [`Potts`] scores bonds again but by
//! whether the two labels agree. Which cell a model expects is part of its
//! contract rather than something it can discover, so each energy entry point
//! checks it with a `debug_assert!` — a field on the wrong cell otherwise panics
//! deep in a loop or, when the two counts happen to coincide, silently computes
//! a different theory.
//!
//! All three hold in any dimension their cell exists in, and none carries a
//! dimension-shaped special case: `Ising` and `Potts` need one direction,
//! `Z2Gauge` needs two.

use crate::configuration::{Cell, Configuration};
use crate::lattice::{Lattice, Loop, Sign};
use crate::state::State;

/// The energy functional the sampler is built around.
///
/// Generic over the state count `Q` and the lattice dimension `D`, so the
/// updater can name the seam without naming a specific model.
pub trait Action<const Q: usize, const D: usize> {
    /// The energy `H` of `config` on `lattice`, computed from scratch — a full
    /// lattice scan, not the hot path.
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<Q>) -> f64;

    /// The energy change `ΔE = H(after) − H(before)` of poking the variable at
    /// `var` to `proposed`, without mutating `config`.
    ///
    /// The index names a cell of whatever kind `config` sits on, which is the
    /// model's own business — a site for [`Ising`], a link for [`Z2Gauge`] — so
    /// the parameter stays grade-neutral rather than promising a site.
    ///
    /// The sampler's hot path: it reads only the terms incident to `var`, so it
    /// is `O(1)` in the lattice size rather than a rescan. It equals
    /// `energy(after) − energy(before)` by construction — exactly when the
    /// couplings and sums are integer-valued, up to rounding otherwise — and is
    /// the more accurate side of that comparison.
    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
        var: usize,
        proposed: State<Q>,
    ) -> f64;
}

/// The Ising model: spins `s_i = ±1` with nearest-neighbor coupling `j` and a
/// uniform external field `h`,
///
/// ```text
/// H = -j * sum_<ij> s_i s_j  -  h * sum_i s_i
/// ```
///
/// where the first sum runs over each nearest-neighbor bond once. `j > 0` is
/// ferromagnetic (aligned neighbors lower the energy). Energies come out in the
/// same units as `j` and `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ising {
    /// Nearest-neighbor coupling `J`.
    j: f64,
    /// Uniform external field `h` (set to `0.0` for the field-free model).
    h: f64,
}

impl Ising {
    /// The fewest dimensions this model is defined in.
    ///
    /// The energy scores nearest-neighbor bonds, and a line has them — a ring of
    /// spins is a perfectly good, and exactly solvable, Ising model. The peer of
    /// [`Z2Gauge::MIN_DIMENSION`], which is two for a reason that bites much
    /// harder; this one exists so both models answer the same question in the
    /// same place.
    pub const MIN_DIMENSION: usize = 1;

    /// An Ising action with coupling `j` and external field `h`.
    pub fn new(j: f64, h: f64) -> Self {
        Ising { j, h }
    }

    /// The total magnetization `M = sum_i s_i` of `config` — the raw *signed*
    /// spin sum, not `|M|` and not a density.
    ///
    /// Keeping the sign is what makes both `<m²>` and `<|m|>` recoverable from
    /// the series downstream. It reads the private `decode` map, so it is an
    /// inherent method rather than part of the energy-only [`Action`] trait.
    pub fn magnetization(&self, config: &Configuration<2>) -> f64 {
        // Integer accumulator: the sum is exact until the final cast.
        let spin_sum: i64 = config.variables().iter().map(|&s| decode(s) as i64).sum();
        spin_sum as f64
    }

    /// The per-config two-point correlator `C_r = (1/N) Σ_i s_i · s_{i+r}`,
    /// measured along each lattice axis under periodic boundaries.
    ///
    /// Returns one row per axis: entry `μ`, index `r` is `C_r` for displacement
    /// `r = 0..=L_μ/2` (length `L_μ/2 + 1`). Only that non-redundant half is
    /// stored, since `C_r = C_{L_μ − r}` by translation invariance, and index `0`
    /// keeps the `C_0 = 1` anchor.
    ///
    /// This is the *raw per-config estimator* only. The connected subtraction
    /// (`− ⟨s⟩²`), the ensemble average, and the correlation-length fit are
    /// reductions over a chain of these arrays and belong to statistics.
    pub fn correlator<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
    ) -> [Vec<f64>; D] {
        let sums = axis_pair_sums(lattice, config, |s_i, s_j| {
            (decode(s_i) * decode(s_j)) as i64
        });
        let n = config.n_vars() as f64;
        std::array::from_fn(|mu| sums[mu].iter().map(|&c| c as f64 / n).collect())
    }
}

/// Sum `pair(s_i, s_{i+r})` over every site, for each axis and each displacement
/// `r = 0..=L_mu/2` — the scan both models' correlators are built from.
///
/// The two differ only in what one pair contributes: a product of signs for
/// [`Ising`], an equality test for [`Potts`]. Everything else is shared and easy
/// to get subtly wrong twice — which half of each axis is stored, that
/// `C_r = C_{L_mu - r}` is why the other half can be dropped, and that the
/// accumulation stays in an integer so the final scaling is the only rounding.
/// Stating it once leaves each model with its one-line kernel.
///
/// The counts are returned raw. What to divide by, and whether to subtract a
/// floor afterwards, is the caller's, since the two conventions differ there.
fn axis_pair_sums<const Q: usize, const D: usize>(
    lattice: &Lattice<D>,
    config: &Configuration<Q>,
    pair: impl Fn(State<Q>, State<Q>) -> i64,
) -> [Vec<i64>; D] {
    let shape = lattice.shape();
    let mut sums: [Vec<i64>; D] = std::array::from_fn(|mu| vec![0i64; shape[mu] / 2 + 1]);
    for site in 0..config.n_vars() {
        let s_i = config.peek(site);
        for (mu, row) in sums.iter_mut().enumerate() {
            for (r, cell) in row.iter_mut().enumerate() {
                *cell += pair(s_i, config.peek(lattice.site_shift(site, mu, r)));
            }
        }
    }
    sums
}

/// Map a two-state index to the value it stands for: `0 → +1`, `1 → −1`. The
/// whole of the two-state value semantics, kept private to this module.
///
/// It is named for the operation rather than for either model's variable,
/// because both read it and not by coincidence: an Ising spin and a Z2 gauge
/// link are the same variable, an element of the group `{+1, −1}` under
/// multiplication, moved from the sites to the links. One function records that
/// where a copy per model would obscure it.
// TODO: value semantics belong to a type, not a function — the decode is fixed
// by `Q`, not by the model reading it, which is why one function serves both.
// The prior art (Grid, QDP++) carries it as a per-cell value type owning its own
// product. Build that when a second value appears that is not `±1` — a Potts
// index, a `U(1)` phase, an `SU(N)` matrix — since one example cannot decide its
// shape. Until then this stays private to the module so the move costs one file.
fn decode(state: State<2>) -> i32 {
    1 - 2 * state.index() as i32
}

impl<const D: usize> Action<2, D> for Ising {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        debug_assert!(config.cell() == Cell::Site, "Ising spins live on sites");

        // Integer accumulators: the sums are exact until the final scaling.
        let mut bond_sum: i64 = 0; // sum over each bond once of s_i s_j
        let mut spin_sum: i64 = 0; // sum_i s_i, for the field term

        for site in 0..config.n_vars() {
            let s_i = decode(config.peek(site));
            spin_sum += s_i as i64;

            // The neighbor row is ordered +0, −0, +1, −1, ...; taking the
            // forward columns only (every other entry) visits each bond once.
            for &j_site in lattice.site_neighbors(site).iter().step_by(2) {
                bond_sum += (s_i * decode(config.peek(j_site))) as i64;
            }
        }

        -self.j * bond_sum as f64 - self.h * spin_sum as f64
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        var: usize,
        proposed: State<2>,
    ) -> f64 {
        debug_assert!(config.cell() == Cell::Site, "Ising spins live on sites");

        // ΔE = -(s'_i - s_i) * (J * sum_{j in nbrs(i)} s_j + h). Only bonds
        // touching `var` and its field term change; everything else cancels.
        let ds = (decode(proposed) - decode(config.peek(var))) as i64;
        if ds == 0 {
            return 0.0; // proposed state equals the current one
        }

        // All 2D neighbors (both directions): every incident bond changes.
        let neighbor_sum: i64 = lattice
            .site_neighbors(var)
            .iter()
            .map(|&j_site| decode(config.peek(j_site)) as i64)
            .sum();

        -(ds as f64) * (self.j * neighbor_sum as f64 + self.h)
    }
}

/// The Z2 lattice gauge theory: variables `σ_ℓ = ±1` on the links, coupled
/// through the product of the four around each elementary square,
///
/// ```text
/// H = -j * sum_□ prod_{l in d□} σ_l
/// ```
///
/// where the sum runs over each plaquette once. `j > 0` favors plaquette
/// products of `+1`, the analogue of the ferromagnetic alignment that lowers
/// the Ising energy. Energies come out in the same units as `j`.
///
/// This is the Wilson action specialised to `Z2`, and the only structural
/// change from [`Ising`] is the unit of interaction: a four-link product around
/// a face in place of a two-spin product across a bond. A plaquette needs two
/// directions to span, so the model is empty below `D = 2` — the lattice
/// reports no plaquettes there and every configuration would score zero, which
/// the entry points reject rather than sample.
///
/// There is deliberately no external-field term to match [`Ising`]'s `h`. The
/// plaquette energy is invariant under flipping every link that touches a
/// chosen site, since each of the site's links appears in a given plaquette
/// exactly twice and the two flips cancel; a term `-h * sum_l σ_l` reading
/// individual links would notice that flip and destroy the local symmetry that
/// makes this a gauge theory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Z2Gauge {
    /// Plaquette coupling `J`.
    j: f64,
}

impl Z2Gauge {
    /// The fewest dimensions this model is defined in.
    ///
    /// The energy scores plaquettes, and a plaquette is spanned by a *pair* of
    /// distinct directions, so one dimension has none. That is not a small
    /// lattice but no theory at all: `C(D, 2)` is zero, every configuration
    /// scores zero, every proposed flip is accepted, and a chain returns noise
    /// wearing the shape of a gauge field. Nothing downstream notices, which is
    /// why the floor is stated rather than left to the arithmetic.
    ///
    /// It lives here because it follows from what this action scores. The config
    /// schema re-exports it as
    /// [`MIN_DIMENSION`](crate::gauge_config::MIN_DIMENSION) and
    /// `GpuGaugeChain::new` reads it too, so the rule has one home and one
    /// wording rather than a literal `2` in each.
    pub const MIN_DIMENSION: usize = 2;

    /// What every guard on [`MIN_DIMENSION`](Z2Gauge::MIN_DIMENSION) says, so
    /// the five places that check it cannot drift into five wordings.
    pub(crate) const TOO_FEW_DIMENSIONS: &'static str =
        "the Z2 gauge action scores plaquettes, which need at least two dimensions";

    /// A Z2 gauge action with plaquette coupling `j`.
    pub fn new(j: f64) -> Self {
        Z2Gauge { j }
    }

    /// The signed plaquette sum `sum_□ σ_□` of `config` — the same scan the
    /// energy is built from, before the coupling is applied.
    ///
    /// The energy is `-j` times this, so the two carry the same information
    /// whenever `j` is nonzero, and this is the side that survives `j = 0` and
    /// the side the literature reports: divided by the plaquette count it is the
    /// mean plaquette, which in two dimensions is exactly `tanh(β)`.
    pub fn plaquette_sum<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
    ) -> f64 {
        assert!(D >= Self::MIN_DIMENSION, "{}", Self::TOO_FEW_DIMENSIONS);
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        // Integer accumulator: the sum is exact until the final cast. Each
        // plaquette is named once by the lattice, so there is no double counting
        // to undo the way the Ising bond sum needs.
        let mut sum: i64 = 0;
        for plaquette in 0..lattice.n_plaquettes() {
            let product: i32 = lattice
                .plaquette_links(plaquette)
                .iter()
                .map(|&link| decode(config.peek(link)))
                .product();
            sum += product as i64;
        }
        sum as f64
    }

    /// The product of the link variables around `path`, walked from `base`:
    /// `±1`, and the only kind of number this model can measure.
    ///
    /// Every observable of a gauge theory is built from this. A single link
    /// variable is not measurable — flipping every link touching one site is a
    /// symmetry of the energy, so any quantity that notices such a flip averages
    /// to zero however long the chain runs — but the product around a closed
    /// path is untouched by it, because the path enters and leaves each site it
    /// visits and so picks up that site's flip exactly twice.
    ///
    /// [`Loop`] having already refused an open path is what makes that argument
    /// hold here: a path with two loose ends picks up its endpoints' flips once
    /// each, and its product would be a gauge-dependent number that looks like a
    /// measurement.
    pub fn loop_product<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        base: usize,
        path: &Loop<D>,
    ) -> f64 {
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        lattice
            .loop_links(base, path)
            .map(|link| decode(config.peek(link)))
            .product::<i32>() as f64
    }

    /// The per-config Polyakov loop along `dir`: the product of the link
    /// variables down a line wrapping that direction, averaged over the lines,
    /// and kept *signed*.
    ///
    /// A line wrapping the torus is closed, so its product is invariant like any
    /// other loop's, but it cannot be deformed to a point the way a rectangle
    /// can, and that is what makes it the order parameter for deconfinement.
    /// Multiplying every link along `dir` in one slice by `-1` leaves the energy
    /// alone — each plaquette crosses that slice with two of its links, or none
    /// — while flipping every one of these products, so the average vanishes in
    /// the confined phase by that symmetry alone and is nonzero only when it
    /// breaks.
    ///
    /// The sign is kept for the same reason [`Ising::magnetization`] keeps it:
    /// the symmetry drives the signed average to zero on a finite lattice, and
    /// keeping it lets statistics recover both the mean square and the mean
    /// magnitude from the series.
    ///
    /// Only one line per position across `dir` is walked, since starting from a
    /// different point along the line traces the same links.
    pub fn polyakov_loop<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        dir: usize,
    ) -> f64 {
        debug_assert!(dir < D, "direction out of range");
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        let wrap: Vec<_> = std::iter::repeat_n((dir, Sign::Plus), lattice.shape()[dir]).collect();
        let path = Loop::new(lattice, &wrap).expect("a full wrap closes");

        // Sums of `±1` stay exact in `f64`, as in `wilson_rectangles`.
        let mut total = 0.0;
        let mut lines = 0usize;
        for site in (0..lattice.n_sites()).filter(|&s| lattice.site_coords(s)[dir] == 0) {
            total += self.loop_product(lattice, config, site, &path);
            lines += 1;
        }
        total / lines as f64
    }

    /// The per-config Wilson loop table: `table[r][t]` is the average of
    /// [`loop_product`](Z2Gauge::loop_product) over every `r`-by-`t` rectangle
    /// on the lattice.
    ///
    /// One rectangle's product is `±1` and says nothing on its own, so what is
    /// measured is the average over a whole symmetry class: the same rectangle
    /// walked from every site, in every plane, and with the two side lengths
    /// assigned to the plane's directions both ways round. Every member of that
    /// class has the same expectation, so averaging them estimates it with far
    /// less noise, and the table comes out symmetric in `r` and `t` by
    /// construction.
    ///
    /// Sides run from `0` to `max_side`, capped at half the smallest extent
    /// because a rectangle wider than that wraps far enough to see itself around
    /// the torus. Row and column `0` are the `1.0` anchor a zero-width rectangle
    /// gives, matching how [`correlator`](Ising::correlator) keeps `C_0 = 1`.
    ///
    /// This is the raw per-config estimator only. The chain average, the area
    /// against perimeter comparison, and the string-tension fit are reductions
    /// over a series of these tables and belong to statistics.
    pub fn wilson_rectangles<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        max_side: usize,
    ) -> Vec<Vec<f64>> {
        assert!(D >= Self::MIN_DIMENSION, "{}", Self::TOO_FEW_DIMENSIONS);
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        let shortest = lattice.shape().iter().copied().min().unwrap_or(0);
        let max = max_side.min(shortest / 2);

        let mut table = vec![vec![1.0; max + 1]; max + 1];
        // Row and column zero keep their `1.0` anchor, so both loops skip them.
        for (r, row) in table.iter_mut().enumerate().skip(1) {
            for (t, entry) in row.iter_mut().enumerate().skip(1) {
                // Ordered direction pairs, so a plane contributes the rectangle
                // both ways round and the two assignments of the sides are
                // averaged together rather than filling separate entries.
                let orientations = (0..D)
                    .flat_map(|mu| (0..D).map(move |nu| (mu, nu)))
                    .filter(|&(mu, nu)| mu != nu);

                // Sums of `±1` stay exact in `f64` far past any lattice size, so
                // this accumulates directly rather than through an integer.
                let mut total = 0.0;
                let mut placements = 0usize;
                for (mu, nu) in orientations {
                    // Both sides are capped below every extent, so no rectangle
                    // here can wrap and the constructor cannot refuse one.
                    let path = Loop::rectangle(lattice, mu, r, nu, t)
                        .expect("a rectangle shorter than the extents closes");
                    for site in 0..lattice.n_sites() {
                        total += self.loop_product(lattice, config, site, &path);
                    }
                    placements += lattice.n_sites();
                }

                *entry = total / placements as f64;
            }
        }
        table
    }
}

impl<const D: usize> Action<2, D> for Z2Gauge {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        assert!(D >= Self::MIN_DIMENSION, "{}", Self::TOO_FEW_DIMENSIONS);
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        -self.j * self.plaquette_sum(lattice, config)
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        var: usize,
        proposed: State<2>,
    ) -> f64 {
        assert!(D >= Self::MIN_DIMENSION, "{}", Self::TOO_FEW_DIMENSIONS);
        debug_assert!(
            config.cell() == Cell::Link,
            "gauge variables live on links, not {:?}",
            config.cell()
        );

        // A plaquette containing this link splits into the link's own variable
        // times the product over the plaquette's three others — its staple — so
        // the whole of H that depends on σ_l is σ_l times the sum of the staple
        // products, and
        //
        //     ΔE = -j * (σ'_l - σ_l) * sum_{g in staples(l)} prod_{l' in g} σ_l'.
        //
        // Plaquettes not containing the link are untouched and cancel, so this
        // reads the `2 * (D - 1)` staple groups and nothing else.
        let ds = (decode(proposed) - decode(config.peek(var))) as i64;
        if ds == 0 {
            return 0.0; // proposed state equals the current one
        }

        // Integer accumulator, exact until the final scaling, as in `energy`.
        let staple_sum: i64 = lattice
            .link_staples(var)
            .chunks_exact(3)
            .map(|group| {
                group
                    .iter()
                    .map(|&link| decode(config.peek(link)))
                    .product::<i32>() as i64
            })
            .sum();

        -self.j * (ds * staple_sum) as f64
    }
}

/// The ferromagnetic `q`-state Potts model: one label out of `Q` on every site,
/// coupled by whether nearest neighbors carry the *same* label, with an optional
/// per-label energy offset,
///
/// ```text
/// H = -j * sum_<ij> delta(s_i, s_j)  -  sum_i h[s_i]
/// ```
///
/// where the first sum runs over each nearest-neighbor bond once and `delta` is
/// `1` when the two labels agree and `0` otherwise. `j > 0` is ferromagnetic:
/// agreeing neighbors lower the energy. Energies come out in the same units as
/// `j` and `h`.
///
/// The offset `h` is one number per label, and a site pays the one matching its
/// own label whatever its neighbors are doing — a chemical potential per label,
/// which is what the second sum literally is once it is read as `-sum_a h[a] *
/// N_a` over the label populations. It is uniform across the lattice: every site
/// carrying a given label gets the same number, so the sum is over the `Q`
/// labels and not over anything geometric. Adding a constant to every entry
/// shifts `H` by a constant and cancels out of every energy difference, so only
/// the differences between entries carry content, and the usual choice is a
/// single entry offset with the rest at zero.
///
/// The labels are *unordered*, and that is the whole character of the model.
/// Nothing distinguishes one from another, so the energy only ever asks whether
/// two indices are equal and never turns one into a number — which is why this
/// is the one action here generic over `Q` rather than fixed at the two states
/// the private `decode` knows. Permuting the `Q` labels the same way at every
/// site leaves the energy untouched, and everything measured has to respect that
/// symmetry; see [`order_parameter`](Potts::order_parameter) for why the Ising
/// magnetization has no counterpart under it.
///
/// At `Q = 2` and zero offset this is the Ising model up to a factor of two in
/// the coupling. Reading the two labels as `±1` gives
/// `delta(s_i, s_j) = (1 + s_i s_j) / 2`, so `Potts` at coupling `2J` differs
/// from zero-field [`Ising`] at `J` only by the constant `-J` per bond — the same
/// energy landscape, and therefore the same `energy_delta` on every move. The
/// offsets extend that: `h = [H, -H]` reproduces an Ising field of `H`, since
/// Ising's `-h * sum_i s_i` gives `+h` to an up spin and `-h` to a down one.
///
/// Both models are kept rather than one wrapping the other: [`Ising`]'s
/// observables are built on reading states as signs, its field is a scalar
/// rather than a vector, and its GPU kernel is a faster specialized path.
///
/// The offsets break the relabelling symmetry, which is the point of having
/// them. At `h = 0` the `Q` ordered ground states are exactly degenerate, so no
/// label-specific average survives a long run; a nonzero entry lifts that
/// degeneracy and picks one out. Everything measured on a symmetric run — the
/// order parameters especially — is built to be invariant under relabelling, and
/// that argument does not survive a nonzero `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Potts<const Q: usize> {
    /// Nearest-neighbor coupling `J`.
    j: f64,
    /// Per-label energy offset: entry `a` is subtracted from the energy once for
    /// every site carrying label `a`. All zeros is the symmetric model.
    h: [f64; Q],
}

/// The fewest dimensions [`Potts`] is defined in.
///
/// One, the same as [`Ising::MIN_DIMENSION`] and for the same reason: the energy
/// scores nearest-neighbor bonds, and a ring of labels has them.
///
/// A free constant rather than an associated one, unlike the other two models',
/// because `Potts` is generic over `Q` and reading an associated constant off it
/// would mean naming some `Q` — which reads as though the floor depended on the
/// state count. It does not, and neither does the one below, which *constrains*
/// the state count and so certainly cannot be indexed by it.
/// [`Potts::MIN_DIMENSION`] aliases this so the type still answers the question
/// its siblings answer.
pub const POTTS_MIN_DIMENSION: usize = 1;

/// The fewest states [`Potts`] is defined at.
///
/// Two. One state is not a small Potts model but no model at all: every site
/// carries the same label, every bond agrees, the energy is the same constant
/// for every configuration, and there is no other state to propose.
/// [`Potts::order_parameter`] would divide by `Q - 1 = 0` on top of that. None of
/// this fails loudly on its own — it produces a chain that runs and reports
/// numbers — which is why the floor is stated rather than left to the
/// arithmetic, exactly as [`Z2Gauge::MIN_DIMENSION`] is.
pub const POTTS_MIN_STATES: usize = 2;

/// What every guard on [`POTTS_MIN_STATES`] says, so the places that check it
/// cannot drift into several wordings.
pub(crate) const TOO_FEW_STATES: &str =
    "the Potts action scores agreement between labels, which needs at least two states";

impl<const Q: usize> Potts<Q> {
    /// The fewest dimensions this model is defined in — [`POTTS_MIN_DIMENSION`].
    pub const MIN_DIMENSION: usize = POTTS_MIN_DIMENSION;

    /// The fewest states this model is defined at — [`POTTS_MIN_STATES`].
    pub const MIN_STATES: usize = POTTS_MIN_STATES;

    /// A Potts action with nearest-neighbor coupling `j` and per-label energy
    /// offsets `h`.
    ///
    /// Pass `[0.0; Q]` for the symmetric model, or use
    /// [`symmetric`](Potts::symmetric), which says the same thing without making
    /// the reader count zeros.
    pub fn new(j: f64, h: [f64; Q]) -> Self {
        Potts { j, h }
    }

    /// A Potts action with coupling `j` and no per-label offsets — the symmetric
    /// model, where all `Q` labels are interchangeable and the `Q` ordered ground
    /// states are degenerate.
    ///
    /// This is the standard case and the one every exact result quoted in
    /// `docs/potts.md` assumes, which is why it gets its own constructor rather
    /// than a defaulted argument.
    pub fn symmetric(j: f64) -> Self {
        Potts::new(j, [0.0; Q])
    }

    /// The nearest-neighbor coupling this action carries.
    pub fn coupling(&self) -> f64 {
        self.j
    }

    /// The per-label energy offsets this action carries.
    pub fn offsets(&self) -> [f64; Q] {
        self.h
    }

    /// Whether every offset is zero, so the `Q` labels are interchangeable.
    ///
    /// Read as a shortcut rather than as a question about the model: the offset
    /// term is then identically zero for every configuration, so `energy` can
    /// skip counting labels altogether. `Q` comparisons in place of a scan of
    /// the lattice, on what is by default every run.
    fn is_symmetric(&self) -> bool {
        self.h.iter().all(|&h_a| h_a == 0.0)
    }

    /// How many sites carry each label — the one lattice scan every quantity
    /// below is a reduction of, and the only thing any of them reads.
    ///
    /// Counting is where the relabelling symmetry is dealt with: a permutation
    /// of the labels permutes this array and nothing else, so any reduction of
    /// it that ignores the *order* of the entries is automatically invariant.
    /// Both order parameters are such a reduction — one takes the maximum, the
    /// other a sum of squares — which is why neither has to argue for its own
    /// invariance separately.
    ///
    /// It is `pub(crate)` so [`potts_measure`](crate::observables::potts_measure)
    /// can scan once and hand the counts to both reductions, rather than each
    /// public method walking the lattice for itself. The contract checks live
    /// here rather than in the callers because every path reaches them through
    /// this one.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub(crate) fn label_counts(config: &Configuration<Q>) -> [usize; Q] {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Integer counts: exact, and the divides by N happen in the callers.
        let mut counts = [0usize; Q];
        for state in config.variables() {
            counts[state.index()] += 1;
        }
        counts
    }

    /// [`order_parameter`](Potts::order_parameter) from counts already in hand.
    pub(crate) fn order_from_counts(counts: &[usize; Q], n_vars: usize) -> f64 {
        let most = counts.iter().copied().max().expect("Q >= 2 counts exist");
        let f_max = most as f64 / n_vars as f64;
        (Q as f64 * f_max - 1.0) / (Q as f64 - 1.0)
    }

    /// [`simplex_order_parameter`](Potts::simplex_order_parameter) from counts
    /// already in hand.
    pub(crate) fn simplex_from_counts(counts: &[usize; Q], n_vars: usize) -> f64 {
        // Integer accumulator: `sum_a N_a^2`, exact until the final scaling. The
        // square of a count fits comfortably, since the counts sum to N.
        let squares: u128 = counts.iter().map(|&c| (c as u128) * (c as u128)).sum();

        let n = n_vars as f64;
        let sum_f_squared = squares as f64 / (n * n);
        // Rounding can put the bracket a hair below zero on an exactly uniform
        // split, where it is analytically zero; clamping keeps the square root
        // real rather than returning NaN at the disordered floor.
        ((Q as f64 * sum_f_squared - 1.0) / (Q as f64 - 1.0))
            .max(0.0)
            .sqrt()
    }

    /// The order parameter `m = (Q * f_max - 1) / (Q - 1)`, where `f_max` is the
    /// fraction of sites carrying the most common label.
    ///
    /// The normalization is the conventional one, chosen so both ends come out
    /// clean: in the disordered phase every label holds about `1/Q` of the
    /// sites, so `f_max -> 1/Q` and `m -> 0`, while a fully ordered
    /// configuration puts every site on one label and gives `m = 1`.
    ///
    /// It is built from the most populated label rather than from a signed sum
    /// because the labels are unordered — there is no `+1` and `−1` to add up,
    /// so [`Ising::magnetization`]'s definition has nothing to carry over. What
    /// does survive the relabelling symmetry is the *imbalance* between the
    /// populations, and reading it off the largest population is what makes the
    /// answer invariant: a permutation of the labels permutes the counts and
    /// leaves their maximum where it was.
    ///
    /// The price is that this is unsigned and cannot be otherwise, so a
    /// downstream reduction cannot recover both `<m>` and `<|m|>` from the
    /// series the way it can from [`Ising::magnetization`]. This already *is*
    /// the analogue of `<|m|>`.
    ///
    /// This is one of the two conventions in use, and
    /// [`simplex_order_parameter`](Potts::simplex_order_parameter) is the other;
    /// they agree at both ends and differ in between, so a comparison against a
    /// published number has to know which one it is.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn order_parameter(&self, config: &Configuration<Q>) -> f64 {
        Self::order_from_counts(&Self::label_counts(config), config.n_vars())
    }

    /// The order parameter in its *vector* form:
    /// `m = sqrt[(Q * sum_a f_a^2 - 1) / (Q - 1)]`, where `f_a` is the fraction
    /// of sites carrying label `a`.
    ///
    /// The other convention in use, and the one most published Potts curves are.
    /// It comes from a different construction: place the `Q` labels at the
    /// vertices of a regular `Q-1` dimensional simplex, so that unit vectors
    /// `e_a` satisfy `e_a . e_b = -1/(Q-1)` for `a != b` and sum to zero, and
    /// take the length of the average `(1/N) sum_i e_{s_i}`. Expanding that
    /// square collapses the vectors out entirely and leaves the populations,
    /// which is why this costs the same scan
    /// [`order_parameter`](Potts::order_parameter) does rather than any geometry.
    ///
    /// It agrees with the other convention at both ends — one at a uniform
    /// configuration, zero at an equal split — and differs between them: half
    /// the sites on one label and half on another gives `1/4` there and `1/2`
    /// here. At `Q = 2` the simplex is an interval and the two coincide
    /// identically, both reducing to `|M| / N` of the corresponding Ising field.
    ///
    /// This is the form whose `Q - 1` components make the Ising `3` in
    /// [`binder_cumulant`](crate::statistics::binder_cumulant) the wrong
    /// normalization for Potts; `docs/potts.md` carries that argument.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn simplex_order_parameter(&self, config: &Configuration<Q>) -> f64 {
        Self::simplex_from_counts(&Self::label_counts(config), config.n_vars())
    }

    /// The per-config connected two-point function
    /// `C_r = (1/N) Σ_i [delta(s_i, s_{i+r}) - 1/Q]`, measured along each lattice
    /// axis under periodic boundaries.
    ///
    /// Returns one row per axis, laid out exactly as [`Ising::correlator`]'s:
    /// entry `μ`, index `r` covers displacements `r = 0..=L_μ/2`, the
    /// non-redundant half that translation invariance leaves.
    ///
    /// Unlike the Ising correlator this subtracts a floor, because the raw
    /// agreement fraction does not decay to zero: two labels far enough apart to
    /// be independent still agree with probability `1/Q`, so an uncorrelated
    /// configuration gives `1/Q` at every separation rather than `0`. Taking
    /// that off is what leaves a quantity that falls to zero in the disordered
    /// phase the way `<s_i s_{i+r}>` does, and so what makes a correlation
    /// length readable off the decay. The `r = 0` anchor is then `1 - 1/Q`
    /// rather than Ising's `1`, since a site always agrees with itself.
    ///
    /// This is the raw per-config estimator only. The ensemble average and the
    /// correlation-length fit are reductions over a chain of these arrays and
    /// belong to statistics.
    ///
    /// # Panics
    ///
    /// Panics if `Q` is below [`MIN_STATES`](Potts::MIN_STATES).
    pub fn correlator<const D: usize>(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
    ) -> [Vec<f64>; D] {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        let agree = axis_pair_sums(lattice, config, |s_i, s_j| i64::from(s_i == s_j));
        let n = config.n_vars() as f64;
        let floor = 1.0 / Q as f64;
        std::array::from_fn(|mu| agree[mu].iter().map(|&c| c as f64 / n - floor).collect())
    }
}

impl<const Q: usize, const D: usize> Action<Q, D> for Potts<Q> {
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<Q>) -> f64 {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Integer accumulator: the count is exact until the final scaling.
        let mut agree: i64 = 0;
        for site in 0..config.n_vars() {
            let s_i = config.peek(site);
            // The neighbor row is ordered +0, −0, +1, −1, ...; taking the
            // forward columns only (every other entry) visits each bond once.
            for &j_site in lattice.site_neighbors(site).iter().step_by(2) {
                agree += i64::from(config.peek(j_site) == s_i);
            }
        }

        // The offset term is `-sum_a h[a] * N_a`, so it needs the label
        // populations and nothing about the geometry — the same counts the order
        // parameters reduce. The symmetric model skips the scan outright rather
        // than walking the lattice to multiply by zeros, which is worth the test
        // because it is the default and what every exact result assumes.
        let offset: f64 = if self.is_symmetric() {
            0.0
        } else {
            let counts = Self::label_counts(config);
            self.h
                .iter()
                .zip(counts)
                .map(|(&h_a, n_a)| h_a * n_a as f64)
                .sum()
        };

        -self.j * agree as f64 - offset
    }

    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<Q>,
        var: usize,
        proposed: State<Q>,
    ) -> f64 {
        assert!(Q >= POTTS_MIN_STATES, "{}", TOO_FEW_STATES);
        debug_assert!(
            config.cell() == Cell::Site,
            "Potts labels live on sites, not {:?}",
            config.cell()
        );

        // Only bonds touching `var` can change their agreement; every other bond
        // reads two untouched labels and cancels. So
        //
        //     ΔE = -j * (agree_after - agree_before),
        //
        // where each term counts how many of the site's 2D neighbors carry the
        // proposed or the current label. The two labels differ, so a neighbor
        // contributes to at most one of the counts.
        let current = config.peek(var);
        if proposed == current {
            return 0.0; // proposed state equals the current one
        }

        // Integer accumulator, exact until the final scaling, as in `energy`.
        let mut change: i64 = 0;
        for &j_site in lattice.site_neighbors(var) {
            let s_j = config.peek(j_site);
            change += i64::from(s_j == proposed) - i64::from(s_j == current);
        }

        // Only this site's own offset changes, and only by the difference
        // between the two labels' entries — which is why a constant added to
        // every entry is invisible to the chain.
        let offset = self.h[proposed.index()] - self.h[current.index()];

        -self.j * change as f64 - offset
    }
}

/// A runtime choice among the built-in actions, so a model named in a config
/// file can be selected without the caller committing to a type at compile
/// time.
///
/// The counterpart of [`AnyUpdater`](crate::updater::AnyUpdater), and the same
/// reasoning: the types are fixed at compile time but which one a run uses is a
/// value read from a file, so the two have to meet at a single type. The
/// variants are a closed set, which is what makes the choice recordable.
///
/// It unifies the energy seam and only that. The two models do not measure the
/// same quantities — [`Ising`] has a magnetization, [`Z2Gauge`] a plaquette sum
/// and Wilson loops — so a caller that measures still branches on the model
/// rather than reading through this enum; see
/// [`observables`](crate::observables).
///
/// [`Potts`] is absent, and not by oversight. This enum implements
/// `Action<2, D>` because both its variants do, whereas `Potts` is generic over
/// `Q` and its `Q = 2` case is a different model from the one anyone runs it
/// for. Admitting it would mean carrying `Q` on the enum, which the other two
/// cannot satisfy at anything but two — so the third model waits for the same
/// unified runtime this is already waiting for.
///
/// Nothing consumes it yet, by design. The Ising and gauge runtimes are separate
/// siblings today — [`IsingSampler`](crate::ising_sampler::IsingSampler) and
/// [`GaugeSampler`](crate::gauge_sampler::GaugeSampler), each holding a concrete
/// model — and this is the seam the eventual *unified* runtime will hold, the way
/// `IsingSampler` already holds an [`AnyUpdater`](crate::updater::AnyUpdater):
/// dispatched from a `ModelKind` discriminant in [`config`](crate::config) that
/// would mirror [`UpdaterKind`](crate::config::UpdaterKind), and built ahead of
/// that consumer only because `AnyUpdater` has already fixed its shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnyAction {
    /// The site model, [`Ising`].
    Ising(Ising),
    /// The link model, [`Z2Gauge`].
    Z2Gauge(Z2Gauge),
}

impl AnyAction {
    /// The cell kind a configuration must sit on for this action to price it.
    ///
    /// Choosing the model and choosing where its variables live is one decision,
    /// and a caller building a configuration for an action picked at runtime has
    /// no other way to ask. Without it the `debug_assert!`s in the energies are
    /// the only thing between a mismatched field and wrong physics, and those
    /// are compiled out of a release build.
    pub fn cell(&self) -> Cell {
        match self {
            AnyAction::Ising(_) => Cell::Site,
            AnyAction::Z2Gauge(_) => Cell::Link,
        }
    }
}

impl<const D: usize> Action<2, D> for AnyAction {
    /// Forward to the wrapped action. The match is the whole cost of runtime
    /// dispatch — one branch per model, and on this path only once per measured
    /// configuration.
    fn energy(&self, lattice: &Lattice<D>, config: &Configuration<2>) -> f64 {
        match self {
            AnyAction::Ising(action) => action.energy(lattice, config),
            AnyAction::Z2Gauge(action) => action.energy(lattice, config),
        }
    }

    /// Forward to the wrapped action. This one sits in the sampler's hot path,
    /// so the branch is per proposed flip; it is predictable, since a run never
    /// switches model mid-chain.
    fn energy_delta(
        &self,
        lattice: &Lattice<D>,
        config: &Configuration<2>,
        var: usize,
        proposed: State<2>,
    ) -> f64 {
        match self {
            AnyAction::Ising(action) => action.energy_delta(lattice, config, var, proposed),
            AnyAction::Z2Gauge(action) => action.energy_delta(lattice, config, var, proposed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_maps_to_plus_minus_one() {
        assert_eq!(decode(State::new(0).unwrap()), 1);
        assert_eq!(decode(State::new(1).unwrap()), -1);
    }

    #[test]
    fn correlator_of_aligned_config_is_all_ones() {
        // Every spin +1: s_i · s_{i+r} = 1 for every site and displacement, so
        // C_r = 1 on every axis, including the C_0 normalization.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let config = Configuration::<2>::cold(&lat, Cell::Site);

        let c = action.correlator(&lat, &config);
        assert_eq!(c.len(), 2); // one row per axis
        for row in &c {
            assert_eq!(row.len(), 3); // r = 0..=L/2, i.e. L/2 + 1 = 3 for L = 4
            assert!(row.iter().all(|&v| v == 1.0));
        }
    }

    #[test]
    fn correlator_of_checkerboard_alternates() {
        // Checkerboard (spin = (−1)^(x+y)): a one-step move along either axis
        // flips the spin, so C_r = (−1)^r. The r = 2 shift exercises the periodic
        // wrap for sites near the boundary; L = 4 is even, so the pattern is
        // consistent under those boundaries.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        let down = State::new(1).unwrap();
        for site in 0..lat.n_sites() {
            let x = lat.site_coords(site);
            if (x[0] + x[1]) % 2 == 1 {
                config.poke(site, down);
            }
        }

        let c = action.correlator(&lat, &config);
        let expected = [1.0, -1.0, 1.0]; // (−1)^r for r = 0..=L/2 = 0, 1, 2
        assert_eq!(c[0], expected);
        assert_eq!(c[1], expected);
    }

    #[test]
    fn magnetization_is_the_signed_spin_sum() {
        // 4x4 = 16 sites. Cold is all state 0 (+1), so M = +N.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(action.magnetization(&config), 16.0);

        // 4 up, 12 down -> M = 4 − 12 = −8: the sign is kept, not folded to |M|.
        let down = State::new(1).unwrap();
        for site in 0..12 {
            config.poke(site, down);
        }
        assert_eq!(action.magnetization(&config), -8.0);
    }

    #[test]
    fn cold_start_is_the_ground_state_energy() {
        // All spins +1: every bond contributes +1. A D-dim periodic lattice has
        // D forward bonds per site, so bond_sum = D * N and E = -j * D * N.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let action = Ising::new(1.0, 0.0);
        assert_eq!(action.energy(&lat, &config), -32.0); // -1 * 2 * 16
    }

    #[test]
    fn field_term_tracks_total_magnetization() {
        // Cold (all +1) with j = 0 isolates the field term: E = -h * N.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let action = Ising::new(0.0, 0.5);
        assert_eq!(action.energy(&lat, &config), -8.0); // -0.5 * 16
    }

    #[test]
    fn energy_delta_matches_from_scratch_difference() {
        // j = 1.0 and h = 0.5 are exactly representable and the sums are integer,
        // so the two sides agree bit-for-bit and bare `==` is legitimate here.
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();

        // A non-uniform configuration so neighbor sums actually vary.
        let mut config = Configuration::<2>::cold(&lat, Cell::Site);
        for &s in &[5usize, 6, 10] {
            config.poke(s, down);
        }

        for &site in &[0usize, 5, 6, 9, 10] {
            let proposed = if config.peek(site) == up { down } else { up };
            let before = action.energy(&lat, &config);
            let delta = action.energy_delta(&lat, &config, site, proposed);

            let mut after = config.clone();
            after.poke(site, proposed);
            assert_eq!(delta, action.energy(&lat, &after) - before);
        }
    }

    /// How many variables `deltas_match_from_scratch` probes on a large lattice.
    ///
    /// Every variable is checked when a lattice is small enough, because that is
    /// the strongest form of the test. Each probe costs a whole-lattice energy
    /// scan, so checking all of them is quadratic in the variable count and a
    /// ten-dimensional box runs into billions of operations under a debug build.
    /// Past the cap the probes are spread evenly across the index range instead,
    /// which still reaches every direction and both parities.
    const MAX_PROBES: usize = 64;

    /// Flip variables of `config` in turn and check the incremental energy
    /// against the difference of two from-scratch energies.
    ///
    /// This is the sharpest check that a model carries no assumption about the
    /// dimension: `energy` walks cells by index while `energy_delta` walks the
    /// incidence tables, so the two agree only if the packing, the strides, and
    /// the neighbor and staple rows all say the same thing about the geometry.
    /// A `D`-specific mistake anywhere in that chain shows up as a mismatch on
    /// some variable rather than as a plausible wrong number.
    fn deltas_match_from_scratch<const Q: usize, const D: usize, A: Action<Q, D>>(
        shape: [usize; D],
        action: &A,
        cell: Cell,
    ) {
        let lat = Lattice::new(shape);
        let mut config = Configuration::<Q>::cold(&lat, cell);
        // A non-uniform configuration, so the neighbor and staple sums vary
        // rather than all collapsing onto the ground state's. Stepping the state
        // by the variable index spreads all `Q` of them over the field, which at
        // `Q = 2` is the same alternating pattern this used to lay down.
        for var in (0..config.n_vars()).step_by(3) {
            config.poke(var, State::new(1 + var % (Q - 1)).unwrap());
        }

        // `config` is never mutated below — each probe clones it — so the
        // reference energy is computed once rather than per probe.
        let before = action.energy(&lat, &config);
        let n_vars = config.n_vars();
        let step = n_vars.div_ceil(MAX_PROBES).max(1);
        for var in (0..n_vars).step_by(step) {
            // The next state cyclically, which is a real change at every `Q` and
            // is the plain flip at two.
            let proposed = State::new((config.peek(var).index() + 1) % Q).unwrap();
            let delta = action.energy_delta(&lat, &config, var, proposed);

            let mut after = config.clone();
            after.poke(var, proposed);
            // The couplings below are exactly representable and the sums are
            // integer, so the two sides agree bit-for-bit.
            assert_eq!(
                delta,
                action.energy(&lat, &after) - before,
                "{shape:?}: variable {var}"
            );
        }
    }

    /// The dimensions both models are checked at, up to ten.
    ///
    /// Ten is a ceiling on what anyone would plausibly run rather than anything
    /// the code knows about — nothing in the library states an upper bound, and
    /// `Lattice<12>` would compile. The point of going this far is that every
    /// count the incidence tables are built from grows with `D`, some of them
    /// quadratically: at ten dimensions a site has twenty neighbors, a link sits
    /// in eighteen plaquettes, and each site anchors forty-five of them.
    ///
    /// The unequal extents matter as much as the dimensions. A cubic shape hides
    /// a transposed stride, since every axis then has the same place value.
    #[test]
    fn ising_deltas_match_from_scratch_in_every_dimension() {
        let action = Ising::new(1.0, 0.5);
        deltas_match_from_scratch([6], &action, Cell::Site);
        deltas_match_from_scratch([4, 4], &action, Cell::Site);
        deltas_match_from_scratch([3, 4, 5], &action, Cell::Site);
        deltas_match_from_scratch([2, 3, 2, 3], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 3, 2, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 3, 2, 2, 3, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 2, 3, 2, 2, 2, 2], &action, Cell::Site);
        deltas_match_from_scratch([2, 2, 2, 2, 3, 2, 2, 2, 2, 2], &action, Cell::Site);
    }

    /// The gauge counterpart, from two dimensions up, where a plaquette exists at
    /// all. See [`ising_deltas_match_from_scratch_in_every_dimension`] for why
    /// the shapes are shaped this way.
    #[test]
    fn gauge_deltas_match_from_scratch_in_every_dimension() {
        let action = Z2Gauge::new(1.0);
        deltas_match_from_scratch([4, 4], &action, Cell::Link);
        deltas_match_from_scratch([3, 4, 5], &action, Cell::Link);
        deltas_match_from_scratch([2, 3, 2, 3], &action, Cell::Link);
        deltas_match_from_scratch([2, 2, 3, 2, 2], &action, Cell::Link);
        deltas_match_from_scratch([2, 3, 2, 2, 3, 2], &action, Cell::Link);
        deltas_match_from_scratch([2, 2, 2, 3, 2, 2, 2, 2], &action, Cell::Link);
        deltas_match_from_scratch([2, 2, 2, 2, 3, 2, 2, 2, 2, 2], &action, Cell::Link);
    }

    /// The Potts counterpart, at three states rather than two.
    ///
    /// Three is the point of it: `Q = 2` would run the same arithmetic the Ising
    /// sweep above already covers, while at three a site's neighbors can carry a
    /// label that is neither the current nor the proposed one, so the `after`
    /// and `before` counts stop being complements of each other. A delta written
    /// as though they were would pass at two states and fail here.
    ///
    /// See [`ising_deltas_match_from_scratch_in_every_dimension`] for why the
    /// shapes are shaped this way.
    #[test]
    fn potts_deltas_match_from_scratch_in_every_dimension() {
        fn probe<const D: usize>(shape: [usize; D]) {
            deltas_match_from_scratch::<3, D, _>(shape, &Potts::symmetric(1.0), Cell::Site);
        }

        probe([6]);
        probe([4, 4]);
        probe([3, 4, 5]);
        probe([2, 3, 2, 3]);
        probe([2, 2, 3, 2, 2]);
        probe([2, 3, 2, 2, 3, 2]);
        probe([2, 2, 2, 3, 2, 2, 2, 2]);
        probe([2, 2, 2, 2, 3, 2, 2, 2, 2, 2]);
    }

    /// A uniform Potts configuration is the ground state: every bond agrees, so
    /// the energy is `-j` times the bond count, whichever label it settled on.
    ///
    /// Running it at each of the `Q` labels in turn is the cheapest form of the
    /// relabelling symmetry — nothing may single out label `0`, which is the one
    /// a cold start happens to pick.
    #[test]
    fn a_uniform_potts_configuration_has_every_bond_agreeing() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6, 4]);
        let action = Potts::symmetric(1.5);
        // A periodic lattice has `D` forward bonds per site, which is exactly
        // the link count.
        let n_bonds = lat.n_links() as f64;

        for label in 0..Q {
            let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
            for site in 0..config.n_vars() {
                config.poke(site, State::new(label).unwrap());
            }

            assert_eq!(
                action.energy(&lat, &config),
                -1.5 * n_bonds,
                "label {label}"
            );
            assert_eq!(action.order_parameter(&config), 1.0, "label {label}");
        }
    }

    /// The Potts symmetry, which has no Ising analogue: relabelling every site
    /// by one permutation of the `Q` states leaves the energy alone.
    ///
    /// This is the sharpest check that the energy compares labels rather than
    /// reading them as numbers. A term that decoded an index into a value — the
    /// way both other models do — would move under a permutation that reorders
    /// those values, while the count of agreeing bonds cannot: a permutation is
    /// a bijection, so `s_i == s_j` before and after are the same statement.
    #[test]
    fn potts_energy_is_invariant_under_a_global_relabelling() {
        const Q: usize = 4;
        let lat = Lattice::new([4, 4, 4]);
        let action = Potts::symmetric(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(31);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);
        let order_before = action.order_parameter(&config);
        let simplex_before = action.simplex_order_parameter(&config);
        let correlator_before = action.correlator(&lat, &config);

        // A three-cycle on the first three labels, leaving the fourth fixed, so
        // the permutation is neither the identity nor a mere swap.
        let permutation = [1usize, 2, 0, 3];
        let mut relabelled = config.clone();
        for site in 0..config.n_vars() {
            let to = permutation[config.peek(site).index()];
            relabelled.poke(site, State::new(to).unwrap());
        }
        assert_ne!(
            relabelled, config,
            "the relabelling should change the field"
        );

        assert_eq!(action.energy(&lat, &relabelled), before);
        assert_eq!(action.order_parameter(&relabelled), order_before);
        assert_eq!(action.simplex_order_parameter(&relabelled), simplex_before);
        assert_eq!(action.correlator(&lat, &relabelled), correlator_before);
    }

    /// Potts at two states is Ising at half the coupling: the two price every
    /// move identically.
    ///
    /// Reading the two labels as `±1` gives `delta = (1 + s_i s_j) / 2`, so a
    /// `Potts` bond at coupling `2J` is a zero-field `Ising` bond at `J` plus the
    /// constant `-J`. Constants cancel in a difference, so the two `energy_delta`
    /// values agree exactly, and the whole-lattice energies differ by `-J` times
    /// the bond count. Two independently written actions landing on the same
    /// numbers is what makes this worth asserting rather than deriving.
    #[test]
    fn potts_at_two_states_prices_moves_like_ising() {
        let lat = Lattice::new([4, 6]);
        let j = 1.5;
        let potts = Potts::symmetric(2.0 * j);
        let ising = Ising::new(j, 0.0);

        let mut rng = crate::rng::RandRng::seed_from_u64(97);
        let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        // The whole-lattice energies differ by the constant the delta drops.
        let offset = -j * lat.n_links() as f64;
        assert_eq!(
            potts.energy(&lat, &config),
            ising.energy(&lat, &config) + offset
        );

        for site in 0..config.n_vars() {
            let proposed = State::new(1 - config.peek(site).index()).unwrap();
            assert_eq!(
                potts.energy_delta(&lat, &config, site, proposed),
                ising.energy_delta(&lat, &config, site, proposed),
                "site {site}"
            );
        }
    }

    /// A uniform configuration agrees at every separation, so every correlator
    /// entry is the full `1 - 1/Q` — the connected form's ceiling, and the value
    /// its `r = 0` anchor takes on any configuration at all.
    #[test]
    fn the_potts_correlator_of_a_uniform_config_is_one_minus_one_over_q() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6]);
        let action = Potts::symmetric(1.0);
        let config = Configuration::<Q>::cold(&lat, Cell::Site);

        let c = action.correlator(&lat, &config);
        assert_eq!(c.len(), 2); // one row per axis
        let ceiling = 1.0 - 1.0 / Q as f64;
        for (mu, row) in c.iter().enumerate() {
            assert_eq!(row.len(), lat.shape()[mu] / 2 + 1);
            assert!(row.iter().all(|&v| v == ceiling), "axis {mu}: {row:?}");
        }
    }

    /// Independent labels agree at the uncorrelated rate `1/Q`, which is exactly
    /// what the connected form subtracts — so every entry past `r = 0` sits near
    /// zero, while `r = 0` keeps its `1 - 1/Q` anchor.
    ///
    /// This is the property that makes the subtraction worth doing: without it
    /// the correlator would tend to `1/Q` rather than to zero at large
    /// separation, and no correlation length could be read off the decay.
    #[test]
    fn the_potts_correlator_of_independent_labels_is_near_zero() {
        const Q: usize = 3;
        let lat = Lattice::new([32, 32]);
        let action = Potts::symmetric(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(20260809);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        let c = action.correlator(&lat, &config);
        for (mu, row) in c.iter().enumerate() {
            assert!(
                (row[0] - (1.0 - 1.0 / Q as f64)).abs() < 1e-12,
                "axis {mu}: C_0 = {}",
                row[0]
            );
            // 1024 sites of a two-valued indicator: the standard error on each
            // entry is about 0.015, so 0.05 is a few of those wide.
            for (r, &value) in row.iter().enumerate().skip(1) {
                assert!(value.abs() < 0.05, "axis {mu}, r = {r}: {value}");
            }
        }
    }

    /// The order parameter runs from 0 to 1: an equal split among the `Q` labels
    /// is the disordered floor, a single label the ordered ceiling.
    ///
    /// The floor is the reason for the `(Q f_max - 1) / (Q - 1)` normalization at
    /// all — the raw most-populated fraction would report `1/Q` there, which is
    /// a different number for every `Q` and no use as a scale.
    #[test]
    fn the_order_parameter_spans_zero_to_one() {
        const Q: usize = 4;
        let lat = Lattice::new([8, 8]); // 64 sites, divisible by Q
        let action = Potts::symmetric(1.0);

        let mut split = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..split.n_vars() {
            split.poke(site, State::new(site % Q).unwrap());
        }
        assert_eq!(action.order_parameter(&split), 0.0);

        let ordered = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(action.order_parameter(&ordered), 1.0);

        // Halfway between: half the sites on one label, the rest spread evenly
        // over the other three, so f_max = 1/2 and m = (4/2 - 1) / 3 = 1/3.
        let mut partial = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 32..partial.n_vars() {
            partial.poke(site, State::new(1 + site % (Q - 1)).unwrap());
        }
        assert!((action.order_parameter(&partial) - 1.0 / 3.0).abs() < 1e-12);
    }

    /// The two order parameters agree at both ends and differ in between.
    ///
    /// The ends are what the normalizations are chosen for, so agreeing there is
    /// no evidence that the two are different quantities — the half-and-half
    /// split is. It gives `1/4` under the most-populated reading, since one
    /// label holds half the sites, and `1/2` under the vector one, since two
    /// simplex vertices at sixty degrees average to half a unit vector. Anyone
    /// comparing a run against a published number has to know which of these it
    /// was, which is the whole reason both are kept.
    #[test]
    fn the_two_order_parameters_agree_at_the_ends_and_differ_between_them() {
        const Q: usize = 3;
        let lat = Lattice::new([6, 6]); // 36 sites, divisible by Q and by 2
        let action = Potts::symmetric(1.0);

        let uniform = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(action.order_parameter(&uniform), 1.0);
        assert_eq!(action.simplex_order_parameter(&uniform), 1.0);

        let mut split = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..split.n_vars() {
            split.poke(site, State::new(site % Q).unwrap());
        }
        assert_eq!(action.order_parameter(&split), 0.0);
        assert_eq!(action.simplex_order_parameter(&split), 0.0);

        let mut half = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..half.n_vars() / 2 {
            half.poke(site, State::new(1).unwrap());
        }
        assert!((action.order_parameter(&half) - 0.25).abs() < 1e-12);
        assert!((action.simplex_order_parameter(&half) - 0.5).abs() < 1e-12);
    }

    /// At two states the two conventions are the same function, and both are the
    /// Ising magnitude.
    ///
    /// A `Q - 1` dimensional simplex is an interval at `Q = 2`, so the vector
    /// construction has nowhere to point but along it and collapses onto the
    /// population imbalance the other convention reads directly. Both then equal
    /// `|M| / N` of the Ising field the same labels describe, which ties the two
    /// new quantities to one that was already tested rather than only to each
    /// other. Every configuration on the lattice is checked, since the identity
    /// is exact rather than statistical.
    #[test]
    fn at_two_states_both_order_parameters_are_the_ising_magnitude() {
        let lat = Lattice::new([4, 4]);
        let potts = Potts::symmetric(1.0);
        let ising = Ising::new(1.0, 0.0);
        let n_sites = lat.n_sites() as f64;

        let mut rng = crate::rng::RandRng::seed_from_u64(20260811);
        for _ in 0..32 {
            let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);
            let expected = (ising.magnetization(&config) / n_sites).abs();

            assert!((potts.order_parameter(&config) - expected).abs() < 1e-12);
            assert!((potts.simplex_order_parameter(&config) - expected).abs() < 1e-12);
        }
    }

    /// The per-label offset does what its definition says: each site pays the
    /// entry matching its own label, with no reference to its neighbors.
    ///
    /// Checked on a configuration whose label populations are fixed by
    /// construction, so the expected energy is arithmetic rather than a
    /// reference run: at `j = 0` the coupling term vanishes and the whole energy
    /// is the offset sum, which the counts give exactly.
    #[test]
    fn the_offsets_are_paid_once_per_site_carrying_the_label() {
        const Q: usize = 3;
        let lat = Lattice::new([6, 6]); // 36 sites, three equal populations
        let h = [0.5, -1.5, 0.25];
        let action = Potts::<Q>::new(0.0, h);

        let mut config = Configuration::<Q>::cold(&lat, Cell::Site);
        for site in 0..config.n_vars() {
            config.poke(site, State::new(site % Q).unwrap());
        }

        // Twelve sites on each label, and `j = 0` leaves only the offset term.
        let expected = -12.0 * (h[0] + h[1] + h[2]);
        assert_eq!(action.energy(&lat, &config), expected);
    }

    /// A constant added to every offset is invisible to the chain.
    ///
    /// The reason only differences carry content: shifting every entry by the
    /// same amount changes the energy of *every* configuration by the same
    /// constant, so it cancels out of every `energy_delta` and out of every
    /// Boltzmann ratio the sampler forms. Both halves are asserted, since a
    /// delta that quietly disagreed with the whole-lattice energy would break
    /// the telescoping identity the sweeps rely on.
    #[test]
    fn a_constant_added_to_every_offset_shifts_the_energy_and_nothing_else() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 6]);
        let shifted = Potts::<Q>::new(1.0, [1.25, 0.75, 2.25]);
        let base = Potts::<Q>::new(1.0, [0.25, -0.25, 1.25]); // each entry 1.0 lower

        let mut rng = crate::rng::RandRng::seed_from_u64(20260812);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);

        // The whole-lattice energies differ by the shift times the site count.
        let sites = config.n_vars() as f64;
        assert_eq!(
            shifted.energy(&lat, &config),
            base.energy(&lat, &config) - sites
        );

        // Every move is priced identically, which is what makes the two runs the
        // same chain rather than merely the same distribution.
        for site in 0..config.n_vars() {
            for label in 0..Q {
                let proposed = State::new(label).unwrap();
                assert_eq!(
                    shifted.energy_delta(&lat, &config, site, proposed),
                    base.energy_delta(&lat, &config, site, proposed),
                    "site {site} -> {label}"
                );
            }
        }
    }

    /// An offset breaks the relabelling symmetry, which is the whole point of
    /// having one.
    ///
    /// The symmetric model's invariance is asserted in
    /// [`potts_energy_is_invariant_under_a_global_relabelling`]; this is the
    /// other side of it. Permuting the labels while leaving `h` alone now moves
    /// the energy, because a site that was paying one entry starts paying
    /// another.
    #[test]
    fn an_offset_breaks_the_relabelling_symmetry() {
        const Q: usize = 3;
        let lat = Lattice::new([4, 4]);
        let action = Potts::<Q>::new(1.0, [0.75, 0.0, 0.0]);

        let mut rng = crate::rng::RandRng::seed_from_u64(20260813);
        let config = Configuration::<Q>::hot(&lat, Cell::Site, &mut rng);
        let before = action.energy(&lat, &config);

        // Cycle every label forward one place.
        let mut relabelled = config.clone();
        for site in 0..config.n_vars() {
            let to = (config.peek(site).index() + 1) % Q;
            relabelled.poke(site, State::new(to).unwrap());
        }

        // The coupling term is still invariant, so any change is the offset's.
        let symmetric = Potts::<Q>::symmetric(1.0);
        assert_eq!(
            symmetric.energy(&lat, &relabelled),
            symmetric.energy(&lat, &config)
        );
        assert_ne!(action.energy(&lat, &relabelled), before);
    }

    /// The offsets reproduce an Ising field at two states.
    ///
    /// `Potts` at coupling `2J` already matches zero-field `Ising` at `J`; this
    /// extends that to the field. Ising's `-h * sum_i s_i` gives `+h` to an up
    /// spin and `-h` to a down one, so `[H, -H]` is the same term written per
    /// label, and the two actions must again price every move identically.
    #[test]
    fn the_offsets_reproduce_an_ising_field_at_two_states() {
        let lat = Lattice::new([4, 6]);
        let (j, field) = (1.5, 0.25);
        let potts = Potts::<2>::new(2.0 * j, [field, -field]);
        let ising = Ising::new(j, field);

        let mut rng = crate::rng::RandRng::seed_from_u64(20260814);
        let config = Configuration::<2>::hot(&lat, Cell::Site, &mut rng);

        // Only the coupling term's constant separates the whole energies; the
        // field terms are equal outright.
        let offset = -j * lat.n_links() as f64;
        assert_eq!(
            potts.energy(&lat, &config),
            ising.energy(&lat, &config) + offset
        );

        for site in 0..config.n_vars() {
            let proposed = State::new(1 - config.peek(site).index()).unwrap();
            assert_eq!(
                potts.energy_delta(&lat, &config, site, proposed),
                ising.energy_delta(&lat, &config, site, proposed),
                "site {site}"
            );
        }
    }

    /// A link field is rejected rather than silently misread, as it is for the
    /// other site model.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Potts labels live on sites")]
    fn the_potts_action_rejects_a_link_field() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<3>::cold(&lat, Cell::Link);
        Potts::symmetric(1.0).energy(&lat, &config);
    }

    /// The state floor holds in release, unlike the cell-kind guards.
    ///
    /// Deliberately not gated on `debug_assertions`, for the reason
    /// [`the_gauge_action_refuses_one_dimension_in_any_profile`] is not: `Q` is a
    /// compile-time constant, so a real `assert!` folds away to nothing and there
    /// is no reason to make it debug-only. At one state every bond agrees for
    /// every configuration, so the chain would report a constant energy and an
    /// order parameter divided by zero — numbers, not a failure.
    #[test]
    #[should_panic(expected = "at least two states")]
    fn the_potts_action_refuses_one_state_in_any_profile() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<1>::cold(&lat, Cell::Site);
        Potts::symmetric(1.0).energy(&lat, &config);
    }

    /// The value-semantic half of the models, swept over dimensions.
    ///
    /// [`deltas_match_from_scratch`] covers the *incremental* seam; these cover
    /// what the observables read. Each is checked on a configuration whose answer
    /// is fixed by construction rather than by a reference run, so the assertions
    /// are exact at any dimension: a cold field has every bond, plaquette, and
    /// loop at its extreme value, and a gauge transformation is a symmetry of the
    /// energy whatever the lattice looks like.
    fn observables_hold_on_known_configurations<const D: usize>(shape: [usize; D]) {
        let lat = Lattice::new(shape);
        let n_sites = lat.n_sites() as f64;

        // Ising: cold is the ground state, `E = -j * n_links - h * n_sites`,
        // since every one of the `D * n_sites` forward bonds is aligned.
        let ising = Ising::new(1.5, 0.25);
        let cold = Configuration::<2>::cold(&lat, Cell::Site);
        assert_eq!(
            ising.energy(&lat, &cold),
            -1.5 * lat.n_links() as f64 - 0.25 * n_sites,
            "{shape:?}"
        );
        assert_eq!(ising.magnetization(&cold), n_sites, "{shape:?}");

        // The correlator is one row per axis, each `L_mu / 2 + 1` long, and all
        // ones on a cold field including the `C_0` anchor.
        let correlator = ising.correlator(&lat, &cold);
        assert_eq!(correlator.len(), D, "{shape:?}: one row per axis");
        for (mu, row) in correlator.iter().enumerate() {
            assert_eq!(row.len(), shape[mu] / 2 + 1, "{shape:?}: axis {mu}");
            assert!(row.iter().all(|&c| c == 1.0), "{shape:?}: axis {mu}");
        }

        // Potts at three states: cold is the ground state here too, and there is
        // no field term to add to it. Its correlator anchors at `1 - 1/Q` rather
        // than at Ising's `1`, because the connected form takes off the rate at
        // which independent labels agree.
        const Q: usize = 3;
        let potts = Potts::symmetric(1.5);
        let uniform = Configuration::<Q>::cold(&lat, Cell::Site);
        assert_eq!(
            potts.energy(&lat, &uniform),
            -1.5 * lat.n_links() as f64,
            "{shape:?}"
        );
        assert_eq!(potts.order_parameter(&uniform), 1.0, "{shape:?}");

        let ceiling = 1.0 - 1.0 / Q as f64;
        let agreement = potts.correlator(&lat, &uniform);
        assert_eq!(agreement.len(), D, "{shape:?}: one row per axis");
        for (mu, row) in agreement.iter().enumerate() {
            assert_eq!(row.len(), shape[mu] / 2 + 1, "{shape:?}: axis {mu}");
            assert!(row.iter().all(|&c| c == ceiling), "{shape:?}: axis {mu}");
        }

        if D < 2 {
            return; // no plaquette below two dimensions, so nothing gauge-like
        }

        // Gauge: cold is the ground state, `E = -j * n_plaquettes`.
        let gauge = Z2Gauge::new(1.5);
        let cold = Configuration::<2>::cold(&lat, Cell::Link);
        assert_eq!(
            gauge.plaquette_sum(&lat, &cold),
            lat.n_plaquettes() as f64,
            "{shape:?}"
        );
        assert_eq!(
            gauge.energy(&lat, &cold),
            -1.5 * lat.n_plaquettes() as f64,
            "{shape:?}"
        );

        // Every Wilson rectangle and every Polyakov line on a cold field is one.
        let max_side = shape.iter().min().copied().unwrap_or(0) / 2;
        let table = gauge.wilson_rectangles(&lat, &cold, max_side);
        assert_eq!(table.len(), max_side + 1, "{shape:?}");
        for (r, row) in table.iter().enumerate() {
            for (t, &w) in row.iter().enumerate() {
                assert_eq!(w, 1.0, "{shape:?}: W({r},{t})");
                assert_eq!(w, table[t][r], "{shape:?}: table is symmetric");
            }
        }
        for dir in 0..D {
            assert_eq!(
                gauge.polyakov_loop(&lat, &cold, dir),
                1.0,
                "{shape:?}: {dir}"
            );
        }

        // A gauge transformation is a symmetry: flipping every link touching a
        // site leaves the energy, the Wilson table, and every Polyakov line
        // alone, because each closed loop crosses the site's links an even
        // number of times. This is the sharpest check that `plaquette_links`,
        // `loop_links`, and `site_links` all agree about the geometry, and the
        // number of links a flip touches is `2 * D`, so it says more the higher
        // the dimension goes.
        let mut transformed = cold.clone();
        for site in (0..lat.n_sites()).step_by(3) {
            for link in lat.site_links(site) {
                let flipped = State::new(1 - transformed.peek(link).index()).unwrap();
                transformed.poke(link, flipped);
            }
        }
        // Without this the rest is vacuous: flipping a site's links twice would
        // put the configuration back and every assertion below would hold
        // trivially.
        assert_ne!(
            transformed, cold,
            "{shape:?}: the transformation left the configuration alone"
        );
        assert_eq!(
            gauge.energy(&lat, &transformed),
            gauge.energy(&lat, &cold),
            "{shape:?}: gauge transformation changed the energy"
        );
        assert_eq!(
            gauge.wilson_rectangles(&lat, &transformed, max_side),
            table,
            "{shape:?}: gauge transformation changed the Wilson table"
        );
        for dir in 0..D {
            assert_eq!(
                gauge.polyakov_loop(&lat, &transformed, dir),
                1.0,
                "{shape:?}: gauge transformation changed the Polyakov loop along {dir}"
            );
        }
    }

    /// The observables hold on known configurations at every dimension up to six.
    ///
    /// Six rather than the ten [`deltas_match_from_scratch`] reaches, because
    /// `wilson_rectangles` needs room for a rectangle — it caps its sides at half
    /// the shortest extent — so every extent here is at least four, and `4^10`
    /// sites is far past what belongs in a unit test. The narrower sweep is the
    /// one that costs volume; the exact energy check pays only for extents of two
    /// and so goes further.
    #[test]
    fn observables_hold_in_every_dimension() {
        observables_hold_on_known_configurations([6]);
        observables_hold_on_known_configurations([4, 6]);
        observables_hold_on_known_configurations([4, 6, 4]);
        observables_hold_on_known_configurations([4, 6, 4, 4]);
        observables_hold_on_known_configurations([4, 4, 6, 4, 4]);
        observables_hold_on_known_configurations([4, 4, 4, 4, 4, 4]);
    }

    #[test]
    fn gauge_cold_start_is_the_ground_state_energy() {
        // Every link +1, so every plaquette product is +1 and E = -j * n_plaq.
        // In 2D there is one plane, hence one plaquette per site: 16 of them.
        let action = Z2Gauge::new(1.0);
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        assert_eq!(action.energy(&lat, &config), -16.0);

        // In 3D there are C(3,2) = 3 planes per site: 3 * 27 = 81.
        let lat = Lattice::new([3, 3, 3]);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        assert_eq!(action.energy(&lat, &config), -81.0);
    }

    #[test]
    fn gauge_transformation_leaves_the_energy_unchanged() {
        // Flipping every link touching a chosen site leaves each plaquette
        // product alone, because a plaquette meeting that site does so with
        // exactly two of its four links and the two sign flips cancel. This is
        // the local symmetry that defines the model, and the sharpest check
        // that `energy` multiplies the right four links per face: a wrong link
        // in the product would break the pairing and move the energy.
        let lat = Lattice::new([3, 3, 3]);
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();

        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        for &link in &[0usize, 5, 13, 20, 44, 61] {
            config.poke(link, down);
        }
        let before = action.energy(&lat, &config);

        for &site in &[0usize, 7, 13] {
            for link in lat.site_links(site) {
                config.poke(link, if config.peek(link) == up { down } else { up });
            }
        }
        assert_eq!(action.energy(&lat, &config), before);
    }

    #[test]
    fn a_loop_product_on_a_cold_config_is_one() {
        // Every link is +1, so any product of them is +1 whatever the path —
        // including one that winds the torus and one that retraces itself.
        let lat = Lattice::new([4, 4]);
        let action = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let square = Loop::new(
            &lat,
            &[
                (0, Sign::Plus),
                (1, Sign::Plus),
                (0, Sign::Minus),
                (1, Sign::Minus),
            ],
        )
        .unwrap();
        let wind = Loop::new(&lat, &[(0, Sign::Plus); 4]).unwrap();

        for site in 0..lat.n_sites() {
            assert_eq!(action.loop_product(&lat, &config, site, &square), 1.0);
            assert_eq!(action.loop_product(&lat, &config, site, &wind), 1.0);
        }
    }

    #[test]
    fn a_unit_square_product_is_the_plaquette_term() {
        // The smallest loop is a plaquette, so its product must be the term the
        // energy sums — the check that this reads the same links through the
        // same decode as `energy` does.
        let lat = Lattice::new([3, 3, 3]);
        let action = Z2Gauge::new(1.0);
        let down = State::new(1).unwrap();
        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        for &link in &[0usize, 5, 13, 20, 44, 61] {
            config.poke(link, down);
        }

        let square = Loop::new(
            &lat,
            &[
                (0, Sign::Plus),
                (2, Sign::Plus),
                (0, Sign::Minus),
                (2, Sign::Minus),
            ],
        )
        .unwrap();

        for site in 0..lat.n_sites() {
            let p = lat.plaquette_index(lat.site_coords(site), 0, 2);
            let expected: i32 = lat
                .plaquette_links(p)
                .iter()
                .map(|&link| decode(config.peek(link)))
                .product();
            assert_eq!(
                action.loop_product(&lat, &config, site, &square),
                expected as f64,
                "site {site}"
            );
        }
    }

    #[test]
    fn a_gauge_transformation_leaves_a_loop_product_unchanged() {
        // The property that makes this the only measurable object: flipping
        // every link touching a site leaves the product alone, because a closed
        // path arrives at that site and leaves it again, picking up the flip
        // twice. A path that did not close would fail this, which is why `Loop`
        // refuses to build one.
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();

        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        for &link in &[1usize, 7, 18, 30, 55, 91, 140] {
            config.poke(link, down);
        }

        // A six-step staircase: three forward steps and their three retreats,
        // so it visits sites the flips below will touch.
        let staircase = Loop::new(
            &lat,
            &[
                (0, Sign::Plus),
                (1, Sign::Plus),
                (2, Sign::Plus),
                (0, Sign::Minus),
                (1, Sign::Minus),
                (2, Sign::Minus),
            ],
        )
        .unwrap();

        let bases = [0usize, 5, 21, 63];
        let before: Vec<f64> = bases
            .iter()
            .map(|&site| action.loop_product(&lat, &config, site, &staircase))
            .collect();

        for &site in &[0usize, 1, 5, 6, 21, 22] {
            for link in lat.site_links(site) {
                config.poke(link, if config.peek(link) == up { down } else { up });
            }
        }

        for (&site, &was) in bases.iter().zip(&before) {
            assert_eq!(
                action.loop_product(&lat, &config, site, &staircase),
                was,
                "site {site}"
            );
        }
    }

    #[test]
    fn a_two_dimensional_loop_is_the_product_of_the_plaquettes_it_encloses() {
        // Stokes' theorem for `Z2`: every link strictly inside the rectangle is
        // shared by two of the enclosed plaquettes and appears twice in their
        // product, and a squared `±1` is 1, so all of them cancel and only the
        // boundary survives. This is the check that ties the loop walk to the
        // action's own plaquettes across sizes rather than at the unit square
        // alone.
        let lat = Lattice::new([6, 6]);
        let action = Z2Gauge::new(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(7);
        let config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        for (r, t) in [(1usize, 1usize), (2, 1), (1, 3), (2, 3), (3, 3)] {
            let path = Loop::rectangle(&lat, 0, r, 1, t).unwrap();
            for base in 0..lat.n_sites() {
                let mut enclosed: i32 = 1;
                for i in 0..r {
                    for j in 0..t {
                        let corner = lat.site_shift(lat.site_shift(base, 0, i), 1, j);
                        let p = lat.plaquette_index(lat.site_coords(corner), 0, 1);
                        enclosed *= lat
                            .plaquette_links(p)
                            .iter()
                            .map(|&link| decode(config.peek(link)))
                            .product::<i32>();
                    }
                }
                assert_eq!(
                    action.loop_product(&lat, &config, base, &path),
                    enclosed as f64,
                    "{r}x{t} at site {base}"
                );
            }
        }
    }

    #[test]
    fn a_cold_wilson_table_is_all_ones() {
        // Every link +1, so every rectangle's product is +1 and so is every
        // average of them, at every size.
        let lat = Lattice::new([6, 6, 6]);
        let action = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let table = action.wilson_rectangles(&lat, &config, 3);
        assert_eq!(table.len(), 4); // sides 0..=3
        for row in &table {
            assert_eq!(row.len(), 4);
            assert!(row.iter().all(|&w| w == 1.0), "{row:?}");
        }
    }

    #[test]
    fn the_unit_wilson_loop_is_the_mean_plaquette() {
        // The smallest rectangle is a plaquette, so its class average has to be
        // the plaquette sum over the plaquette count — the quantity the exact
        // two-dimensional result `tanh(β)` is stated for. Each plaquette is
        // reached twice, once per ordering of its two directions, which cancels
        // in the average.
        let mut rng = crate::rng::RandRng::seed_from_u64(11);
        let action = Z2Gauge::new(1.0);

        let lat = Lattice::new([6, 6]);
        let config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
        let mean = action.plaquette_sum(&lat, &config) / lat.n_plaquettes() as f64;
        assert_eq!(action.wilson_rectangles(&lat, &config, 2)[1][1], mean);

        // And in three dimensions, where the average also runs over the planes.
        let lat = Lattice::new([4, 4, 4]);
        let config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);
        let mean = action.plaquette_sum(&lat, &config) / lat.n_plaquettes() as f64;
        assert_eq!(action.wilson_rectangles(&lat, &config, 2)[1][1], mean);
    }

    #[test]
    fn the_wilson_table_is_symmetric_and_capped_at_half_the_shortest_extent() {
        // Both assignments of the sides to a plane's directions land in the same
        // entry, so `r` by `t` and `t` by `r` are the same average by
        // construction. The cap follows the shortest extent, since every plane
        // has to be able to supply every size.
        let lat = Lattice::new([4, 8]);
        let action = Z2Gauge::new(1.0);
        let mut rng = crate::rng::RandRng::seed_from_u64(3);
        let config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        // Asking for 10 gets 4 / 2 = 2, so sides 0..=2.
        let table = action.wilson_rectangles(&lat, &config, 10);
        assert_eq!(table.len(), 3);
        for (r, row) in table.iter().enumerate() {
            for (t, &value) in row.iter().enumerate() {
                assert_eq!(value, table[t][r], "{r}x{t}");
            }
        }
    }

    #[test]
    fn a_gauge_transformation_leaves_the_wilson_table_unchanged() {
        // The same invariance as for a single loop, now across every size and
        // placement at once: this is what makes the table a measurement rather
        // than an artifact of the gauge the chain wandered into.
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();
        let mut rng = crate::rng::RandRng::seed_from_u64(5);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        let before = action.wilson_rectangles(&lat, &config, 2);
        for site in (0..lat.n_sites()).step_by(3) {
            for link in lat.site_links(site) {
                config.poke(link, if config.peek(link) == up { down } else { up });
            }
        }
        assert_eq!(action.wilson_rectangles(&lat, &config, 2), before);
    }

    #[test]
    fn a_cold_polyakov_loop_is_one_in_every_direction() {
        let lat = Lattice::new([4, 5, 6]);
        let action = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        for dir in 0..3 {
            assert_eq!(action.polyakov_loop(&lat, &config, dir), 1.0);
        }
    }

    #[test]
    fn a_center_transformation_flips_the_polyakov_loop_but_not_the_energy() {
        // Multiplying every link along one direction in a single slice by -1 is
        // a symmetry of the energy — a plaquette meets that slice with two of
        // its links or with none, so the flips cancel either way — but a line
        // wrapping that direction crosses the slice exactly once and changes
        // sign. That is the symmetry the confined phase has and the deconfined
        // phase breaks, and it is why this average is kept signed: it is driven
        // to zero by a symmetry rather than being small by accident.
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();
        let mut rng = crate::rng::RandRng::seed_from_u64(13);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        let energy_before = action.energy(&lat, &config);
        let before: Vec<f64> = (0..3)
            .map(|d| action.polyakov_loop(&lat, &config, d))
            .collect();

        // The slice at coordinate 2 along direction 0, links along direction 0.
        for site in (0..lat.n_sites()).filter(|&s| lat.site_coords(s)[0] == 2) {
            let link = lat.link_index(lat.site_coords(site), 0);
            config.poke(link, if config.peek(link) == up { down } else { up });
        }

        assert_eq!(action.energy(&lat, &config), energy_before);
        assert_eq!(action.polyakov_loop(&lat, &config, 0), -before[0]);
        // The other directions never cross the flipped links.
        assert_eq!(action.polyakov_loop(&lat, &config, 1), before[1]);
        assert_eq!(action.polyakov_loop(&lat, &config, 2), before[2]);
    }

    #[test]
    fn a_gauge_transformation_leaves_the_polyakov_loop_unchanged() {
        let lat = Lattice::new([4, 4, 4]);
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();
        let mut rng = crate::rng::RandRng::seed_from_u64(17);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        let before: Vec<f64> = (0..3)
            .map(|d| action.polyakov_loop(&lat, &config, d))
            .collect();
        for site in (0..lat.n_sites()).step_by(5) {
            for link in lat.site_links(site) {
                config.poke(link, if config.peek(link) == up { down } else { up });
            }
        }
        for (dir, &was) in before.iter().enumerate() {
            assert_eq!(
                action.polyakov_loop(&lat, &config, dir),
                was,
                "direction {dir}"
            );
        }
    }

    #[test]
    fn two_dimensional_wilson_rectangles_follow_the_exact_area_law() {
        // Two-dimensional `Z2` is solvable: fixing the gauge leaves one free
        // variable per plaquette, so the plaquettes are independent, each
        // averages to `tanh(β)`, and a rectangle — being the product of the
        // plaquettes it encloses — averages to `tanh(β)` raised to its area.
        // A chain has to land on that, and the fixed seed makes it reproducible.
        use crate::chain::Chain;
        use crate::updater::Metropolis;

        let lat = Lattice::new([8, 8]);
        let action = Z2Gauge::new(1.0);
        let updater = Metropolis;
        let beta = 0.5;
        let samples = 800;

        let mut rng = crate::rng::RandRng::seed_from_u64(20_240_728);
        let mut config = Configuration::<2>::hot(&lat, Cell::Link, &mut rng);

        let mut sums = [[0.0f64; 3]; 3];
        {
            let mut chain = Chain::new(&mut config, &lat, &action, &updater, beta, &mut rng, 5);
            for _ in 0..100 {
                chain.next(); // burn in
            }
            for _ in 0..samples {
                let c = chain.next().unwrap();
                let table = action.wilson_rectangles(&lat, &c, 2);
                for (r, row) in sums.iter_mut().enumerate().skip(1) {
                    for (t, entry) in row.iter_mut().enumerate().skip(1) {
                        *entry += table[r][t];
                    }
                }
            }
        }

        for (r, row) in sums.iter().enumerate().skip(1) {
            for (t, &sum) in row.iter().enumerate().skip(1) {
                let measured = sum / samples as f64;
                let exact = beta.tanh().powi((r * t) as i32);
                assert!(
                    (measured - exact).abs() < 0.02,
                    "W({r},{t}) = {measured}, expected {exact}"
                );
            }
        }
    }

    #[test]
    fn energy_delta_is_zero_for_the_current_state() {
        let lat = Lattice::new([4, 4]);
        let action = Ising::new(1.0, 0.5);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        let same = config.peek(7);
        assert_eq!(action.energy_delta(&lat, &config, 7, same), 0.0);
    }

    /// Flip each of `probed` in turn, on a link field first disordered at
    /// `flipped`, and check the `O(1)` delta against a full rescan either side.
    /// Generic over `D` so the same body serves both dimensions.
    fn check_gauge_delta<const D: usize>(lat: &Lattice<D>, flipped: &[usize], probed: &[usize]) {
        let action = Z2Gauge::new(1.0);
        let up = State::new(0).unwrap();
        let down = State::new(1).unwrap();

        let mut config = Configuration::<2>::cold(lat, Cell::Link);
        for &link in flipped {
            config.poke(link, down);
        }

        for &link in probed {
            let proposed = if config.peek(link) == up { down } else { up };
            let before = action.energy(lat, &config);
            let delta = action.energy_delta(lat, &config, link, proposed);

            let mut after = config.clone();
            after.poke(link, proposed);
            assert_eq!(delta, action.energy(lat, &after) - before, "link {link}");
        }
    }

    #[test]
    fn gauge_energy_delta_matches_from_scratch_difference() {
        // j = 1.0 is exactly representable and the sums are integer, so the two
        // sides agree bit-for-bit and bare `==` is legitimate here. The two
        // dimensions exercise both staple-group counts: 2 * (D − 1) is two
        // groups per link in 2D and four in 3D.
        let lat = Lattice::new([4, 4]);
        check_gauge_delta(&lat, &[3, 8, 17, 24], &[0, 3, 8, 17, 24, 31]);

        let lat = Lattice::new([3, 3, 3]);
        check_gauge_delta(&lat, &[0, 5, 13, 20, 44, 61], &[0, 1, 5, 13, 40, 61, 80]);
    }

    #[test]
    fn gauge_energy_delta_is_zero_for_the_current_state() {
        let lat = Lattice::new([3, 3, 3]);
        let action = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        let same = config.peek(7);
        assert_eq!(action.energy_delta(&lat, &config, 7, same), 0.0);
    }

    /// The dimension floor holds in release, unlike the cell-kind guards above.
    ///
    /// Deliberately not gated on `debug_assertions`: `D` is a compile-time
    /// constant, so a real `assert!` folds away to nothing and there is no
    /// reason to make this one debug-only. Below two dimensions a lattice has no
    /// plaquettes, so the energy is zero for every configuration and a chain
    /// accepts every flip — noise that looks like a sampled theory. A guard that
    /// disappeared in release would hide that exactly where it matters.
    #[test]
    #[should_panic(expected = "which need at least two dimensions")]
    fn the_gauge_action_refuses_one_dimension_in_any_profile() {
        let lat = Lattice::new([8]);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        Z2Gauge::new(1.0).energy(&lat, &config);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "gauge variables live on links")]
    fn the_gauge_action_rejects_a_site_field() {
        // Without the guard this surfaces further in as an out-of-range read,
        // whose message names an index rather than the actual fault.
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Site);
        Z2Gauge::new(1.0).energy(&lat, &config);
    }

    #[test]
    fn any_action_forwards_to_the_wrapped_action() {
        // Same lattice, same field, same numbers on both seams: the enum adds a
        // branch and nothing else. One lattice serves both variants, since the
        // fields differ only in length — links are twice the sites in 2D.
        let lat = Lattice::new([4, 4]);
        let down = State::new(1).unwrap();

        let ising = Ising::new(1.0, 0.5);
        let mut sites = Configuration::<2>::cold(&lat, Cell::Site);
        sites.poke(5, down);
        let any = AnyAction::Ising(ising);
        assert_eq!(any.energy(&lat, &sites), ising.energy(&lat, &sites));
        assert_eq!(
            any.energy_delta(&lat, &sites, 6, down),
            ising.energy_delta(&lat, &sites, 6, down)
        );

        let gauge = Z2Gauge::new(1.0);
        let mut links = Configuration::<2>::cold(&lat, Cell::Link);
        links.poke(5, down);
        let any = AnyAction::Z2Gauge(gauge);
        assert_eq!(any.energy(&lat, &links), gauge.energy(&lat, &links));
        assert_eq!(
            any.energy_delta(&lat, &links, 6, down),
            gauge.energy_delta(&lat, &links, 6, down)
        );
    }

    #[test]
    fn any_action_reports_the_cell_its_field_must_sit_on() {
        assert_eq!(AnyAction::Ising(Ising::new(1.0, 0.0)).cell(), Cell::Site);
        assert_eq!(AnyAction::Z2Gauge(Z2Gauge::new(1.0)).cell(), Cell::Link);
    }

    // Debug-only, like its gauge twin below: the guard it asserts on is a
    // `debug_assert!`, which release builds compile out.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "Ising spins live on sites")]
    fn the_ising_action_rejects_a_link_field() {
        let lat = Lattice::new([4, 4]);
        let config = Configuration::<2>::cold(&lat, Cell::Link);
        Ising::new(1.0, 0.0).energy(&lat, &config);
    }
}
