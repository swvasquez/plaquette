//! The Z2 lattice gauge theory: variables `±1` on the links, coupled through
//! plaquettes, and the loop observables that are the only things it can
//! measure.
//!
//! [`Z2Gauge`] owns the Wilson action specialized to `Z2` and the value-semantic
//! observables built on closed loops: the plaquette sum, Wilson rectangles, and
//! the Polyakov loop. [`gauge_measure`] composes the primaries into a
//! per-config [`GaugeSample`]; [`wilson_rectangles`] and [`polyakov_loop`] wrap
//! the expensive loop measurements a run opts into. The run-config schema, the
//! CPU/GPU samplers, and the GPU backend sit in the submodules, re-exported
//! here.

pub mod gpu;
pub mod run_config;
pub mod sampler;

pub use gpu::GpuGaugeChain;
pub use run_config::GaugeRunConfig;
pub use sampler::{AnyGaugeChain, GaugeSampler};

use super::decode;
use crate::action::Action;
use crate::configuration::{Cell, Configuration};
use crate::lattice::{Lattice, Loop, Sign};
use crate::state::State;

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
/// change from [`Ising`](crate::models::ising::Ising) is the unit of interaction: a four-link product around
/// a face in place of a two-spin product across a bond. A plaquette needs two
/// directions to span, so the model is empty below `D = 2` — the lattice
/// reports no plaquettes there and every configuration would score zero, which
/// the entry points reject rather than sample.
///
/// There is deliberately no external-field term to match [`Ising`](crate::models::ising::Ising)'s `h`. The
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
    /// [`MIN_DIMENSION`](crate::models::gauge::run_config::MIN_DIMENSION) and
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
    /// The sign is kept for the same reason [`Ising::magnetization`](crate::models::ising::Ising::magnetization) keeps it:
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
    /// gives, matching how [`correlator`](crate::models::ising::Ising::correlator)
    /// keeps `C_0 = 1`.
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

/// The per-config measurement record for the gauge model, the counterpart of
/// [`Sample`](crate::models::ising::Sample).
///
/// It has no magnetization, and no field can be added to give it one. The gauge
/// symmetry is local, so for every configuration there is an equally probable
/// one with any single link flipped, and the average of a lone link variable is
/// therefore exactly zero rather than merely small — Elitzur's theorem. A field
/// summing the links would produce a series fluctuating around zero forever,
/// which is why the confinement transition is read off closed loops instead
/// (see [`WilsonRectangles`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaugeSample {
    /// Total energy `H` of the configuration (from [`Action::energy`]).
    pub energy: f64,
    /// The signed plaquette sum `sum_□ σ_□` (from [`Z2Gauge::plaquette_sum`]).
    ///
    /// Redundant with `energy`, which is `-j` times it, and kept anyway: the
    /// mean plaquette is what the literature reports and what the exact
    /// two-dimensional result `tanh(β)` is stated for, while `energy` is what
    /// [`specific_heat`](crate::statistics::specific_heat) consumes. Neither
    /// caller should have to undo the other's scaling, and at `j = 0` only this
    /// one survives.
    pub plaquette_sum: f64,
}

/// Measure one `config` of the gauge `model` on `lattice` into a
/// [`GaugeSample`].
pub fn gauge_measure<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
) -> GaugeSample {
    GaugeSample {
        energy: model.energy(lattice, config),
        plaquette_sum: model.plaquette_sum(lattice, config),
    }
}

/// The per-config table of rectangular Wilson loops: `per_size[r][t]` is the
/// product of the link variables around an `r`-by-`t` rectangle, averaged over
/// every placement of that rectangle on the lattice.
///
/// Named for rectangles rather than for loops in general, because that is what
/// it enumerates. A Wilson loop is defined for *any* closed path, and
/// [`Z2Gauge::loop_product`] is that definition; this measures one particular
/// family of paths, the one whose average answers the confinement question.
///
/// This is to [`GaugeSample`] what [`Correlator`](crate::observables::Correlator) is to [`Sample`](crate::models::ising::Sample) — the
/// separate, expensive one. It costs the rectangle count times the lattice
/// volume times the perimeter and it allocates, so runs that don't want it don't
/// pay for it.
///
/// The raw per-config estimator only. What the table is *for* is the comparison
/// downstream: if the chain average falls off like the exponential of `r * t`
/// the theory confines, and if it falls off with the perimeter instead it does
/// not. That fit belongs to statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct WilsonRectangles {
    /// `per_size[r][t]` for both sides running `0..=max_side`, symmetric in the
    /// two indices, with row and column `0` the trivial `1.0`.
    pub per_size: Vec<Vec<f64>>,
}

/// Measure the per-config [`WilsonRectangles`] table of `config` on `lattice`, with
/// both side lengths running up to `max_side`.
///
/// `max_side` is capped at half the smallest extent, past which a rectangle
/// wraps the torus and stops measuring what it is meant to.
pub fn wilson_rectangles<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
    max_side: usize,
) -> WilsonRectangles {
    WilsonRectangles {
        per_size: model.wilson_rectangles(lattice, config, max_side),
    }
}

/// Measure the per-config Polyakov loop of `config` along `dir` — the product
/// down a line wrapping that direction, averaged over the lines and kept signed.
///
/// One number rather than a record, so there is nothing to bundle; it is a free
/// function here rather than a field on [`GaugeSample`] because it costs a walk
/// of the whole lattice and only a run studying deconfinement wants it. Which
/// direction wraps is the caller's to say, since nothing in the geometry marks
/// one out as time.
pub fn polyakov_loop<const D: usize>(
    model: &Z2Gauge,
    lattice: &Lattice<D>,
    config: &Configuration<2>,
    dir: usize,
) -> f64 {
    model.polyakov_loop(lattice, config, dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::deltas_match_from_scratch;

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
    fn gauge_measure_bundles_energy_and_plaquette_sum() {
        // Cold 4x4x4 with j = 2: every plaquette product is +1, so the sum is
        // the plaquette count (3 planes * 64 sites) and E = -j times it.
        let lat = Lattice::new([4, 4, 4]);
        let model = Z2Gauge::new(2.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let sample = gauge_measure(&model, &lat, &config);
        assert_eq!(sample.plaquette_sum, 192.0);
        assert_eq!(sample.energy, -384.0);
        assert_eq!(sample.energy, -2.0 * sample.plaquette_sum);
    }

    #[test]
    fn the_plaquette_sum_survives_a_zero_coupling() {
        // The reason both fields are kept: at j = 0 the energy is identically
        // zero and carries nothing, while the plaquette sum still measures the
        // configuration.
        let lat = Lattice::new([4, 4]);
        let model = Z2Gauge::new(0.0);
        let mut config = Configuration::<2>::cold(&lat, Cell::Link);
        config.poke(0, State::new(1).unwrap());

        let sample = gauge_measure(&model, &lat, &config);
        assert_eq!(sample.energy, 0.0);
        // Link 0 sits on two of the sixteen plaquettes, flipping both to -1.
        assert_eq!(sample.plaquette_sum, 12.0);
    }

    #[test]
    fn wilson_rectangles_wraps_the_model_measurement() {
        // Cold gives 1.0 at every size, and the record carries the model's own
        // table through unchanged.
        let lat = Lattice::new([6, 6]);
        let model = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        let w = wilson_rectangles(&model, &lat, &config, 3);
        assert_eq!(w.per_size, model.wilson_rectangles(&lat, &config, 3));
        assert_eq!(w.per_size.len(), 4); // sides 0..=3
        assert!(w.per_size.iter().all(|row| row.iter().all(|&v| v == 1.0)));
    }

    #[test]
    fn polyakov_loop_wraps_the_model_measurement() {
        // The one observables free function that lacked a wrapper test: it must
        // carry the model's own Polyakov value through unchanged. A cold link
        // field has every loop equal to 1, the torus-winding Polyakov line
        // included.
        let lat = Lattice::new([4, 4, 4]);
        let model = Z2Gauge::new(1.0);
        let config = Configuration::<2>::cold(&lat, Cell::Link);

        for dir in 0..3 {
            assert_eq!(
                polyakov_loop(&model, &lat, &config, dir),
                model.polyakov_loop(&lat, &config, dir)
            );
        }
        assert_eq!(polyakov_loop(&model, &lat, &config, 0), 1.0);
    }
}
