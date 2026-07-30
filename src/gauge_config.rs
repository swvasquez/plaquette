//! Gauge run configuration: the file that specifies a single Z2 gauge run's
//! parameters.
//!
//! [`GaugeRunConfig`] is to [`Z2Gauge`] what
//! [`IsingRunConfig`](crate::ising_config::IsingRunConfig) is to
//! [`Ising`](crate::model::Ising), and it is a separate type rather than a
//! widened one because the two models take different parameters: there is no
//! external field here, and no field can be added — the plaquette energy is
//! invariant under flipping every link that touches a site, and a term reading
//! individual links would destroy exactly that symmetry. A schema covering both
//! models would carry an `h` that a gauge run must leave unset, which is a hole
//! rather than a generalization.
//!
//! What the two do share is the vocabulary in [`config`](crate::config) —
//! [`Start`], [`UpdaterKind`], [`ConfigError`] — so the driver fields, the
//! reproducibility fields, and the load/parse/validate/round-trip contract are
//! written the same way on both sides and read the same way from a file.
//!
//! The dimension is fixed at `D = 3`, the same way the Ising schema is nailed to
//! `D = 2` and for the same reason: a file is read at runtime, so the loader has
//! to commit to a concrete dimension. Three is the interesting one for `Z2` —
//! the two-dimensional theory is exactly solvable and confines at every coupling,
//! so it has no transition to find, while three dimensions has one. Dispatching
//! to a general `Lattice<D>` from a variable-length shape is deferred.

use crate::config::{ConfigError, Start, UpdaterKind, deserialize_shape};
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::model::Z2Gauge;
use crate::rng::RandRng;
use serde::{Deserialize, Serialize};

/// The dimension every gauge run is fixed at (see the module docs).
const D: usize = 3;

/// A single gauge run's parameters in serializable form: everything needed to
/// produce a run's configurations, and nothing about what is later measured from
/// them.
///
/// The specification is *complete* — seed, start, and update algorithm included
/// — so feeding the same `GaugeRunConfig` back in produces the same
/// configurations. Wilson loop sizes and which direction to read a Polyakov loop
/// along are deliberately absent: they only affect what is computed *from* a
/// configuration, and a stored configuration can be measured any number of ways
/// after the fact.
///
/// Unknown keys are rejected rather than ignored, which also gives the schema a
/// second job: an Ising file handed to this parser fails on `h` and on its
/// two-element `shape`, rather than running a gauge theory at whatever the file
/// happened to say.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeRunConfig {
    // --- physics ---
    /// Per-axis lattice extents `[L_0, L_1, L_2]`. Fixed length 3 (see module
    /// docs), and a file giving any other length is refused rather than truncated:
    /// a too-long `shape` would otherwise drop its trailing extents in silence and
    /// run a smaller lattice than the file names.
    #[serde(deserialize_with = "deserialize_shape")]
    pub shape: [usize; D],
    /// Plaquette coupling `J`. There is no `h` to go with it — see the module
    /// docs.
    pub j: f64,
    /// Inverse temperature `beta = 1 / T`.
    pub beta: f64,

    // --- driver controls ---
    /// Which update algorithm advances the chain; defaults to
    /// [`UpdaterKind::Metropolis`], and [`validate`](GaugeRunConfig::validate)
    /// rejects the others. A run parameter rather than a caller's choice,
    /// because two runs of the same physics under different algorithms are
    /// different runs.
    #[serde(default = "default_updater")]
    pub updater: UpdaterKind,
    /// Warmup sweeps run and discarded before any sample.
    pub thermalize: usize,
    /// Decorrelation sweeps between recorded samples.
    pub sweeps_between: usize,
    /// Number of samples to record — the size of the resulting ensemble.
    pub n_samples: usize,

    // --- reproducibility ---
    /// Seed for the run's generator, so the whole run replays deterministically.
    pub seed: u64,
    /// Start configuration; defaults to [`Start::Cold`]. Sits with the seed
    /// because a [`Start::Hot`] start is *drawn from* the seeded generator (see
    /// [`build`](GaugeRunConfig::build)).
    #[serde(default = "default_start")]
    pub start: Start,

    // --- provenance ---
    /// A free-form human note about the run, deliberately not part of its
    /// identity. It has to be a *field* rather than a TOML comment because serde
    /// drops comments on the way back out, so a comment would not survive into
    /// the dumped record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

fn default_start() -> Start {
    Start::Cold
}

fn default_updater() -> UpdaterKind {
    UpdaterKind::Metropolis
}

impl GaugeRunConfig {
    /// Read and validate a config from a TOML file at `path`. Validation runs as
    /// part of loading, so a malformed run fails here rather than partway
    /// through a long chain.
    ///
    /// This is the gauge entry point, and naming it is how a caller says which
    /// model the file describes: there is no discriminant key in either schema,
    /// because nothing reads a config file without already knowing what it is
    /// running.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Parse and validate a config from TOML text — the filesystem-free half of
    /// [`load`](GaugeRunConfig::load), with the same guarantees.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: GaugeRunConfig = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Check that the values describe a runnable gauge run.
    ///
    /// Rejects what would otherwise panic or produce nothing — a zero extent, a
    /// non-positive or non-finite `beta`, a non-finite coupling, a run recording
    /// no samples — and one rule the Ising side does not have: the updater must
    /// be [`UpdaterKind::Metropolis`]. The checkerboard schedule colors *sites*
    /// by the parity of their coordinates and the GPU backend is built around
    /// that same site field, so neither has a meaning for variables sitting on
    /// links; a gauge run under either would not be a slower or faster version
    /// of the same physics, it would be wrong. The rule lives here rather than
    /// in a narrower enum so the message can name what was asked for.
    ///
    /// `thermalize` and `sweeps_between` may be zero — no warmup and no
    /// decorrelation gap are unusual but legitimate.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if let Some(axis) = self.shape.iter().position(|&l| l == 0) {
            return Err(ConfigError::Invalid(format!(
                "every lattice extent must be positive, but shape{:?} is 0 on axis {axis}",
                self.shape
            )));
        }
        // `is_finite` first, so the comparison below is NaN-free.
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(ConfigError::Invalid(format!(
                "beta must be positive and finite, got {}",
                self.beta
            )));
        }
        if !self.j.is_finite() {
            return Err(ConfigError::Invalid(format!(
                "j must be finite, got {}",
                self.j
            )));
        }
        if self.updater != UpdaterKind::Metropolis {
            return Err(ConfigError::Invalid(format!(
                "a z2 gauge run updates links, which only the metropolis updater does, \
                 but the config asks for {:?}",
                self.updater
            )));
        }
        if self.n_samples == 0 {
            return Err(ConfigError::Invalid(
                "n_samples must be positive, or the run records nothing".to_string(),
            ));
        }
        Ok(())
    }

    /// Assemble the constructed pieces a gauge run needs, in the order that
    /// makes the run reproducible from [`seed`](GaugeRunConfig::seed) alone.
    ///
    /// Returns `(lattice, model, rng, config, beta)` — the arguments
    /// [`Chain::new`](crate::chain::Chain::new) takes, minus the updater, which
    /// is a plain unit value rather than something to construct. It does not
    /// return a `Chain`, because the chain borrows these pieces and the caller
    /// must own them first.
    ///
    /// The ordering is why this lives in one place: the generator is seeded
    /// first, and a [`Start::Hot`] configuration draws from **that same
    /// generator**, which then carries on into the chain. A caller assembling
    /// this by hand could seed one generator and initialize the config from
    /// another, leaving a run that looks reproducible and quietly is not.
    ///
    /// The configuration sits on [`Cell::Link`], not on the sites the Ising side
    /// uses. That is not a parameter: it is where the gauge variables live, and
    /// the model's energy asserts it.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid. [`load`](GaugeRunConfig::load) and
    /// [`parse`](GaugeRunConfig::parse) validate, so this can only fire on a
    /// `GaugeRunConfig` built by hand that skipped
    /// [`validate`](GaugeRunConfig::validate).
    pub fn build(&self) -> (Lattice<D>, Z2Gauge, RandRng, Configuration<2>, f64) {
        self.validate()
            .expect("build called on an unvalidated config");

        let lattice = Lattice::new(self.shape);
        let model = Z2Gauge::new(self.j);
        let mut rng = RandRng::seed_from_u64(self.seed);
        // Drawn from `rng` *after* seeding and *before* the chain uses it, so the
        // whole run — start configuration included — replays from `seed`.
        let config = match self.start {
            Start::Cold => Configuration::cold(&lattice, Cell::Link),
            Start::Hot => Configuration::hot(&lattice, Cell::Link, &mut rng),
        };

        (lattice, model, rng, config, self.beta)
    }

    /// Render the config back to TOML — the inverse of
    /// [`parse`](GaugeRunConfig::parse), so a run can dump exactly the config it
    /// ran and feeding that text back reproduces it.
    pub fn to_toml(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A fully-specified config used as a fixture across the tests.
    fn sample_config() -> GaugeRunConfig {
        GaugeRunConfig {
            shape: [8, 8, 8],
            j: 1.0,
            beta: 0.75,
            updater: UpdaterKind::Metropolis,
            thermalize: 1000,
            sweeps_between: 10,
            n_samples: 500,
            seed: 42,
            start: Start::Hot,
            description: Some("confinement check".to_string()),
        }
    }

    /// Assert a config fails validation, and hand back the message.
    fn invalid_message(config: &GaugeRunConfig) -> String {
        match config.validate() {
            Err(ConfigError::Invalid(msg)) => msg,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let config = sample_config();
        let text = config.to_toml().unwrap();
        let parsed = GaugeRunConfig::parse(&text).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn parses_a_known_toml_document() {
        // Leading whitespace before a key is ignored by TOML, so the document
        // can be indented with the test rather than jammed against column 0.
        let text = r#"
            shape = [6, 6, 6]
            j = 1.0
            beta = 0.75
            updater = "metropolis"
            thermalize = 2000
            sweeps_between = 20
            n_samples = 1000
            seed = 12345
            start = "hot"
            description = "near the transition"
        "#;
        let config = GaugeRunConfig::parse(text).unwrap();
        assert_eq!(config.shape, [6, 6, 6]);
        assert_eq!(config.j, 1.0);
        assert_eq!(config.beta, 0.75);
        assert_eq!(config.thermalize, 2000);
        assert_eq!(config.sweeps_between, 20);
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.start, Start::Hot);
        assert_eq!(config.updater, UpdaterKind::Metropolis);
        assert_eq!(config.description.as_deref(), Some("near the transition"));
    }

    #[test]
    fn load_reads_a_file_from_disk() {
        let config = sample_config();
        let path = std::env::temp_dir().join("plaquette_gauge_config_load_test.toml");
        std::fs::write(&path, config.to_toml().unwrap()).unwrap();

        let loaded = GaugeRunConfig::load(&path).unwrap();
        assert_eq!(config, loaded);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn description_is_omitted_from_dumps_when_absent() {
        let mut config = sample_config();
        config.description = None;
        let text = config.to_toml().unwrap();
        assert!(!text.contains("description"));
        assert_eq!(GaugeRunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn optional_fields_default_when_omitted() {
        let text = r#"
            shape = [4, 4, 4]
            j = 1.0
            beta = 0.5
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        let config = GaugeRunConfig::parse(text).unwrap();
        assert_eq!(config.start, Start::Cold);
        assert_eq!(config.updater, UpdaterKind::Metropolis);
        assert_eq!(config.description, None);
    }

    #[test]
    fn validate_accepts_the_sample_config() {
        assert!(sample_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_zero_extent() {
        let mut config = sample_config();
        config.shape = [8, 0, 8];
        assert!(invalid_message(&config).contains("extent"));
    }

    #[test]
    fn validate_rejects_non_positive_beta() {
        let mut config = sample_config();
        config.beta = 0.0;
        assert!(invalid_message(&config).contains("beta"));

        config.beta = -0.5;
        assert!(invalid_message(&config).contains("beta"));
    }

    #[test]
    fn validate_rejects_zero_samples() {
        let mut config = sample_config();
        config.n_samples = 0;
        assert!(invalid_message(&config).contains("n_samples"));
    }

    #[test]
    fn validate_allows_zero_thermalize_and_gap() {
        let mut config = sample_config();
        config.thermalize = 0;
        config.sweeps_between = 0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn validate_rejects_every_updater_but_metropolis() {
        // The rule the Ising schema does not have: both other schedules are
        // built around a site field, so neither is a valid way to run this.
        for kind in [UpdaterKind::Checkerboard, UpdaterKind::GpuCheckerboard] {
            let mut config = sample_config();
            config.updater = kind;
            let message = invalid_message(&config);
            assert!(message.contains("metropolis"), "{message}");
            assert!(message.contains(&format!("{kind:?}")), "{message}");
        }
    }

    #[test]
    fn parse_rejects_an_invalid_config() {
        // Validation runs as part of parsing, so a bad file fails at load time —
        // including the updater rule, which a parse alone would let through.
        let text = r#"
            shape = [4, 4, 4]
            j = 1.0
            beta = 0.5
            updater = "gpu_checkerboard"
            thermalize = 10
            sweeps_between = 1
            n_samples = 10
            seed = 1
        "#;
        assert!(matches!(
            GaugeRunConfig::parse(text),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // A typo'd *optional* field is the dangerous case: it parses cleanly
        // without `deny_unknown_fields`, and the run silently ignores it.
        let text = r#"
            shape = [4, 4, 4]
            j = 1.0
            beta = 0.5
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
            startt = "hot"
        "#;
        assert!(matches!(
            GaugeRunConfig::parse(text),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn a_shape_of_the_wrong_length_is_refused_rather_than_truncated() {
        // Too *long* is the dangerous direction, and it does not fail on its own:
        // serde's array visitor stops after three entries and the TOML
        // deserializer says nothing about the fourth, so without
        // `deserialize_shape` a four-dimensional file would quietly run as a
        // three-dimensional one. Too short fails either way.
        let text = r#"
            shape = [4, 4, 4, 4]
            j = 1.0
            beta = 0.75
            thermalize = 10
            sweeps_between = 1
            n_samples = 10
            seed = 1
        "#;
        assert!(matches!(
            GaugeRunConfig::parse(text),
            Err(ConfigError::Parse(_))
        ));

        let short = text.replace("[4, 4, 4, 4]", "[4, 4]");
        assert!(matches!(
            GaugeRunConfig::parse(&short),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn an_ising_file_is_refused_rather_than_run_as_a_gauge_theory() {
        // The two schemas have no discriminant key, so this is what stops a file
        // meant for the other model. Both of its marks are checked separately,
        // since either alone has to be enough: an Ising file without an `h` is
        // still an Ising file, and one that happened to name three extents would
        // still have a field this model cannot price.
        let ising = r#"
            shape = [8, 8]
            j = 1.0
            h = 0.25
            beta = 0.44
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        assert!(matches!(
            GaugeRunConfig::parse(ising),
            Err(ConfigError::Parse(_))
        ));

        // The two-element shape alone, with the field dropped.
        let no_field = ising.replace("h = 0.25\n", "");
        assert!(matches!(
            GaugeRunConfig::parse(&no_field),
            Err(ConfigError::Parse(_))
        ));

        // And `h` alone, with the shape corrected.
        let three_axes = ising.replace("[8, 8]", "[8, 8, 8]");
        assert!(matches!(
            GaugeRunConfig::parse(&three_axes),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn build_produces_pieces_matching_the_file() {
        let text = r#"
            shape = [4, 6, 8]
            j = 1.5
            beta = 0.75
            thermalize = 100
            sweeps_between = 5
            n_samples = 50
            seed = 99
            start = "cold"
        "#;
        let config = GaugeRunConfig::parse(text).unwrap();
        let (lattice, model, _rng, start, beta) = config.build();

        assert_eq!(lattice.shape(), [4, 6, 8]);
        assert_eq!(lattice.n_sites(), 192);
        assert_eq!(model, Z2Gauge::new(1.5));
        assert_eq!(beta, 0.75);
        // The field sits on links, three per site in three dimensions.
        assert_eq!(start.cell(), Cell::Link);
        assert_eq!(start.n_vars(), 576);
        // `start = "cold"` must give the aligned ground state.
        assert_eq!(start, Configuration::<2>::cold(&lattice, Cell::Link));
    }

    #[test]
    fn build_honors_a_hot_start() {
        let mut config = sample_config();
        config.start = Start::Hot;
        let (lattice, _, _, start, _) = config.build();

        // A hot start on 1536 links is aligned with vanishing probability, so
        // this distinguishes it from cold without being flaky.
        assert_ne!(start, Configuration::<2>::cold(&lattice, Cell::Link));
    }

    #[test]
    fn build_is_reproducible_from_the_seed() {
        // The whole point of assembling in one place: same config in, same run
        // out — start configuration *and* the generator state the chain inherits.
        let mut config = sample_config();
        config.start = Start::Hot;

        let (_, _, mut rng_a, start_a, _) = config.build();
        let (_, _, mut rng_b, start_b, _) = config.build();

        assert_eq!(start_a, start_b);
        for _ in 0..16 {
            assert_eq!(rng_a.next_f64(), rng_b.next_f64());
        }
    }

    #[test]
    fn hot_start_consumes_the_same_generator_the_chain_inherits() {
        // A hot start draws from `rng`, so the stream the chain continues with
        // must be *past* those draws — not a fresh seed. Otherwise the config
        // and the chain would silently share randomness.
        let mut config = sample_config();
        config.start = Start::Hot;
        let (_, _, mut rng, _, _) = config.build();

        let mut fresh = RandRng::seed_from_u64(config.seed);
        assert_ne!(rng.next_f64(), fresh.next_f64());
    }
}
