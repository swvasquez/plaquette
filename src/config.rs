//! Run configuration: the file that specifies a single run's parameters.
//!
//! [`RunConfig`] is the serializable form of the constructor arguments the
//! driver's pieces already take. It deserializes from TOML and assembles those
//! pieces — lattice, model, seeded generator, start configuration, `beta` — so
//! that serde and I/O stay confined to this module and the physics core never
//! learns a file was involved.
//!
//! The same struct round-trips, configuring a run *and* recording one: it
//! derives `Serialize` too, so a run can dump the exact config it ran.
//!
//! Its scope is a single `(L, beta, h)` point; a later `Scan` component would
//! wrap this rather than the reverse. The dimension is fixed at `D = 2` because
//! a file is read at runtime and the loader must commit to a concrete dimension.
//! Dispatching to a general `Lattice<D>` from a variable-length shape is
//! deferred.

use crate::configuration::Configuration;
use crate::lattice::Lattice;
use crate::model::Ising;
use crate::rng::RandRng;
use serde::{Deserialize, Serialize};

/// Which start configuration a run begins from, before thermalization —
/// mirroring the two [`Configuration`] initializers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Start {
    /// All spins aligned (`Configuration::cold`).
    Cold,
    /// Each spin independently random (`Configuration::hot`).
    Hot,
}

/// Which update algorithm a run uses to advance the chain.
///
/// The *serializable choice of* an updater, not an updater itself — hence the
/// name, which also avoids colliding with the
/// [`Updater`](crate::updater::Updater) trait. Being a closed set is what makes
/// it recordable, unlike an arbitrary caller-supplied implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UpdaterKind {
    /// Single-spin-flip Metropolis ([`Metropolis`](crate::updater::Metropolis)).
    Metropolis,
    /// Single-spin-flip Metropolis under a checkerboard schedule
    /// ([`Checkerboard`](crate::updater::Checkerboard)).
    Checkerboard,
}

/// A single run's parameters in serializable form: everything needed to produce
/// a run's configurations, and nothing about what is later measured from them.
///
/// The specification is *complete* — seed, start, and update algorithm included
/// — so feeding the same `RunConfig` back in produces the same configurations.
/// Nothing that only affects downstream analysis belongs here, since the stored
/// configurations can be measured any number of ways after the fact.
///
/// Unknown keys are rejected rather than ignored: a typo in an *optional* field
/// (`updatr = "metropolis"`) would otherwise parse cleanly and silently not do
/// what the file says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunConfig {
    // --- physics ---
    /// Per-axis lattice extents `[L_0, L_1]`. Fixed length 2 (see module docs).
    pub shape: [usize; 2],
    /// Nearest-neighbour coupling `J`.
    pub j: f64,
    /// Uniform external field `h`; defaults to `0.0` when omitted — the
    /// field-free model, which is the standard case. `j` stays required, since
    /// omitting a coupling does not obviously mean "unit coupling", and
    /// normalizing it silently would be a physics choice made on the run's
    /// behalf.
    #[serde(default)]
    pub h: f64,
    /// Inverse temperature `beta = 1 / T`.
    pub beta: f64,

    // --- driver controls ---
    /// Which update algorithm advances the chain; defaults to
    /// [`UpdaterKind::Metropolis`]. A run parameter rather than a caller's
    /// choice, because two runs of the same physics under different algorithms
    /// are different runs.
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
    /// [`build`](RunConfig::build)).
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

/// What can go wrong turning a file into a usable [`RunConfig`]. The stages are
/// kept distinct so a failure says *where* it happened.
#[derive(Debug)]
pub enum ConfigError {
    /// The config file could not be read.
    Io(std::io::Error),
    /// The text was not valid TOML for this struct.
    Parse(toml::de::Error),
    /// The config serialized cleanly but could not be rendered back to TOML.
    Serialize(toml::ser::Error),
    /// The values parsed but describe an unrunnable run.
    Invalid(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "could not read config file: {e}"),
            ConfigError::Parse(e) => write!(f, "could not parse config: {e}"),
            ConfigError::Serialize(e) => write!(f, "could not serialize config: {e}"),
            ConfigError::Invalid(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ConfigError::Io(e) => Some(e),
            ConfigError::Parse(e) => Some(e),
            ConfigError::Serialize(e) => Some(e),
            ConfigError::Invalid(_) => None,
        }
    }
}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

impl From<toml::de::Error> for ConfigError {
    fn from(e: toml::de::Error) -> Self {
        ConfigError::Parse(e)
    }
}

impl From<toml::ser::Error> for ConfigError {
    fn from(e: toml::ser::Error) -> Self {
        ConfigError::Serialize(e)
    }
}

impl RunConfig {
    /// Read and validate a config from a TOML file at `path`. Validation runs as
    /// part of loading, so a malformed run fails here rather than partway
    /// through a long chain.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Parse and validate a config from TOML text — the filesystem-free half of
    /// [`load`](RunConfig::load), with the same guarantees.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: RunConfig = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Check that the values describe a runnable run.
    ///
    /// Rejects what would otherwise panic or produce nothing: a zero extent, a
    /// non-positive or non-finite `beta`, a non-finite coupling or field, and a
    /// run that would record no samples. `thermalize` and `sweeps_between` may
    /// be zero — no warmup and no decorrelation gap are unusual but legitimate.
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
        if !self.h.is_finite() {
            return Err(ConfigError::Invalid(format!(
                "h must be finite, got {}",
                self.h
            )));
        }
        if self.n_samples == 0 {
            return Err(ConfigError::Invalid(
                "n_samples must be positive, or the run records nothing".to_string(),
            ));
        }
        Ok(())
    }

    /// Assemble the constructed pieces a run needs, in the order that makes the
    /// run reproducible from [`seed`](RunConfig::seed) alone.
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
    /// # Panics
    ///
    /// Panics if the config is invalid. [`load`](RunConfig::load) and
    /// [`parse`](RunConfig::parse) validate, so this can only fire on a
    /// `RunConfig` built by hand (as a future `Scan` would) that skipped
    /// [`validate`](RunConfig::validate).
    pub fn build(&self) -> (Lattice<2>, Ising, RandRng, Configuration<2>, f64) {
        self.validate()
            .expect("build called on an unvalidated config");

        let lattice = Lattice::new(self.shape);
        let model = Ising::new(self.j, self.h);
        let mut rng = RandRng::seed_from_u64(self.seed);
        // Drawn from `rng` *after* seeding and *before* the chain uses it, so the
        // whole run — start configuration included — replays from `seed`.
        let config = match self.start {
            Start::Cold => Configuration::cold(&lattice),
            Start::Hot => Configuration::hot(&lattice, &mut rng),
        };

        (lattice, model, rng, config, self.beta)
    }

    /// Render the config back to TOML — the inverse of
    /// [`parse`](RunConfig::parse), so a run can dump exactly the config it ran
    /// and feeding that text back reproduces it.
    pub fn to_toml(&self) -> Result<String, ConfigError> {
        Ok(toml::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    /// A fully-specified config used as a fixture across the tests.
    fn sample_config() -> RunConfig {
        RunConfig {
            shape: [8, 8],
            j: 1.0,
            h: 0.0,
            beta: 0.44,
            updater: UpdaterKind::Metropolis,
            thermalize: 1000,
            sweeps_between: 10,
            n_samples: 500,
            seed: 42,
            start: Start::Hot,
            description: Some("critical-point check".to_string()),
        }
    }

    /// Assert a config fails validation, and hand back the message.
    fn invalid_message(config: &RunConfig) -> String {
        match config.validate() {
            Err(ConfigError::Invalid(msg)) => msg,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let config = sample_config();
        let text = config.to_toml().unwrap();
        let parsed = RunConfig::parse(&text).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn parses_a_known_toml_document() {
        // Leading whitespace before a key is ignored by TOML, so the document
        // can be indented with the test rather than jammed against column 0.
        let text = r#"
            shape = [16, 16]
            j = 1.0
            h = 0.25
            beta = 0.44
            updater = "metropolis"
            thermalize = 2000
            sweeps_between = 20
            n_samples = 1000
            seed = 12345
            start = "hot"
            description = "near T_c"
        "#;
        let config = RunConfig::parse(text).unwrap();
        assert_eq!(config.shape, [16, 16]);
        assert_eq!(config.j, 1.0);
        assert_eq!(config.h, 0.25);
        assert_eq!(config.beta, 0.44);
        assert_eq!(config.thermalize, 2000);
        assert_eq!(config.sweeps_between, 20);
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.start, Start::Hot);
        assert_eq!(config.updater, UpdaterKind::Metropolis);
        assert_eq!(config.description.as_deref(), Some("near T_c"));
    }

    #[test]
    fn parses_and_round_trips_the_checkerboard_updater() {
        let mut config = sample_config();
        config.updater = UpdaterKind::Checkerboard;

        // Serializes with the lowercase variant name, and survives the round-trip.
        let text = config.to_toml().unwrap();
        assert!(text.contains(r#"updater = "checkerboard""#));
        assert_eq!(
            RunConfig::parse(&text).unwrap().updater,
            UpdaterKind::Checkerboard
        );
    }

    #[test]
    fn load_reads_a_file_from_disk() {
        let config = sample_config();
        let path = std::env::temp_dir().join("plaquette_config_load_test.toml");
        std::fs::write(&path, config.to_toml().unwrap()).unwrap();

        let loaded = RunConfig::load(&path).unwrap();
        assert_eq!(config, loaded);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn description_is_omitted_from_dumps_when_absent() {
        // A TOML comment would not survive the round-trip, which is why the note
        // is a field; when unset it should not clutter the dumped record.
        let mut config = sample_config();
        config.description = None;
        let text = config.to_toml().unwrap();
        assert!(!text.contains("description"));
        assert_eq!(RunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn validate_accepts_the_sample_config() {
        assert!(sample_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_zero_extent() {
        let mut config = sample_config();
        config.shape = [8, 0];
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
        // Legitimate, if unusual: no warmup and no decorrelation gap.
        let mut config = sample_config();
        config.thermalize = 0;
        config.sweeps_between = 0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // A typo'd *optional* field is the dangerous case: it parses cleanly
        // without `deny_unknown_fields`, and the run silently ignores it.
        let text = r#"
            shape = [4, 4]
            j = 1.0
            beta = 0.5
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
            corelator = true
        "#;
        assert!(matches!(RunConfig::parse(text), Err(ConfigError::Parse(_))));
    }

    #[test]
    fn h_defaults_to_the_field_free_model() {
        let text = r#"
            shape = [4, 4]
            j = 1.0
            beta = 0.5
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        let config = RunConfig::parse(text).unwrap();
        assert_eq!(config.h, 0.0);
        // `j`, by contrast, stays required.
        let without_j = text.replace("j = 1.0\n", "");
        assert!(matches!(
            RunConfig::parse(&without_j),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn build_produces_pieces_matching_the_file() {
        let text = r#"
            shape = [8, 4]
            j = 1.5
            h = 0.25
            beta = 0.44
            thermalize = 100
            sweeps_between = 5
            n_samples = 50
            seed = 99
            start = "cold"
        "#;
        let config = RunConfig::parse(text).unwrap();
        let (lattice, model, _rng, start, beta) = config.build();

        assert_eq!(lattice.shape(), [8, 4]);
        assert_eq!(lattice.n_sites(), 32);
        assert_eq!(model, Ising::new(1.5, 0.25));
        assert_eq!(beta, 0.44);
        assert_eq!(start.n_sites(), 32);
        // `start = "cold"` must give the aligned ground state.
        assert_eq!(start, Configuration::<2>::cold(&lattice));
    }

    #[test]
    fn build_honors_a_hot_start() {
        let mut config = sample_config();
        config.start = Start::Hot;
        let (lattice, _, _, start, _) = config.build();

        // A hot start on 64 sites is aligned with probability 2^-64, so this
        // distinguishes it from cold without being flaky.
        assert_ne!(start, Configuration::<2>::cold(&lattice));
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

    #[test]
    fn parse_rejects_an_invalid_config() {
        // Validation runs as part of parsing, so a bad file fails at load time.
        let text = r#"
            shape = [0, 4]
            j = 1.0
            h = 0.0
            beta = 0.5
            thermalize = 10
            sweeps_between = 1
            n_samples = 10
            seed = 1
        "#;
        assert!(matches!(
            RunConfig::parse(text),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn optional_fields_default_when_omitted() {
        // `start` and `updater` are absent: they fall back to their defaults.
        let text = r#"
            shape = [4, 4]
            j = 1.0
            h = 0.0
            beta = 0.5
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        let config = RunConfig::parse(text).unwrap();
        assert_eq!(config.start, Start::Cold);
        assert_eq!(config.updater, UpdaterKind::Metropolis);
        assert_eq!(config.description, None);
    }
}
