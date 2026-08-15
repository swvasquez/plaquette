//! Ising run configuration: the file that specifies a single Ising run's
//! parameters.
//!
//! [`IsingRunConfig`] is the serializable form of the constructor arguments the
//! driver's pieces already take. It deserializes from TOML and assembles those
//! pieces — lattice, model, seeded generator, start configuration, `beta` — so
//! that serde and I/O stay confined to this module and the physics core never
//! learns a file was involved. The vocabulary it is written in —
//! [`Start`], [`UpdaterRule`], [`ConfigError`] — comes from
//! [`config`](crate::config) and is shared with the gauge schema; the schema
//! itself is this module's alone.
//!
//! The same struct round-trips, configuring a run *and* recording one: it
//! derives `Serialize` too, so a run can dump the exact config it ran.
//!
//! Its scope is a single `(L, beta, h)` point; a later `Scan` component would
//! wrap this rather than the reverse. The dimension is whatever the file's
//! `shape` is long: the energy scores nearest-neighbor bonds, which exist along
//! a line as much as on a square, so one dimension is the floor. The `D` that
//! [`build`](IsingRunConfig::build) needs is a compile-time parameter a driver
//! names in its own source, and
//! [`check_dimension`](IsingRunConfig::check_dimension) reports a file that
//! disagrees.

use crate::config::{
    BackendKind, ConfigError, ScheduleKind, Start, UpdaterRule, check_dimension, check_shape,
    check_updater, shape_array,
};
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::models::ising::Ising;
use crate::rng::RandRng;
use serde::{Deserialize, Serialize};

/// The fewest dimensions an Ising run can be built on, re-exported from the
/// model that owns it.
pub const MIN_DIMENSION: usize = Ising::MIN_DIMENSION;

/// A single run's parameters in serializable form: everything needed to produce
/// a run's configurations, and nothing about what is later measured from them.
///
/// The specification is *complete* — seed, start, and update algorithm included
/// — so feeding the same `IsingRunConfig` back in produces the same configurations.
/// Nothing that only affects downstream analysis belongs here, since the stored
/// configurations can be measured any number of ways after the fact.
///
/// Unknown keys are rejected rather than ignored: a typo in an *optional* field
/// (`updatr = "metropolis"`) would otherwise parse cleanly and silently not do
/// what the file says.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IsingRunConfig {
    // --- physics ---
    /// Per-axis lattice extents `[L_0, ..., L_{D-1}]`. Its length *is* the
    /// dimension of the run, which is why it is a list rather than a fixed-width
    /// tuple: nothing else in the file says how many directions there are.
    pub shape: Vec<usize>,
    /// Nearest-neighbor coupling `J`.
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
    /// Which update rule advances the chain; defaults to
    /// [`UpdaterRule::Metropolis`]. A run parameter rather than a caller's
    /// choice, because two runs of the same physics under different algorithms
    /// are different runs.
    #[serde(default = "default_updater")]
    pub updater: UpdaterRule,
    /// Which schedule a local rule runs under; omitted means the random
    /// schedule. Must stay unset for the cluster rules
    /// ([`UpdaterRule::SwendsenWang`] and [`UpdaterRule::Wolff`]), which have
    /// no schedule at all.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<ScheduleKind>,
    /// Where the run executes; defaults to the CPU. A local rule on the GPU
    /// needs `schedule = "checkerboard"` and even extents.
    #[serde(default)]
    pub backend: BackendKind,
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
    /// [`build`](IsingRunConfig::build)).
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

fn default_updater() -> UpdaterRule {
    UpdaterRule::Metropolis
}

impl IsingRunConfig {
    /// Read and validate a config from a TOML file at `path`. Validation runs as
    /// part of loading, so a malformed run fails here rather than partway
    /// through a long chain.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Parse and validate a config from TOML text — the filesystem-free half of
    /// [`load`](IsingRunConfig::load), with the same guarantees.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: IsingRunConfig = toml::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// The dimension this run is on: the number of extents its shape names.
    pub fn dimension(&self) -> usize {
        self.shape.len()
    }

    /// Check that this run's dimension is `D`, the one the calling program was
    /// built for.
    ///
    /// A driver names `D` in its own source and calls this right after loading,
    /// so a file written for another dimension is a clean message rather than a
    /// panic partway through a run. See
    /// [`check_dimension`] for why the dimension
    /// is a compile-time choice.
    pub fn check_dimension<const D: usize>(&self) -> Result<(), ConfigError> {
        check_dimension(&self.shape, D)
    }

    /// Check that the values describe a runnable run.
    ///
    /// Rejects what would otherwise panic or produce nothing: an empty shape or
    /// a zero extent, a non-positive or non-finite `beta`, a non-finite coupling
    /// or field, and a run that would record no samples. `thermalize` and
    /// `sweeps_between` may be zero — no warmup and no decorrelation gap are
    /// unusual but legitimate.
    ///
    /// The updater rules are `check_updater`'s: a cluster rule takes no
    /// schedule, and a local rule on the GPU needs the checkerboard schedule
    /// and even extents.
    ///
    /// One rule on top of those is this schema's own: a cluster schedule needs
    /// the spin-flip symmetry a field breaks, so a non-zero `h` is refused.
    pub fn validate(&self) -> Result<(), ConfigError> {
        check_shape(&self.shape, MIN_DIMENSION, "nearest-neighbor bonds")?;
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
        check_updater(self.updater, self.schedule, self.backend, &self.shape)?;
        // A cluster move relabels a whole cluster at once, which is only
        // weight-preserving when the two spin values are interchangeable — and a
        // field is exactly what tells them apart. The graceful counterpart of
        // `ClusterUpdate::new`'s panic.
        if self.updater.builds_clusters() && self.h != 0.0 {
            return Err(ConfigError::Invalid(format!(
                "{:?} relabels whole clusters at once, which needs the spin-flip \
                 symmetry an external field breaks, but h is {}",
                self.updater, self.h
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
    /// run reproducible from [`seed`](IsingRunConfig::seed) alone.
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
    /// `D` must match [`dimension`](IsingRunConfig::dimension). It is a
    /// parameter rather than something read off the config because a lattice's
    /// dimension is part of its type, so a driver names it in its own source and
    /// calls [`check_dimension`](IsingRunConfig::check_dimension) first to turn
    /// a mismatch into a message.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid. [`load`](IsingRunConfig::load) and
    /// [`parse`](IsingRunConfig::parse) validate, so this can only fire on a
    /// `IsingRunConfig` built by hand (as a future `Scan` would) that skipped
    /// [`validate`](IsingRunConfig::validate). Panics too if `D` disagrees with
    /// the shape's length.
    pub fn build<const D: usize>(&self) -> (Lattice<D>, Ising, RandRng, Configuration<2>, f64) {
        self.validate()
            .expect("build called on an unvalidated config");
        let lattice = Lattice::new(shape_array::<D>(&self.shape));
        let model = Ising::new(self.j, self.h);
        let mut rng = RandRng::seed_from_u64(self.seed);
        // Drawn from `rng` *after* seeding and *before* the chain uses it, so the
        // whole run — start configuration included — replays from `seed`.
        let config = match self.start {
            Start::Cold => Configuration::cold(&lattice, Cell::Site),
            Start::Hot => Configuration::hot(&lattice, Cell::Site, &mut rng),
        };

        (lattice, model, rng, config, self.beta)
    }

    /// Render the config back to TOML — the inverse of
    /// [`parse`](IsingRunConfig::parse), so a run can dump exactly the config it ran
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
    fn sample_config() -> IsingRunConfig {
        IsingRunConfig {
            shape: vec![8, 8],
            j: 1.0,
            h: 0.0,
            beta: 0.44,
            updater: UpdaterRule::Metropolis,
            schedule: None,
            backend: BackendKind::Cpu,
            thermalize: 1000,
            sweeps_between: 10,
            n_samples: 500,
            seed: 42,
            start: Start::Hot,
            description: Some("critical-point check".to_string()),
        }
    }

    /// Assert a config fails validation, and hand back the message.
    fn invalid_message(config: &IsingRunConfig) -> String {
        match config.validate() {
            Err(ConfigError::Invalid(msg)) => msg,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let config = sample_config();
        let text = config.to_toml().unwrap();
        let parsed = IsingRunConfig::parse(&text).unwrap();
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
        let config = IsingRunConfig::parse(text).unwrap();
        assert_eq!(config.shape, [16, 16]);
        assert_eq!(config.j, 1.0);
        assert_eq!(config.h, 0.25);
        assert_eq!(config.beta, 0.44);
        assert_eq!(config.thermalize, 2000);
        assert_eq!(config.sweeps_between, 20);
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.start, Start::Hot);
        assert_eq!(config.updater, UpdaterRule::Metropolis);
        assert_eq!(config.description.as_deref(), Some("near T_c"));
    }

    #[test]
    fn parses_and_round_trips_the_checkerboard_schedule() {
        let mut config = sample_config();
        config.schedule = Some(ScheduleKind::Checkerboard);

        // Serializes with the lowercase variant name, and survives the round-trip.
        let text = config.to_toml().unwrap();
        assert!(text.contains(r#"schedule = "checkerboard""#));
        assert_eq!(IsingRunConfig::parse(&text).unwrap(), config);
    }

    /// The heat bath rule survives the round trip under every schedule and
    /// backend pairing a run can name. The name is what a config file is
    /// written against, so it is checked literally rather than by `Debug`.
    #[test]
    fn parses_and_round_trips_the_heat_bath_rule() {
        for (schedule, backend) in [
            (None, BackendKind::Cpu),
            (Some(ScheduleKind::Checkerboard), BackendKind::Cpu),
            (Some(ScheduleKind::Checkerboard), BackendKind::Gpu),
        ] {
            let mut config = sample_config();
            config.updater = UpdaterRule::HeatBath;
            config.schedule = schedule;
            config.backend = backend;

            let text = config.to_toml().unwrap();
            assert!(text.contains(r#"updater = "heat_bath""#));
            let parsed = IsingRunConfig::parse(&text).unwrap();
            assert_eq!(parsed, config);
            assert!(parsed.validate().is_ok(), "{schedule:?} on {backend:?}");
        }
    }

    #[test]
    fn parses_and_round_trips_the_gpu_backend() {
        let mut config = sample_config();
        config.schedule = Some(ScheduleKind::Checkerboard);
        config.backend = BackendKind::Gpu;

        let text = config.to_toml().unwrap();
        assert!(text.contains(r#"backend = "gpu""#));
        assert_eq!(IsingRunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn parses_and_round_trips_the_cluster_updaters() {
        for (rule, rendered) in [
            (UpdaterRule::SwendsenWang, r#"updater = "swendsen_wang""#),
            (UpdaterRule::Wolff, r#"updater = "wolff""#),
        ] {
            let mut config = sample_config();
            config.updater = rule;

            let text = config.to_toml().unwrap();
            assert!(text.contains(rendered), "{rule:?}: {text}");
            assert_eq!(IsingRunConfig::parse(&text).unwrap().updater, rule);
        }
    }

    #[test]
    fn validate_rejects_a_cluster_updater_with_a_field() {
        // A field tells the two spin values apart, which is exactly the symmetry
        // flipping a whole cluster at once relies on. The load-time counterpart
        // of `ClusterUpdate::new`'s panic.
        for backend in [BackendKind::Cpu, BackendKind::Gpu] {
            for rule in [UpdaterRule::SwendsenWang, UpdaterRule::Wolff] {
                let mut config = sample_config();
                config.updater = rule;
                config.backend = backend;
                config.h = 0.25;
                let message = invalid_message(&config);
                assert!(
                    message.contains("spin-flip symmetry"),
                    "{backend:?} / {rule:?}: {message}"
                );

                config.h = 0.0;
                assert!(
                    config.validate().is_ok(),
                    "{backend:?} / {rule:?}: field-free is fine"
                );
            }
        }
    }

    #[test]
    fn parses_and_round_trips_the_gpu_cluster_updater() {
        // The cluster rule composes with the gpu backend like any other —
        // the device cluster chain names no model.
        let mut config = sample_config();
        config.updater = UpdaterRule::SwendsenWang;
        config.backend = BackendKind::Gpu;

        let text = config.to_toml().unwrap();
        assert!(text.contains(r#"updater = "swendsen_wang""#));
        assert!(text.contains(r#"backend = "gpu""#));
        assert_eq!(IsingRunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn validate_accepts_an_odd_extent_for_a_cluster_kind() {
        // A cluster update has no coloring for the periodic wrap to spoil, so
        // unlike the GPU checkerboard it runs on any shape.
        for backend in [BackendKind::Cpu, BackendKind::Gpu] {
            for rule in [UpdaterRule::SwendsenWang, UpdaterRule::Wolff] {
                let mut config = sample_config();
                config.updater = rule;
                config.backend = backend;
                config.shape = vec![9, 7];
                assert!(
                    config.validate().is_ok(),
                    "{backend:?} / {rule:?} on an odd lattice"
                );
            }
        }
    }

    #[test]
    fn load_reads_a_file_from_disk() {
        let config = sample_config();
        let path = std::env::temp_dir().join("plaquette_config_load_test.toml");
        std::fs::write(&path, config.to_toml().unwrap()).unwrap();

        let loaded = IsingRunConfig::load(&path).unwrap();
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
        assert_eq!(IsingRunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn validate_accepts_the_sample_config() {
        assert!(sample_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_zero_extent() {
        let mut config = sample_config();
        config.shape = vec![8, 0];
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
    fn validate_rejects_a_random_schedule_on_the_gpu() {
        // The device runs a local sweep as parallel color passes, and a random
        // schedule has no colors — the combination is unrunnable, not slow.
        let mut config = sample_config();
        config.backend = BackendKind::Gpu;
        let message = invalid_message(&config);
        assert!(message.contains("checkerboard"), "{message}");

        config.schedule = Some(ScheduleKind::Random);
        let message = invalid_message(&config);
        assert!(message.contains("checkerboard"), "{message}");
    }

    #[test]
    fn validate_rejects_a_schedule_on_the_cluster_rule() {
        // The cluster update is not a kernel under a schedule, so naming one
        // alongside it is a contradiction rather than a preference.
        for rule in [UpdaterRule::SwendsenWang, UpdaterRule::Wolff] {
            let mut config = sample_config();
            config.updater = rule;
            config.schedule = Some(ScheduleKind::Checkerboard);
            let message = invalid_message(&config);
            assert!(message.contains("takes no schedule"), "{rule:?}: {message}");
        }
    }

    #[test]
    fn validate_rejects_an_odd_extent_on_the_gpu() {
        // A parallel color pass needs even extents. Caught here so a typo'd
        // shape is a load-time error rather than a panic when the device chain
        // is built — the CPU schedules are unaffected, since a sequential sweep
        // has no such requirement.
        let mut config = sample_config();
        config.schedule = Some(ScheduleKind::Checkerboard);
        config.backend = BackendKind::Gpu;
        config.shape = vec![16, 15];
        let message = invalid_message(&config);
        assert!(message.contains("even extents"), "{message}");
        assert!(message.contains("axis 1"), "{message}");

        config.backend = BackendKind::Cpu;
        assert!(
            config.validate().is_ok(),
            "an odd extent is fine on the CPU"
        );
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
        assert!(matches!(
            IsingRunConfig::parse(text),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn the_shape_length_is_the_dimension() {
        // The schema puts no length on `shape`: whatever it names is the run's
        // dimension. This is the field that used to be fixed at two, so what is
        // asserted is that three, two, and one all load and report themselves.
        let text = r#"
            shape = [8, 8, 8]
            j = 1.0
            beta = 0.44
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        for (shape, dimension) in [("[8, 8, 8]", 3), ("[8, 8]", 2), ("[8]", 1)] {
            let config = IsingRunConfig::parse(&text.replace("[8, 8, 8]", shape)).unwrap();
            assert_eq!(config.dimension(), dimension);
        }

        // An empty shape has no axes at all, which is a lattice with nothing on
        // it rather than a zero-dimensional model.
        let none = text.replace("[8, 8, 8]", "[]");
        assert!(matches!(
            IsingRunConfig::parse(&none),
            Err(ConfigError::Invalid(_))
        ));
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
        let config = IsingRunConfig::parse(text).unwrap();
        assert_eq!(config.h, 0.0);
        // `j`, by contrast, stays required.
        let without_j = text.replace("j = 1.0\n", "");
        assert!(matches!(
            IsingRunConfig::parse(&without_j),
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
        let config = IsingRunConfig::parse(text).unwrap();
        let (lattice, model, _rng, start, beta) = config.build::<2>();

        assert_eq!(lattice.shape(), [8, 4]);
        assert_eq!(lattice.n_sites(), 32);
        assert_eq!(model, Ising::new(1.5, 0.25));
        assert_eq!(beta, 0.44);
        assert_eq!(start.n_vars(), 32);
        // `start = "cold"` must give the aligned ground state.
        assert_eq!(start, Configuration::<2>::cold(&lattice, Cell::Site));
    }

    #[test]
    fn build_honors_a_hot_start() {
        let mut config = sample_config();
        config.start = Start::Hot;
        let (lattice, _, _, start, _) = config.build::<2>();

        // A hot start on 64 sites is aligned with probability 2^-64, so this
        // distinguishes it from cold without being flaky.
        assert_ne!(start, Configuration::<2>::cold(&lattice, Cell::Site));
    }

    #[test]
    fn build_is_reproducible_from_the_seed() {
        // The whole point of assembling in one place: same config in, same run
        // out — start configuration *and* the generator state the chain inherits.
        let mut config = sample_config();
        config.start = Start::Hot;

        let (_, _, mut rng_a, start_a, _) = config.build::<2>();
        let (_, _, mut rng_b, start_b, _) = config.build::<2>();

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
        let (_, _, mut rng, _, _) = config.build::<2>();

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
            IsingRunConfig::parse(text),
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
        let config = IsingRunConfig::parse(text).unwrap();
        assert_eq!(config.start, Start::Cold);
        assert_eq!(config.updater, UpdaterRule::Metropolis);
        assert_eq!(config.description, None);
    }
}
