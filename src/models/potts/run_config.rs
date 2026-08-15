//! Potts run configuration: the file that specifies a single `q`-state Potts
//! run's parameters.
//!
//! [`PottsRunConfig`] is to [`Potts`] what
//! [`IsingRunConfig`](crate::models::ising::run_config::IsingRunConfig) is to
//! [`Ising`](crate::models::ising::Ising), and it is a separate type rather than a
//! widened one because the two models' `h` are different shapes. Ising's is a
//! single number, since `+1` and `−1` are values one dial can pull between;
//! this model's is a list with one entry per label, since favoring red is not
//! the opposite of favoring blue. A schema covering both would have to accept
//! either a scalar or a list under one key, which is a hole rather than a
//! generalization.
//!
//! What the schemas share is the vocabulary in [`config`](crate::config) —
//! [`Start`], [`UpdaterRule`], [`ConfigError`] — so the driver fields, the
//! reproducibility fields, and the load/parse/validate/round-trip contract are
//! written and read the same way across all three models.
//!
//! # Why the file cannot choose `Q` or `D`
//!
//! Both are const generics, resolved when the program is compiled, and a config
//! file is read when it runs. The action and the updaters stay generic over both
//! — nothing in [`Potts`] or in the checkerboard schedule
//! names a state count or a dimension — but a *driver* has to pick one pair to
//! build, because the lattice's dimension and the configuration's state count are
//! part of their types. So the driver names them in its own source, the way
//! `examples/potts/potts.rs` does, and this module supplies the pair the shipped
//! example and its `potts.toml` are written for as [`POTTS_Q`] and [`POTTS_D`].
//!
//! The dimension is at least checkable after the fact, since the file's `shape`
//! says how many axes it names: [`check_dimension`](PottsRunConfig::check_dimension)
//! turns a disagreement into a message. Nothing in the file mentions `Q` at all,
//! so there is nothing to check it against — which is why the constant is worth
//! naming here rather than leaving loose in the example.

use crate::config::{
    BackendKind, ConfigError, ScheduleKind, Start, UpdaterRule, check_dimension, check_shape,
    check_updater, shape_array,
};
use crate::configuration::{Cell, Configuration};
use crate::lattice::Lattice;
use crate::models::potts::{self as model, Potts};
use crate::rng::RandRng;
use serde::{Deserialize, Serialize};

/// The fewest dimensions a Potts run can be built on, re-exported from the model
/// that owns it — the floor follows from what [`Potts`] scores, not from
/// anything about config files.
pub const MIN_DIMENSION: usize = model::POTTS_MIN_DIMENSION;

/// The fewest states a Potts run can be built at, re-exported from [`Potts`] for
/// the same reason [`MIN_DIMENSION`] is.
pub const MIN_STATES: usize = model::POTTS_MIN_STATES;

/// The state count the shipped example and `examples/potts/potts.toml` are built
/// for.
///
/// Three, which is the smallest `q` that is not the Ising model in disguise and
/// the one the two-dimensional transition is most often quoted at. Two would run
/// the same physics [`Ising`](crate::models::ising::Ising) already does, up to a factor
/// of two in the coupling; four is where the transition stops being continuous.
pub const POTTS_Q: usize = 3;

/// The lattice dimension the shipped example and `examples/potts/potts.toml` are
/// built for.
///
/// Two, which is where the exact critical coupling
/// `beta_c = ln(1 + sqrt(q))` applies, so a run has a number to be checked
/// against rather than only a contrast.
pub const POTTS_D: usize = 2;

/// A single Potts run's parameters in serializable form: everything needed to
/// produce a run's configurations, and nothing about what is later measured from
/// them.
///
/// The specification is *complete* — seed, start, and update algorithm included
/// — so feeding the same `PottsRunConfig` back in produces the same
/// configurations. The state count is the one exception and cannot be otherwise;
/// see the module docs.
///
/// Unknown keys are rejected rather than ignored, and `h`'s shape does the rest:
/// an Ising file handed to this parser fails because its `h` is a number where
/// this schema wants a list, rather than running a Potts model at whatever the
/// file happened to say. That is the whole of the discrimination, since a
/// field-free Ising file and a Potts file are otherwise the same document — and
/// at `q = 2` they very nearly describe the same run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PottsRunConfig {
    // --- physics ---
    /// Per-axis lattice extents `[L_0, ..., L_{D-1}]`. Its length *is* the
    /// dimension of the run, which is why it is a list rather than a fixed-width
    /// tuple: nothing else in the file says how many directions there are.
    pub shape: Vec<usize>,
    /// Nearest-neighbor coupling `J`.
    pub j: f64,
    /// Per-label energy offsets, one entry per label; defaults to empty, which
    /// means the symmetric model with no offsets at all.
    ///
    /// A list rather than a fixed-width array because the schema has no way to
    /// know `Q` — nothing in the file names it. That leaves the length
    /// unvalidated at parse time and checked in [`build`](PottsRunConfig::build),
    /// which is the first point where the driver's `Q` is visible. Empty or
    /// exactly `Q` entries are accepted; anything else is the file and the
    /// program disagreeing about how many labels there are, which is worth a
    /// message rather than a silent pad or truncate.
    ///
    /// Only the *differences* between entries do anything: adding a constant to
    /// every one of them shifts the energy of every configuration alike and
    /// cancels out of the sampling, so `[0.5, 0.5, 0.5]` runs exactly as
    /// `[0.0, 0.0, 0.0]` does.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub h: Vec<f64>,
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
    /// [`build`](PottsRunConfig::build)).
    #[serde(default = "default_start")]
    pub start: Start,

    // --- provenance ---
    /// A free-form human note about the run, deliberately not part of its
    /// identity. It has to be a *field* rather than a TOML comment because serde
    /// drops comments on the way back out, so a comment would not survive into
    /// the dumped record. It is also the only place a file can record which `q`
    /// it was written for, since the schema has no field for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

fn default_start() -> Start {
    Start::Cold
}

fn default_updater() -> UpdaterRule {
    UpdaterRule::Metropolis
}

impl PottsRunConfig {
    /// Read and validate a config from a TOML file at `path`. Validation runs as
    /// part of loading, so a malformed run fails here rather than partway
    /// through a long chain.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        Self::parse(&text)
    }

    /// Parse and validate a config from TOML text — the filesystem-free half of
    /// [`load`](PottsRunConfig::load), with the same guarantees.
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: PottsRunConfig = toml::from_str(text)?;
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
    /// panic partway through a run. See [`check_dimension`] for why the dimension
    /// is a compile-time choice — and the module docs for why `Q` has no
    /// counterpart to this.
    pub fn check_dimension<const D: usize>(&self) -> Result<(), ConfigError> {
        check_dimension(&self.shape, D)
    }

    /// Check that the values describe a runnable Potts run.
    ///
    /// Rejects what would otherwise panic or produce nothing — an empty shape or
    /// a zero extent, a non-positive or non-finite `beta`, a non-finite coupling,
    /// a run recording no samples — and applies `check_updater` against
    /// [`Cell::Site`]: this model's variables live on sites, so it takes the
    /// grade-neutral Metropolis or a site schedule and refuses a link one, whose
    /// direction-and-parity coloring is a statement about gauge variables and
    /// says nothing about a label on a site. A schedule that colors in parallel
    /// additionally needs even extents.
    ///
    /// A cluster schedule carries a rule of its own: it needs the label symmetry,
    /// so a non-zero `h` is refused. That is the graceful counterpart of the
    /// panic in [`ClusterUpdate::new`](crate::updater::ClusterUpdate::new),
    /// the same pairing [`check_dimension`](PottsRunConfig::check_dimension) and
    /// `shape_array` already use.
    ///
    /// `thermalize` and `sweeps_between` may be zero — no warmup and no
    /// decorrelation gap are unusual but legitimate.
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
        // The entries are checked here; how many there should be is `build`'s
        // job, since the count is `Q` and nothing in the file names it.
        if let Some(label) = self.h.iter().position(|v| !v.is_finite()) {
            return Err(ConfigError::Invalid(format!(
                "every entry of h must be finite, but entry {label} is {}",
                self.h[label]
            )));
        }
        check_updater(self.updater, self.schedule, self.backend, &self.shape)?;
        // The load-time counterpart of `ClusterUpdate::new`'s panic. A
        // cluster move relabels a whole cluster at once, which is only
        // weight-preserving when nothing distinguishes one label from another,
        // and an offset is exactly what distinguishes them.
        if self.updater.builds_clusters() && self.h.iter().any(|&h_a| h_a != 0.0) {
            return Err(ConfigError::Invalid(format!(
                "{:?} relabels whole clusters at once, which needs the label \
                 symmetry a per-label offset breaks, but h{:?} is not all zero",
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

    /// Assemble the constructed pieces a Potts run needs, in the order that
    /// makes the run reproducible from [`seed`](PottsRunConfig::seed) alone.
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
    /// another, leaving a run that looks reproducible and quietly is not. A hot
    /// Potts start draws a label uniformly from all `Q`, so `Q` changes the
    /// stream as well as the field.
    ///
    /// Both `Q` and `D` are named by the caller, since neither is anything the
    /// file can decide; `D` must match [`dimension`](PottsRunConfig::dimension),
    /// which [`check_dimension`](PottsRunConfig::check_dimension) reports on
    /// cleanly beforehand.
    ///
    /// # Panics
    ///
    /// Panics if the config is invalid. [`load`](PottsRunConfig::load) and
    /// [`parse`](PottsRunConfig::parse) validate, so this can only fire on a
    /// `PottsRunConfig` built by hand that skipped
    /// [`validate`](PottsRunConfig::validate). Panics too if `D` disagrees with
    /// the shape's length, or if `Q` is below [`MIN_STATES`].
    pub fn build<const Q: usize, const D: usize>(
        &self,
    ) -> (Lattice<D>, Potts<Q>, RandRng, Configuration<Q>, f64) {
        self.validate()
            .expect("build called on an unvalidated config");
        assert!(Q >= MIN_STATES, "{}", model::TOO_FEW_STATES);
        let lattice = Lattice::new(shape_array::<D>(&self.shape));
        // The one thing in the file whose shape depends on `Q`, so this is the
        // first place it can be checked at all.
        let model = if self.h.is_empty() {
            Potts::symmetric(self.j)
        } else {
            let offsets = <[f64; Q]>::try_from(self.h.as_slice()).unwrap_or_else(|_| {
                panic!(
                    "this program is built for {Q} labels, but h{:?} gives {}; \
                     leave h out for the symmetric model",
                    self.h,
                    self.h.len()
                )
            });
            Potts::new(self.j, offsets)
        };
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
    /// [`parse`](PottsRunConfig::parse), so a run can dump exactly the config it
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
    fn sample_config() -> PottsRunConfig {
        PottsRunConfig {
            shape: vec![16, 16],
            j: 1.0,
            h: Vec::new(),
            beta: 1.0,
            updater: UpdaterRule::Metropolis,
            schedule: None,
            backend: BackendKind::Cpu,
            thermalize: 1000,
            sweeps_between: 10,
            n_samples: 500,
            seed: 42,
            start: Start::Hot,
            description: Some("q = 3, ordered phase".to_string()),
        }
    }

    /// Assert a config fails validation, and hand back the message.
    fn invalid_message(config: &PottsRunConfig) -> String {
        match config.validate() {
            Err(ConfigError::Invalid(msg)) => msg,
            other => panic!("expected an Invalid error, got {other:?}"),
        }
    }

    #[test]
    fn round_trips_through_toml() {
        let config = sample_config();
        let text = config.to_toml().unwrap();
        let parsed = PottsRunConfig::parse(&text).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn parses_a_known_toml_document() {
        // Leading whitespace before a key is ignored by TOML, so the document
        // can be indented with the test rather than jammed against column 0.
        let text = r#"
            shape = [24, 24]
            j = 1.0
            beta = 1.005
            schedule = "checkerboard"
            thermalize = 2000
            sweeps_between = 20
            n_samples = 1000
            seed = 12345
            start = "hot"
            description = "q = 3 near beta_c"
        "#;
        let config = PottsRunConfig::parse(text).unwrap();
        assert_eq!(config.shape, [24, 24]);
        assert_eq!(config.j, 1.0);
        assert_eq!(config.beta, 1.005);
        assert_eq!(config.thermalize, 2000);
        assert_eq!(config.sweeps_between, 20);
        assert_eq!(config.n_samples, 1000);
        assert_eq!(config.seed, 12345);
        assert_eq!(config.start, Start::Hot);
        assert_eq!(config.schedule, Some(ScheduleKind::Checkerboard));
        assert_eq!(config.description.as_deref(), Some("q = 3 near beta_c"));
    }

    #[test]
    fn load_reads_a_file_from_disk() {
        let config = sample_config();
        let path = std::env::temp_dir().join("plaquette_potts_config_load_test.toml");
        std::fs::write(&path, config.to_toml().unwrap()).unwrap();

        let loaded = PottsRunConfig::load(&path).unwrap();
        assert_eq!(config, loaded);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn description_is_omitted_from_dumps_when_absent() {
        let mut config = sample_config();
        config.description = None;
        let text = config.to_toml().unwrap();
        assert!(!text.contains("description"));
        assert_eq!(PottsRunConfig::parse(&text).unwrap(), config);
    }

    #[test]
    fn optional_fields_default_when_omitted() {
        let text = r#"
            shape = [8, 8]
            j = 1.0
            beta = 1.0
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        let config = PottsRunConfig::parse(text).unwrap();
        assert_eq!(config.start, Start::Cold);
        assert_eq!(config.updater, UpdaterRule::Metropolis);
        assert_eq!(config.description, None);
    }

    #[test]
    fn validate_accepts_the_sample_config() {
        assert!(sample_config().validate().is_ok());
    }

    #[test]
    fn validate_rejects_a_zero_extent() {
        let mut config = sample_config();
        config.shape = vec![16, 0];
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
    fn validate_rejects_a_random_schedule_on_the_gpu() {
        // The same rule the Ising schema applies, for the same reason: the
        // device runs a local sweep as parallel color passes, and a random
        // schedule has no colors.
        let mut config = sample_config();
        config.backend = BackendKind::Gpu;
        let message = invalid_message(&config);
        assert!(message.contains("checkerboard"), "{message}");
    }

    #[test]
    fn validate_rejects_a_schedule_on_the_cluster_rules() {
        // A cluster update is not a kernel under a schedule, so naming one
        // alongside either cluster rule is a contradiction, not a preference.
        for rule in [UpdaterRule::SwendsenWang, UpdaterRule::Wolff] {
            let mut config = sample_config();
            config.updater = rule;
            config.schedule = Some(ScheduleKind::Checkerboard);
            let message = invalid_message(&config);
            assert!(message.contains("takes no schedule"), "{rule:?}: {message}");
        }
    }

    #[test]
    fn parses_and_round_trips_the_composed_updaters() {
        // Every runnable (rule, schedule, backend) triple survives validation
        // on the way through TOML and back, rendered under the composed names.
        for (rule, rendered, schedule, backend) in [
            (
                UpdaterRule::Metropolis,
                "metropolis",
                None,
                BackendKind::Cpu,
            ),
            (UpdaterRule::HeatBath, "heat_bath", None, BackendKind::Cpu),
            (
                UpdaterRule::Metropolis,
                "metropolis",
                Some(ScheduleKind::Checkerboard),
                BackendKind::Cpu,
            ),
            (
                UpdaterRule::HeatBath,
                "heat_bath",
                Some(ScheduleKind::Checkerboard),
                BackendKind::Gpu,
            ),
            (
                UpdaterRule::Metropolis,
                "metropolis",
                Some(ScheduleKind::Checkerboard),
                BackendKind::Gpu,
            ),
            (
                UpdaterRule::SwendsenWang,
                "swendsen_wang",
                None,
                BackendKind::Cpu,
            ),
            (
                UpdaterRule::SwendsenWang,
                "swendsen_wang",
                None,
                BackendKind::Gpu,
            ),
            (UpdaterRule::Wolff, "wolff", None, BackendKind::Cpu),
            (UpdaterRule::Wolff, "wolff", None, BackendKind::Gpu),
        ] {
            let mut config = sample_config();
            config.updater = rule;
            config.schedule = schedule;
            config.backend = backend;

            let text = config.to_toml().unwrap();
            assert!(
                text.contains(&format!(r#"updater = "{rendered}""#)),
                "{text}"
            );
            let parsed = PottsRunConfig::parse(&text).unwrap();
            assert_eq!(parsed, config);
            assert!(
                parsed.validate().is_ok(),
                "{rule:?} / {schedule:?} / {backend:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_an_odd_extent_on_the_gpu() {
        // A parallel color pass needs even extents, so an odd shape is a
        // load-time error naming the axis rather than a panic when the device
        // chain is built. The CPU schedules are unaffected — run in sequence,
        // any site order is a valid Metropolis schedule.
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
    fn validate_rejects_a_cluster_updater_with_offsets() {
        // The load-time half of the pair `ClusterUpdate::new` panics on: a
        // per-label offset is exactly what makes relabeling a whole cluster
        // change the energy, so the move would sample the wrong distribution
        // rather than fail.
        for backend in [BackendKind::Cpu, BackendKind::Gpu] {
            for rule in [UpdaterRule::SwendsenWang, UpdaterRule::Wolff] {
                let mut config = sample_config();
                config.updater = rule;
                config.backend = backend;
                config.h = vec![0.1, 0.0, 0.0];
                let message = invalid_message(&config);
                assert!(message.contains("label symmetry"), "{rule:?}: {message}");
            }
            let mut config = sample_config();
            config.updater = UpdaterRule::SwendsenWang;
            config.backend = backend;

            // Absent and all-zero both mean the symmetric model, and both run.
            config.h = Vec::new();
            assert!(config.validate().is_ok(), "{backend:?} with no offsets");
            config.h = vec![0.0, 0.0, 0.0];
            assert!(config.validate().is_ok(), "{backend:?} with zero offsets");
        }
    }

    #[test]
    fn a_cluster_updater_accepts_odd_extents_even_on_the_gpu() {
        // The one place a reader is likely to expect the gpu cluster run to
        // behave like the gpu checkerboard and be wrong. The even-extent rule
        // exists because an odd wrap puts two same-color neighbors in one
        // parallel pass; a cluster update has no coloring to break, so any
        // shape runs.
        for backend in [BackendKind::Cpu, BackendKind::Gpu] {
            let mut config = sample_config();
            config.updater = UpdaterRule::SwendsenWang;
            config.backend = backend;
            config.shape = vec![15, 15, 7];
            assert!(config.validate().is_ok(), "{backend:?} on an odd lattice");
        }

        // The contrast, so this test would notice if the rule leaked the other
        // way and stopped applying to the coloring that does need it.
        let mut checkerboard = sample_config();
        checkerboard.schedule = Some(ScheduleKind::Checkerboard);
        checkerboard.backend = BackendKind::Gpu;
        checkerboard.shape = vec![15, 15, 7];
        assert!(invalid_message(&checkerboard).contains("even extents"));
    }

    #[test]
    fn parse_rejects_an_invalid_config() {
        // Validation runs as part of parsing, so a bad file fails at load time —
        // including an unrunnable triple, which a parse alone would let through.
        let text = r#"
            shape = [8, 8]
            j = 1.0
            beta = 1.0
            updater = "metropolis"
            backend = "gpu"
            thermalize = 10
            sweeps_between = 1
            n_samples = 10
            seed = 1
        "#;
        assert!(matches!(
            PottsRunConfig::parse(text),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // A typo'd *optional* field is the dangerous case: it parses cleanly
        // without `deny_unknown_fields`, and the run silently ignores it. `q` is
        // the one worth naming here — the schema has no such field, and a file
        // that set one would otherwise look like it had chosen the state count.
        let text = r#"
            shape = [8, 8]
            j = 1.0
            beta = 1.0
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
            q = 3
        "#;
        assert!(matches!(
            PottsRunConfig::parse(text),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn an_ising_file_is_refused_rather_than_run_as_a_potts_model() {
        // The schemas have no discriminant key, so `h` is what stops a file
        // meant for the other site model: `deny_unknown_fields` refuses a field
        // this model cannot price. It is the *whole* of the discrimination — a
        // field-free Ising file is otherwise a legitimate Potts file, and at
        // `q = 2` the two very nearly describe the same run.
        let ising = r#"
            shape = [16, 16]
            j = 1.0
            h = 0.25
            beta = 0.44
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        assert!(matches!(
            PottsRunConfig::parse(ising),
            Err(ConfigError::Parse(_))
        ));
    }

    #[test]
    fn the_shape_length_is_the_dimension() {
        // The schema puts no length on `shape`: whatever it names is the run's
        // dimension, down to the one nearest-neighbor bonds need.
        let text = r#"
            shape = [8, 8, 8]
            j = 1.0
            beta = 1.0
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        for (shape, dimension) in [("[8, 8, 8]", 3), ("[8, 8]", 2), ("[8]", 1)] {
            let config = PottsRunConfig::parse(&text.replace("[8, 8, 8]", shape)).unwrap();
            assert_eq!(config.dimension(), dimension);
        }

        // An empty shape has no axes at all, which is a lattice with nothing on
        // it rather than a zero-dimensional model.
        let none = text.replace("[8, 8, 8]", "[]");
        assert!(matches!(
            PottsRunConfig::parse(&none),
            Err(ConfigError::Invalid(_))
        ));
    }

    #[test]
    fn build_produces_pieces_matching_the_file() {
        let text = r#"
            shape = [8, 4]
            j = 1.5
            beta = 1.0
            thermalize = 100
            sweeps_between = 5
            n_samples = 50
            seed = 99
            start = "cold"
        "#;
        let config = PottsRunConfig::parse(text).unwrap();
        let (lattice, model, _rng, start, beta) = config.build::<POTTS_Q, 2>();

        assert_eq!(lattice.shape(), [8, 4]);
        assert_eq!(lattice.n_sites(), 32);
        assert_eq!(model, Potts::symmetric(1.5));
        assert_eq!(beta, 1.0);
        assert_eq!(start.cell(), Cell::Site);
        assert_eq!(start.n_vars(), 32);
        // `start = "cold"` must put every site on the same label.
        assert_eq!(start, Configuration::<POTTS_Q>::cold(&lattice, Cell::Site));
    }

    #[test]
    fn h_defaults_to_the_symmetric_model_and_round_trips_when_set() {
        // Absent means symmetric, and an absent `h` must not clutter a dump.
        let text = r#"
            shape = [8, 8]
            j = 1.0
            beta = 1.0
            thermalize = 100
            sweeps_between = 5
            n_samples = 200
            seed = 7
        "#;
        let config = PottsRunConfig::parse(text).unwrap();
        assert!(config.h.is_empty());
        // The key itself, not the letter — "shape" and "thermalize" carry one.
        assert!(!config.to_toml().unwrap().contains("h = "));
        let (_, model, _, _, _) = config.build::<POTTS_Q, 2>();
        assert_eq!(model, Potts::symmetric(1.0));

        // Present, it survives the round-trip and reaches the model in order.
        let mut with_offsets = config.clone();
        with_offsets.h = vec![0.5, 0.0, -0.25];
        let text = with_offsets.to_toml().unwrap();
        assert_eq!(PottsRunConfig::parse(&text).unwrap(), with_offsets);
        let (_, model, _, _, _) = with_offsets.build::<POTTS_Q, 2>();
        assert_eq!(model, Potts::new(1.0, [0.5, 0.0, -0.25]));
    }

    #[test]
    fn validate_rejects_a_non_finite_offset() {
        // The entries are checkable at parse time even though their count is
        // not, so the finiteness rule lives in `validate` with `j` and `beta`.
        let mut config = sample_config();
        config.h = vec![0.5, f64::NAN, 0.0];
        let message = invalid_message(&config);
        assert!(message.contains("entry 1"), "{message}");
    }

    #[test]
    #[should_panic(expected = "built for 3 labels")]
    fn build_refuses_an_offset_list_of_the_wrong_length() {
        // The one thing in the file whose shape depends on `Q`, so `build` is
        // the first place it can be checked — and it has to be checked, or the
        // list would be silently padded or truncated into a different model.
        let mut config = sample_config();
        config.h = vec![0.5, 0.0];
        config.build::<POTTS_Q, 2>();
    }

    #[test]
    fn build_honors_a_hot_start() {
        let mut config = sample_config();
        config.start = Start::Hot;
        let (lattice, _, _, start, _) = config.build::<POTTS_Q, 2>();

        // A hot start on 256 sites is uniform with vanishing probability, so
        // this distinguishes it from cold without being flaky.
        assert_ne!(start, Configuration::<POTTS_Q>::cold(&lattice, Cell::Site));
        // Every label a hot start draws is in range, and at three states it
        // really does reach past the two an Ising field would.
        assert!(start.variables().iter().all(|s| s.index() < POTTS_Q));
        assert!(start.variables().iter().any(|s| s.index() == 2));
    }

    #[test]
    fn build_is_reproducible_from_the_seed() {
        // The whole point of assembling in one place: same config in, same run
        // out — start configuration *and* the generator state the chain inherits.
        let mut config = sample_config();
        config.start = Start::Hot;

        let (_, _, mut rng_a, start_a, _) = config.build::<POTTS_Q, 2>();
        let (_, _, mut rng_b, start_b, _) = config.build::<POTTS_Q, 2>();

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
        let (_, _, mut rng, _, _) = config.build::<POTTS_Q, 2>();

        let mut fresh = RandRng::seed_from_u64(config.seed);
        assert_ne!(rng.next_f64(), fresh.next_f64());
    }

    #[test]
    fn check_dimension_accepts_only_the_shape_the_file_names() {
        // The graceful half of the pair below: a driver calls this after loading
        // and prints the message rather than letting `build` panic.
        let config = sample_config();
        assert!(config.check_dimension::<2>().is_ok());
        let message = match config.check_dimension::<3>() {
            Err(ConfigError::Invalid(msg)) => msg,
            other => panic!("expected an Invalid error, got {other:?}"),
        };
        assert!(message.contains("built for 3 dimensions"), "{message}");
        assert!(message.contains("names 2"), "{message}");
    }

    #[test]
    #[should_panic(expected = "this program is built for 3 dimensions")]
    fn build_refuses_a_dimension_the_shape_does_not_name() {
        // The backstop for a driver that skipped `check_dimension`. Without it
        // the shape would be silently truncated or padded to `D`.
        sample_config().build::<POTTS_Q, 3>();
    }

    #[test]
    #[should_panic(expected = "at least two states")]
    fn build_refuses_a_single_state() {
        // The `Q` counterpart of the guard above, and the reason it has to be
        // here rather than in `validate`: nothing in the file names `Q`, so this
        // is the first point at which the driver's choice is visible at all.
        sample_config().build::<1, 2>();
    }
}
