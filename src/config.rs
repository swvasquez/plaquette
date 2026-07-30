//! Config vocabulary: the pieces every run config is written in, whichever
//! model it configures.
//!
//! A run config is one struct per model —
//! [`IsingRunConfig`](crate::ising_config::IsingRunConfig) and
//! [`GaugeRunConfig`](crate::gauge_config::GaugeRunConfig) — because the two
//! models take different
//! parameters and a schema wide enough for both would carry a hole for whichever
//! one is running. What they do share is the vocabulary those schemas are
//! written in: how a load can fail, where a chain starts, and which algorithm
//! advances it. Those live here so neither schema owns them and neither has to
//! reach into the other's module to name them.
//!
//! Sharing stops at the vocabulary. The two schemas stay separate types with
//! separate entry points, and a caller names the one it means.

use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize a lattice shape, rejecting an array whose length is not `N`.
///
/// Needed because serde's fixed-size array visitor stops reading after `N`
/// entries and the TOML deserializer does not object to the ones left over, so a
/// *too long* `shape` would otherwise load silently truncated: `[8, 8, 8]` in a
/// two-dimensional schema drops an axis and runs a different theory than the
/// file describes, without a word. A shape that is too *short* already fails on
/// its own — the visitor runs out and says so. Going through a `Vec` first is
/// what makes the extra entries visible, and it is the same guarantee
/// `deny_unknown_fields` gives for keys, applied to the one field whose length
/// carries the dimension.
pub(crate) fn deserialize_shape<'de, De, const N: usize>(de: De) -> Result<[usize; N], De::Error>
where
    De: Deserializer<'de>,
{
    let extents = Vec::<usize>::deserialize(de)?;
    let found = extents.len();
    extents.try_into().map_err(|_| {
        let expected = format!("a lattice shape of length {N}");
        serde::de::Error::invalid_length(found, &expected.as_str())
    })
}

/// Which start configuration a run begins from, before thermalization —
/// mirroring the two [`Configuration`](crate::configuration::Configuration)
/// initializers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Start {
    /// Every variable in the same state (`Configuration::cold`).
    Cold,
    /// Each variable independently random (`Configuration::hot`).
    Hot,
}

/// Which update algorithm a run uses to advance the chain.
///
/// The *serializable choice of* an updater, not an updater itself — hence the
/// name, which also avoids colliding with the
/// [`Updater`](crate::updater::Updater) trait. Being a closed set is what makes
/// it recordable, unlike an arbitrary caller-supplied implementation.
///
/// The set is the union across models, not a promise that every model runs
/// every entry: the checkerboard schedule and the GPU backend are built for the
/// site models, so a gauge run rejects them in
/// [`validate`](crate::gauge_config::GaugeRunConfig::validate) rather than
/// through a separate, narrower enum. That keeps the rule where it can say what
/// went wrong, and relaxing it later — when a link-field checkerboard exists —
/// is a change to validation rather than a new type and a mapping between the
/// two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdaterKind {
    /// Single-variable-flip Metropolis on the CPU
    /// ([`Metropolis`](crate::updater::Metropolis)).
    Metropolis,
    /// Single-spin-flip Metropolis under a checkerboard schedule, on the CPU
    /// ([`Checkerboard`](crate::updater::Checkerboard)).
    Checkerboard,
    /// The checkerboard schedule run on the GPU
    /// ([`GpuChain`](crate::gpu::GpuChain)); serializes as `gpu_checkerboard`.
    /// The GPU's only algorithm is the checkerboard, so backend and schedule are
    /// one choice rather than two — which keeps illegal combinations
    /// unrepresentable.
    GpuCheckerboard,
}

/// What can go wrong turning a file into a usable run config. The stages are
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
