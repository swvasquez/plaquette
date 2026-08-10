//! Config vocabulary shared by the per-model run configs
//! ([`IsingRunConfig`](crate::models::ising::run_config::IsingRunConfig) and
//! [`GaugeRunConfig`](crate::models::gauge::run_config::GaugeRunConfig)): how a
//! load can fail, where a chain starts, and which algorithm advances it.
//!
//! The schemas stay separate types — one wide enough for both would carry a
//! hole for whichever model is running.

use crate::configuration::Cell;
use serde::{Deserialize, Serialize};

/// Check that a config's shape names `expected` axes, the dimension the driver
/// was built for.
///
/// `D` is a compile-time parameter and a file is read at runtime, so a driver
/// names its dimension and this reports a file that disagrees; dispatching on
/// the dimension at load would mean instantiating the whole sampler stack once
/// per value, so the point is fixed at compile time as the lattice-gauge codes
/// do. This is the graceful path — `build` panics on the same condition, as the
/// backstop for a caller that skipped the check.
pub fn check_dimension(shape: &[usize], expected: usize) -> Result<(), ConfigError> {
    if shape.len() != expected {
        return Err(ConfigError::Invalid(format!(
            "this program is built for {expected} dimensions, \
             but shape{shape:?} names {}",
            shape.len()
        )));
    }
    Ok(())
}

/// A config's shape as the fixed-width array a [`Lattice`](crate::lattice::Lattice)
/// needs, or a panic if it does not name `D` axes.
///
/// The panicking counterpart of [`check_dimension`], shared by both schemas so
/// the message a caller who skipped the check sees is written once.
pub(crate) fn shape_array<const D: usize>(shape: &[usize]) -> [usize; D] {
    <[usize; D]>::try_from(shape).unwrap_or_else(|_| {
        panic!(
            "this program is built for {D} dimensions, but shape{shape:?} names {}; \
             call check_dimension::<{D}> after loading to report this cleanly",
            shape.len()
        )
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

/// Which update algorithm a run uses to advance the chain — the *serializable
/// choice of* an [`Updater`](crate::updater::Updater), not an updater itself.
///
/// The set is the union across models, not a promise that every model runs
/// every entry: each schema's `validate` rejects the kinds that color the wrong
/// grade, rather than each model carrying a separate, narrower enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdaterKind {
    /// Single-variable-flip Metropolis on the CPU
    /// ([`Metropolis`](crate::updater::Metropolis)).
    Metropolis,
    /// Metropolis under a site checkerboard schedule, on the CPU
    /// ([`SiteCheckerboard`](crate::updater::SiteCheckerboard)).
    SiteCheckerboard,
    /// Metropolis under a link checkerboard schedule, on the CPU
    /// ([`LinkCheckerboard`](crate::updater::LinkCheckerboard)).
    LinkCheckerboard,
    /// The site checkerboard schedule run on the GPU
    /// ([`GpuIsingChain`](crate::models::ising::gpu::GpuIsingChain)). Backend
    /// and schedule are one variant, so a GPU run cannot be named without the
    /// coloring its kernel is written for.
    GpuSiteCheckerboard,
    /// The link checkerboard schedule run on the GPU
    /// ([`GpuGaugeChain`](crate::models::gauge::gpu::GpuGaugeChain)), fused for
    /// the same reason as
    /// [`GpuSiteCheckerboard`](UpdaterKind::GpuSiteCheckerboard).
    GpuLinkCheckerboard,
}

impl UpdaterKind {
    /// Which lattice cell this kind's schedule colors, or `None` for a schedule
    /// that reads a bare variable index and so works on either grade.
    ///
    /// The one fact behind both schemas' updater rule, stated here so a new
    /// variant must answer it to compile rather than the two rules drifting.
    pub fn cell(self) -> Option<Cell> {
        match self {
            UpdaterKind::Metropolis => None,
            UpdaterKind::SiteCheckerboard | UpdaterKind::GpuSiteCheckerboard => Some(Cell::Site),
            UpdaterKind::LinkCheckerboard | UpdaterKind::GpuLinkCheckerboard => Some(Cell::Link),
        }
    }

    /// Whether this kind colors in parallel, so its coloring must be
    /// collision-free *simultaneously* and every extent has to be even.
    ///
    /// The requirement follows from the pass running concurrently, not from the
    /// work happening on a device — a parallel CPU schedule would need it too.
    /// Both schemas ask this in `validate` so an odd shape fails at load rather
    /// than when the device chain is built; the device constructors assert it
    /// again, since they are reachable without a config.
    pub fn colors_in_parallel(self) -> bool {
        matches!(
            self,
            UpdaterKind::GpuSiteCheckerboard | UpdaterKind::GpuLinkCheckerboard
        )
    }
}

/// Check that `shape` describes a lattice this model can live on: at least
/// `min_dimension` axes, each of positive extent.
///
/// The floor follows from which cell the model's energy scores (`scores` names
/// it for the message): links need one dimension, plaquettes two. Below the
/// bound nothing fails loudly — the cell count and energy come out identically
/// zero — so the bound has to be stated rather than left to the arithmetic.
pub(crate) fn check_shape(
    shape: &[usize],
    min_dimension: usize,
    scores: &str,
) -> Result<(), ConfigError> {
    if shape.len() < min_dimension {
        return Err(ConfigError::Invalid(format!(
            "this run scores {scores}, which need at least {min_dimension} dimensions, \
             but shape{shape:?} names {}",
            shape.len()
        )));
    }
    if let Some(axis) = shape.iter().position(|&l| l == 0) {
        return Err(ConfigError::Invalid(format!(
            "every lattice extent must be positive, but shape{shape:?} is 0 on axis {axis}"
        )));
    }
    Ok(())
}

/// Check that `updater` can advance a field on `cell` over a lattice of
/// `shape`, or say what is wrong with the pairing.
///
/// The two rules both schemas apply, in one place. The grade: only a
/// grade-neutral schedule or one coloring `cell` itself can run here — a
/// mismatched schedule would be wrong physics, not slower physics. The shape: a
/// parallel color pass needs every extent even, or an odd extent wraps a
/// variable onto a same-color one and detailed balance breaks silently; see
/// `docs/metropolis.md`. Sequential schedules have no such requirement, which
/// is why the second rule asks about the updater and not the shape alone.
pub(crate) fn check_updater(
    updater: UpdaterKind,
    cell: Cell,
    shape: &[usize],
) -> Result<(), ConfigError> {
    if let Some(colored) = updater.cell()
        && colored != cell
    {
        return Err(ConfigError::Invalid(format!(
            "this run updates {cell:?} variables, so it cannot use {updater:?}, \
             which colors {colored:?} variables"
        )));
    }
    if updater.colors_in_parallel()
        && let Some(axis) = shape.iter().position(|l| !l.is_multiple_of(2))
    {
        return Err(ConfigError::Invalid(format!(
            "{updater:?} colors in parallel, which needs even extents, \
             but shape{shape:?} is odd on axis {axis}"
        )));
    }
    Ok(())
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
