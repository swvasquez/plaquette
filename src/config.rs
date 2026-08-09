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

use crate::configuration::Cell;
use serde::{Deserialize, Serialize};

/// Check that a config's shape names `expected` axes, the dimension the driver
/// was built for.
///
/// The dimension is the one run parameter a file cannot decide on its own. Every
/// layer of the library is generic over `D`, but `D` is a compile-time parameter
/// and a file is read at runtime, so a driver names the dimension it is built for
/// and this reports a file that disagrees. Naming it in the driver rather than
/// dispatching on it at load is deliberate: a dispatch would have to instantiate
/// the whole sampler stack once per dimension, and once a second parameter is
/// generic — the state count `Q`, for a Potts or `U(1)` model — the set of
/// instantiations to enumerate is a grid rather than a list. Fixing the point at
/// compile time is also what the lattice-gauge codes do.
///
/// This is the graceful path, for a driver that wants to report a mismatch and
/// exit. `build` panics on the same condition, as the backstop for a caller that
/// skipped the check.
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
/// The panicking counterpart of [`check_dimension`], for the point where a
/// driver has already committed to a `D` and is building the run. Both schemas
/// convert here rather than each spelling out the conversion, so the message a
/// caller who skipped the check sees is written once.
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

/// Which update algorithm a run uses to advance the chain.
///
/// The *serializable choice of* an updater, not an updater itself — hence the
/// name, which also avoids colliding with the
/// [`Updater`](crate::updater::Updater) trait. Being a closed set is what makes
/// it recordable, unlike an arbitrary caller-supplied implementation.
///
/// The set is the union across models, not a promise that every model runs
/// every entry: the two checkerboard schedules each name the grade they color,
/// and a model accepts only the one matching its field. A gauge run rejects the
/// site schedules and the GPU backend in
/// [`validate`](crate::gauge_config::GaugeRunConfig::validate), and an Ising run
/// rejects the link schedule in
/// [`validate`](crate::ising_config::IsingRunConfig::validate), rather than each
/// going through a separate, narrower enum. That keeps each rule where it can
/// say what went wrong, and admitting a further schedule is a change to
/// validation rather than a new type and a mapping between the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdaterKind {
    /// Single-variable-flip Metropolis on the CPU
    /// ([`Metropolis`](crate::updater::Metropolis)).
    Metropolis,
    /// Single-spin-flip Metropolis under a site checkerboard schedule, on the
    /// CPU ([`SiteCheckerboard`](crate::updater::SiteCheckerboard)); serializes
    /// as `site_checkerboard`.
    SiteCheckerboard,
    /// Single-link-flip Metropolis under a link checkerboard schedule, on the
    /// CPU ([`LinkCheckerboard`](crate::updater::LinkCheckerboard)); serializes
    /// as `link_checkerboard`.
    LinkCheckerboard,
    /// The site checkerboard schedule run on the GPU
    /// ([`GpuIsingChain`](crate::ising_gpu::GpuIsingChain)); serializes as
    /// `gpu_site_checkerboard`. Backend and schedule are one choice rather than
    /// two, which keeps illegal combinations unrepresentable — there is no way
    /// to name a GPU run without also naming the coloring its kernel is written
    /// for.
    GpuSiteCheckerboard,
    /// The link checkerboard schedule run on the GPU
    /// ([`GpuGaugeChain`](crate::gauge_gpu::GpuGaugeChain)); serializes as
    /// `gpu_link_checkerboard`. The link counterpart of
    /// [`GpuSiteCheckerboard`](UpdaterKind::GpuSiteCheckerboard), and fused for
    /// the same reason.
    GpuLinkCheckerboard,
}

impl UpdaterKind {
    /// Which lattice cell this kind's schedule colors, or `None` for a schedule
    /// that reads a bare variable index and so works on either grade.
    ///
    /// This is the one fact behind both schemas' updater rule: a model accepts a
    /// kind when the kind is grade-neutral or names the model's own grade.
    /// Stating it once here — rather than as an allowlist on one side and a
    /// denylist on the other — is what keeps the two from drifting when a kind
    /// is added, since a new variant must answer this question to compile.
    pub fn cell(self) -> Option<Cell> {
        match self {
            // Metropolis draws a uniform index and hands it to a grade-neutral
            // `energy_delta`; nothing in it knows a site from a link.
            UpdaterKind::Metropolis => None,
            UpdaterKind::SiteCheckerboard | UpdaterKind::GpuSiteCheckerboard => Some(Cell::Site),
            UpdaterKind::LinkCheckerboard | UpdaterKind::GpuLinkCheckerboard => Some(Cell::Link),
        }
    }

    /// Whether this kind colors in parallel, so that its coloring must be
    /// collision-free *simultaneously* and every extent therefore has to be
    /// even.
    ///
    /// The requirement follows from the pass running concurrently, not from the
    /// work happening on a device — a parallel CPU schedule would need it too.
    /// Both config schemas ask this in `validate` so an odd shape is a load-time
    /// error naming the axis, rather than a panic when the device chain is
    /// built. The device constructors still assert it, since they are reachable
    /// without a config at all.
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
/// The dimension floor is the one thing the two schemas genuinely disagree
/// about, and it follows from which cell the model's energy scores. Ising scores
/// links, which exist in one dimension; the gauge action scores plaquettes,
/// which need a pair of directions and so two. Below its bound a model does not
/// fail loudly — the cell count goes to zero and the energy comes out
/// identically zero — so the bound has to be stated rather than left to the
/// arithmetic. `scores` names the cell for the message.
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
/// The two rules both schemas apply, in one place. First the grade: a schedule
/// colors a particular kind of cell (see [`UpdaterKind::cell`]) and only a
/// grade-neutral one or a matching one can run here — the site checkerboard's
/// coordinate parity says nothing about a variable on a link, and a run under it
/// would not be a slower version of the same physics, it would be wrong. Then
/// the shape: a parallel color pass needs every extent even, because under
/// periodic boundaries an odd extent wraps a variable onto a same-color one, so
/// a color stops being collision-free and detailed balance breaks silently
/// rather than loudly. See `docs/metropolis.md`. Sequential schedules have no
/// such requirement, which is why the second rule asks about the updater and not
/// about the shape alone.
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
