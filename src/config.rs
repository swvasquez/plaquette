//! Config vocabulary shared by the per-model run configs
//! ([`IsingRunConfig`](crate::models::ising::run_config::IsingRunConfig) and
//! [`GaugeRunConfig`](crate::models::gauge::run_config::GaugeRunConfig)): how a
//! load can fail, where a chain starts, and which algorithm advances it.
//!
//! The schemas stay separate types — one wide enough for both would carry a
//! hole for whichever model is running.

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

/// Which update rule advances the chain — the serializable choice of the
/// *algorithm*, one third of the vocabulary a run is named in.
///
/// The other two thirds are [`ScheduleKind`] and [`BackendKind`], and the
/// three fields are orthogonal, so a config file reads the way the physics is
/// spoken: "heat bath under a checkerboard schedule on the GPU". Nothing here
/// names a cell: whether a checkerboard colors sites or links follows from the
/// model being run, so a config cannot pair a schedule with the wrong grade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdaterRule {
    /// Propose one alternative per variable and accept with `min(1, e^{-β ΔE})`
    /// — [`Kernel::Metropolis`](crate::updater::Kernel::Metropolis) under the
    /// run's schedule.
    Metropolis,
    /// Price every state a variable could take and draw from the conditional
    /// they define — [`Kernel::HeatBath`](crate::updater::Kernel::HeatBath)
    /// under the run's schedule.
    HeatBath,
    /// The Swendsen–Wang cluster update — [`ClusterUpdate`](crate::updater::ClusterUpdate)
    /// with [`Extent::All`](crate::updater::Extent::All) and
    /// [`Relabel::Redraw`](crate::updater::Relabel::Redraw). Not a kernel under
    /// a schedule, so a config naming it must leave `schedule` unset.
    ///
    /// It runs on any lattice shape, on either backend (the device chain,
    /// [`GpuClusterChain`](crate::gpu_cluster::GpuClusterChain), names no
    /// model), and needs a model whose energy is invariant under relabeling —
    /// which each schema checks against its own field or offsets, since what
    /// breaks the symmetry is the model's business and not this enum's.
    SwendsenWang,
    /// The Wolff cluster update — [`ClusterUpdate`](crate::updater::ClusterUpdate)
    /// with [`Extent::Seeded`](crate::updater::Extent::Seeded) and
    /// [`Relabel::ForcedChange`](crate::updater::Relabel::ForcedChange): one
    /// cluster grown from a random seed, forced onto a different label. The
    /// same schedule, shape, backend, and symmetry rules as
    /// [`SwendsenWang`](UpdaterRule::SwendsenWang) apply; the one difference a
    /// config author must hold in mind is that a Wolff sweep is a single
    /// cluster rather than a pass over the lattice, so `thermalize` and
    /// `sweeps_between` count much smaller units — see `docs/wolff.md`.
    Wolff,
}

impl UpdaterRule {
    /// Whether this rule builds clusters, and so needs a model whose energy is
    /// invariant under relabeling. The load-time counterpart of the panic in
    /// [`ClusterUpdate::new`](crate::updater::ClusterUpdate::new).
    pub fn builds_clusters(self) -> bool {
        matches!(self, UpdaterRule::SwendsenWang | UpdaterRule::Wolff)
    }

    /// The kernel a local rule composes with, or `None` for a cluster rule.
    /// With [`cluster_axes`](UpdaterRule::cluster_axes) this is the whole
    /// config-to-updater mapping for the rule axis, stated once so the
    /// samplers cannot drift from each other.
    pub fn kernel(self) -> Option<crate::updater::Kernel> {
        match self {
            UpdaterRule::Metropolis => Some(crate::updater::Kernel::Metropolis),
            UpdaterRule::HeatBath => Some(crate::updater::Kernel::HeatBath),
            UpdaterRule::SwendsenWang | UpdaterRule::Wolff => None,
        }
    }

    /// The extent and relabel rule a cluster rule composes, or `None` for a
    /// local rule — the counterpart of [`kernel`](UpdaterRule::kernel) for the
    /// other family.
    pub fn cluster_axes(self) -> Option<(crate::updater::Extent, crate::updater::Relabel)> {
        match self {
            UpdaterRule::Metropolis | UpdaterRule::HeatBath => None,
            UpdaterRule::SwendsenWang => {
                Some((crate::updater::Extent::All, crate::updater::Relabel::Redraw))
            }
            UpdaterRule::Wolff => Some((
                crate::updater::Extent::Seeded,
                crate::updater::Relabel::ForcedChange,
            )),
        }
    }
}

/// Which schedule a local rule runs under — the serializable choice of a
/// [`Schedule`](crate::updater::Schedule).
///
/// In a config file the field is optional and means the random schedule when
/// omitted; a run naming a cluster rule ([`UpdaterRule::SwendsenWang`] or
/// [`UpdaterRule::Wolff`]) must leave it unset, since a cluster update has no
/// schedule at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleKind {
    /// `n_vars` uniformly random picks per sweep.
    Random,
    /// Every variable once, color by color — sites by parity, links by
    /// direction and base parity, decided by the model's grade.
    Checkerboard,
}

impl From<ScheduleKind> for crate::updater::Schedule {
    fn from(kind: ScheduleKind) -> Self {
        match kind {
            ScheduleKind::Random => crate::updater::Schedule::Random,
            ScheduleKind::Checkerboard => crate::updater::Schedule::Checkerboard,
        }
    }
}

/// Where a run executes: sequentially on the CPU, or as parallel color passes
/// (or the parallel cluster labeling) on the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Sequential sweeps on the host — the default.
    #[default]
    Cpu,
    /// Device sweeps via `wgpu`. A local rule needs the checkerboard schedule
    /// here, since a random schedule has no colors to run in parallel.
    Gpu,
}

/// The random-when-omitted reading of an optional `schedule` field, in one
/// place so the schemas and samplers agree on it.
pub fn effective_schedule(schedule: Option<ScheduleKind>) -> ScheduleKind {
    schedule.unwrap_or(ScheduleKind::Random)
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

/// Check that the `(updater, schedule, backend)` triple names a runnable
/// combination on a lattice of `shape`, or say what is wrong with it.
///
/// The rules both schemas apply, in one place. A cluster rule takes no
/// schedule, since it is not a kernel under one. A local rule on the GPU needs
/// the checkerboard schedule — the device runs a sweep as parallel color
/// passes, and a random schedule has no colors — and even extents, because an
/// odd extent wraps a variable onto a same-color one, the coloring stops being
/// collision-free, and detailed balance breaks silently; see
/// `docs/metropolis.md`. Sequential (CPU) schedules carry no shape
/// requirement, and neither does the cluster update on either backend: it
/// labels a graph rather than coloring a lattice, so a device cluster run is
/// correct on any shape.
pub(crate) fn check_updater(
    updater: UpdaterRule,
    schedule: Option<ScheduleKind>,
    backend: BackendKind,
    shape: &[usize],
) -> Result<(), ConfigError> {
    if updater.builds_clusters() && schedule.is_some() {
        return Err(ConfigError::Invalid(format!(
            "{updater:?} builds clusters rather than walking variables, so it \
             takes no schedule; remove the schedule field"
        )));
    }
    if backend == BackendKind::Gpu && !updater.builds_clusters() {
        if effective_schedule(schedule) != ScheduleKind::Checkerboard {
            return Err(ConfigError::Invalid(format!(
                "the gpu backend runs {updater:?} as parallel color passes, \
                 which needs schedule = \"checkerboard\""
            )));
        }
        if let Some(axis) = shape.iter().position(|l| !l.is_multiple_of(2)) {
            return Err(ConfigError::Invalid(format!(
                "the gpu checkerboard colors in parallel, which needs even \
                 extents, but shape{shape:?} is odd on axis {axis}"
            )));
        }
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
