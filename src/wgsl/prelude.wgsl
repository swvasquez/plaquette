// Shared prelude for every assembled local-update shader: the declarations the
// model, kernel, and schedule fragments rest on.
//
// A compiled shader is the concatenation of five fragments — `rng.wgsl`, this
// prelude, a model snippet, a kernel, and a schedule — mirroring the CPU
// composition of `LocalUpdate` one-to-one (see `device::assemble_shader`).
// WGSL has no include, so the concatenation happens on the host, and
// module-scope declarations need no forward order, so each fragment may refer
// to the others' names.

// The lattice dimension, resolved when the pipeline is built so that loop
// bounds derived from it are still constants when the shader is translated.
override D: u32;

// The evolving configuration, one `u32` state index per variable — sites and
// links look identical here, which is the device half of the kernel's
// grade-neutrality. The kernel owns the write; the model snippet reads it to
// price a move.
@group(0) @binding(0) var<storage, read_write> vars: array<u32>;

// Each site's coordinate-sum parity, uploaded rather than derived so both
// backends read one coloring and no geometry lives in any kernel.
@group(0) @binding(3) var<storage, read> site_color: array<u32>;

// Per-dispatch state: the global sweep index keys the RNG, and the color says
// which variables this dispatch owns.
struct Push {
    sweep: u32,
    color: u32,
};
var<push_constant> pc: Push;
