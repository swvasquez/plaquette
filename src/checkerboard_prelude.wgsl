// Shared preamble for every checkerboard kernel: the state encoding and the
// counter-based random source.
//
// Prepended to each model's shader at build time (see `device::shader_source`),
// because WGSL has no include and these must stay byte-identical across
// backends. The CPU-versus-GPU tests are distributional, so a hash that drifted
// in one kernel alone would bias it without obviously failing.

// Threads per workgroup, matching the `@workgroup_size(64)` both kernels declare
// and the `WORKGROUP_SIZE` the host dispatches with.
const WORKGROUP_SIZE: u32 = 64u;

// The variable a thread owns, from a dispatch grid that may be two-dimensional.
//
// A single row of workgroups is capped at `max_compute_workgroups_per_dimension`
// — 65535 on essentially every adapter, which is an API-level ceiling rather
// than a conservative default, so no limits request raises it. Past about four
// million threads the host spreads the launch over a rectangle instead, and this
// folds the two axes back into the linear index the tables are keyed by. Rows
// are full by construction, so the mapping is contiguous and gap-free; the
// leftovers at the end return early on the caller's bounds check.
fn linear_index(gid: vec3<u32>, grid: vec3<u32>) -> u32 {
    return gid.x + gid.y * grid.x * WORKGROUP_SIZE;
}

// index 0 -> +1, index 1 -> -1 (matches the CPU `decode` map).
fn spin_pm(s: u32) -> i32 {
    return 1 - 2 * i32(s);
}

// A 32-bit integer finalizer (the "lowbias32" hash) — good mixing, cheap, and
// identical in Rust and WGSL since u32 arithmetic wraps in both.
fn lowbias32(x: u32) -> u32 {
    var v = x;
    v = v ^ (v >> 16u);
    v = v * 0x7feb352du;
    v = v ^ (v >> 15u);
    v = v * 0x846ca68bu;
    v = v ^ (v >> 16u);
    return v;
}

// A uniform in [0, 1) keyed by (seed, variable, sweep), with 24 bits of
// mantissa. Keying on the variable rather than drawing from a stream is what
// makes a sweep's result independent of thread order.
fn keyed_uniform(seed: u32, variable: u32, sweep: u32) -> f32 {
    var k = seed;
    k = lowbias32(k ^ variable);
    k = lowbias32(k ^ (sweep * 0x9e3779b9u));
    return f32(k >> 8u) * (1.0 / 16777216.0);
}
