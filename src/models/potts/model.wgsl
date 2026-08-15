// The Potts model snippet: the model's side of the device seam. It owns the
// two bindings that are about the model — the neighbor table and the uniform
// block with its per-label offsets — and the one function the kernel
// fragments call, `energy_delta`.
//
// This file is a *template*, not a finished snippet: the host substitutes
// `$H_VECTORS$` before assembly (`potts_gpu::shader_for`). That is the one
// thing here that cannot ride an `override` the way the dimension does — an
// override cannot size an array in the uniform address space, and the
// per-label offsets are such an array. Packing four offsets into a `vec4`
// rather than declaring `array<f32, N>` avoids WGSL's sixteen-byte stride on
// uniform array elements, which would otherwise waste three words in four.

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

// How many vec4 slots the per-label offsets occupy: four labels to a slot,
// rounded up, substituted by the host.
const H_VECTORS: u32 = $H_VECTORS$u;

// The static run parameters. The head — `n_sites`, `seed`, `beta` — is the
// layout contract with the kernel and schedule fragments; the tail is this
// model's own. `q` rides here as a runtime uniform for completeness, but the
// kernels take the state count as the substituted `Q_STATES` constant, which
// is the same number by construction (`GpuChain` fills both from `Q`).
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
    q: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    h: array<vec4<f32>, H_VECTORS>,
};

@group(0) @binding(1) var<storage, read> neighbors: array<u32>;        // 2D per site: +0,-0,+1,-1,...
@group(0) @binding(2) var<uniform> params: Params;

// One label's energy offset, unpacked from the vec4 it was folded into.
fn offset_for(label: u32) -> f32 {
    return params.h[label / 4u][label % 4u];
}

// ΔE for moving site `v` from `current` to `proposed`, matching the CPU:
// `-j * (agree_after - agree_before) - (h[proposed] - h[current])`. Only the
// bonds touching this site change, and each counts how many neighbors carry
// the proposed or the current label; the two labels differ, so a neighbor
// lands in at most one of the counts. The offset term reads only this site's
// own two entries, which is why a constant added to every entry cancels here
// as it does on the host.
fn energy_delta(v: u32, current: u32, proposed: u32) -> f32 {
    var change: i32 = 0;
    for (var d = 0u; d < N_NEIGHBORS; d = d + 1u) {
        let s = vars[neighbors[v * N_NEIGHBORS + d]];
        if (s == proposed) {
            change = change + 1;
        }
        if (s == current) {
            change = change - 1;
        }
    }
    let offset = offset_for(proposed) - offset_for(current);
    return -params.j * f32(change) - offset;
}
