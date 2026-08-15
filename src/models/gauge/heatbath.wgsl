// Checkerboard heat bath sweep for the Z2 gauge theory, in any dimension of at
// least two.
//
// The sibling of this model's `checkerboard.wgsl`, identical to it through the
// staple sum: same bindings, same bounds check, same `(direction, base parity)`
// color guard, same walk over the link's staple groups. The coloring is the
// schedule's and not the kernel's, so the heat bath inherits the independence
// argument unchanged — see `checkerboard.wgsl`, where it is made.
//
// What differs is the decision at the end. A link variable takes two values, so
// the conditional is the same one-expression form the Ising heat bath uses, and
// the staple sum plays the part the neighbor sum plays there. `docs/heat-bath.md`
// describes the update.

override D: u32;

// Staple groups per link and the row width of the `staples` table. Both are
// fixed by the dimension, resolved when the pipeline is built.
override STAPLE_GROUPS: u32 = 2u * (D - 1u);
override STAPLE_STRIDE: u32 = 6u * (D - 1u);

struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    color: u32,   // (direction << 1) | base-site parity
};

@group(0) @binding(0) var<storage, read_write> links: array<u32>;      // 0 or 1 per link
@group(0) @binding(1) var<storage, read> staples: array<u32>;          // 3 links per staple group
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> site_color: array<u32>;       // coordinate-sum parity
var<push_constant> pc: Push;

@compute @workgroup_size(64)
fn sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }

    // This dispatch touches only links whose base carries the pass's parity.
    if (site_color[site] != (pc.color & 1u)) {
        return;
    }

    let dir = pc.color >> 1u;
    let link = site * D + dir;

    // Sum over the link's staple groups of the product of the group's three
    // links. The plaquettes containing the link each split into this link's own
    // variable times its staple, so this is the whole of H that depends on it.
    var staple_sum: i32 = 0;
    for (var g = 0u; g < STAPLE_GROUPS; g = g + 1u) {
        let base = link * STAPLE_STRIDE + g * 3u;
        var product: i32 = 1;
        for (var k = 0u; k < 3u; k = k + 1u) {
            product = product * spin_pm(links[staples[base + k]]);
        }
        staple_sum = staple_sum + product;
    }

    // ΔE for the flip, matching the CPU's `-j * ds * staple_sum` at ds = -2σ_l.
    let current = links[link];
    let dE = 2.0 * params.j * f32(spin_pm(current)) * f32(staple_sum);

    // The conditional probability of landing on the flipped link, in the same
    // rearranged form the Ising heat bath uses and for the same reason: the two
    // ways it can overflow in f32 both saturate toward the correct answer.
    let p_flip = 1.0 / (1.0 + exp(params.beta * dE));

    if (keyed_uniform(params.seed, link, pc.sweep) < p_flip) {
        links[link] = 1u - current;
    }
}
