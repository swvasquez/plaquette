// The Z2 gauge model snippet: the model's side of the device seam. It owns
// the two bindings that are about the model — the staple table and the
// uniform block — and the one function the kernel fragments call,
// `energy_delta`. The variables live on links, but nothing here says so: `v`
// is a bare index into `vars`, and it is the *schedule* fragment that hands in
// link indices, exactly as the CPU checkerboard hands them to `step`.

// Plaquettes containing a given link, and the flattened width of its staple
// row: `2(D - 1)` groups of the three other links of each.
override STAPLE_GROUPS: u32 = 2u * (D - 1u);
override STAPLE_STRIDE: u32 = 6u * (D - 1u);

// The static run parameters. The head — `n_sites`, `seed`, `beta` — is the
// layout contract with the kernel and schedule fragments; the tail is this
// model's own.
struct Params {
    n_sites: u32,
    seed: u32,
    beta: f32,
    j: f32,
};

@group(0) @binding(1) var<storage, read> staples: array<u32>;          // STAPLE_STRIDE per link
@group(0) @binding(2) var<uniform> params: Params;

// ΔE for moving link `v` from `current` to `proposed` — the flip, at two
// states. Sum over the link's staple groups of the product of the group's
// three links: the plaquettes containing the link each split into this link's
// own variable times its staple, so this is the whole of H that depends on
// it. Matches the CPU's `-j * ds * staple_sum` at `ds = -2σ_l`.
fn energy_delta(v: u32, current: u32, proposed: u32) -> f32 {
    var staple_sum: i32 = 0;
    for (var g = 0u; g < STAPLE_GROUPS; g = g + 1u) {
        let base = v * STAPLE_STRIDE + g * 3u;
        var product: i32 = 1;
        for (var k = 0u; k < 3u; k = k + 1u) {
            product = product * spin_pm(vars[staples[base + k]]);
        }
        staple_sum = staple_sum + product;
    }
    return 2.0 * params.j * f32(spin_pm(current)) * f32(staple_sum);
}
