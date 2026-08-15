// The site checkerboard schedule fragment: the entry point that maps a thread
// to the variable it owns and hands it to the kernel's `update`.
//
// One dispatch updates one color: a thread per site, acting only on sites
// whose coordinate-sum parity matches `pc.color`. Within a color no two sites
// are neighbors, so every thread reads only the *other* color — untouched
// during this dispatch — and there is no read/write conflict. A full sweep is
// two dispatches, color 0 then color 1, with the pass boundary as the barrier.
// That argument reads only the adjacency, so it holds for any model snippet
// whose `energy_delta` reads nearest neighbors; see `docs/metropolis.md`.
//
// Both color passes of one sweep share `pc.sweep`; their site sets are
// disjoint, so no site draws twice in a sweep.

@compute @workgroup_size(64)
fn sweep(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }

    // This dispatch touches only its color.
    if (site_color[site] != pc.color) {
        return;
    }

    update(site);
}
