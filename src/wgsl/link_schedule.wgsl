// The link checkerboard schedule fragment: the entry point that maps a thread
// to the link it owns and hands it to the kernel's `update`.
//
// Variables live on links, and the unit of interaction is the plaquette, so
// what two threads must avoid is sharing a plaquette, not sharing a bond.
// Base-site parity alone cannot deliver that — two of a plaquette's four links
// share a base site — so a link is colored by its direction *and* its base
// site's parity: `2D` colors, and a sweep is `2D` dispatches rather than two.
//
// Threads are launched per site, not per link: a dispatch owns one direction
// of `D`, so a per-link launch would idle all but one thread in `2D`, while
// per-site idles one in two. The link a thread owns is `site * D + dir`,
// since links are packed with direction fastest.
//
// Fixing the direction freezes every link of another direction, and a
// plaquette then holds exactly two links of the pass's direction, whose base
// sites differ by one step and so carry opposite parity. Nothing a thread
// reads can change while it runs; see `docs/metropolis.md` for the argument.
//
// All `2D` passes of one sweep share `pc.sweep`; their link sets are disjoint,
// so no link draws twice in a sweep.

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

    let link = site * D + (pc.color >> 1u);
    update(link);
}
