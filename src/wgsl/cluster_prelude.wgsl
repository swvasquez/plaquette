// Shared preamble for every cluster kernel: the buffers a Swendsen-Wang sweep
// works over, and the three stages that are pure graph work.
//
// Compiled after `wgsl/rng.wgsl`, whose `linear_index`, `lowbias32`
// and `keyed_uniform` it reuses rather than restating — every kernel in the
// crate draws from the same hash, so a device run and a host run differ in how
// they consume randomness and never in what the randomness is.
//
// Nothing below knows what a label means. `bonds_pass` compares two of them for
// equality, `label_init` and `propagate` never look at one at all, and the model
// supplies the fourth stage — `relabel` — which is the only place a state count
// or a proposal appears. That split is the point: a future Ising or clock-model
// cluster backend reuses this file whole and writes only its own relabel step.
//
// The two counts a stage is bounded by both derive from the dimension: a site
// has `2D` neighbors and a lattice has `D * n_sites` forward bonds, one per site
// per axis, indexed exactly as a link is (`site * D + axis`).

override D: u32;

// Neighbors per site, one forward and one backward along each direction — the
// row width of the `neighbors` table.
override N_NEIGHBORS: u32 = 2u * D;

// Keys the bond draw off a different stream from the relabel draw, the same way
// the checkerboard kernels separate their proposal and accept draws. Any
// constant unrelated to the ones inside `keyed_uniform` does the job.
const BOND_KEY: u32 = 0x27d4eb2fu;

struct Params {
    n_sites: u32,
    n_bonds: u32,       // D * n_sites
    seed: u32,
    p: f32,             // bond probability, 1 - exp(-beta * gap), computed on the host
    n_states: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

struct Push {
    sweep: u32,   // global sweep index, the RNG counter
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> labels: array<u32>;              // 0..n_states per site
@group(0) @binding(1) var<storage, read> neighbors: array<u32>;                 // 2D per site: +0,-0,+1,-1,...
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> bonds: array<u32>;               // 1 open, 0 closed, per bond
@group(0) @binding(4) var<storage, read_write> cluster: array<atomic<u32>>;     // a site index naming the cluster
@group(0) @binding(5) var<storage, read_write> changed: atomic<u32>;            // set by any propagation that moved
var<push_constant> pc: Push;

// Stage 1. One thread per bond: open it when the two sites agree and the draw
// falls below `p`.
//
// The host computes `p` because it is constant over a run, so nothing here needs
// beta or a coupling. Keying the draw on the bond index rather than on a stream
// is what makes the result independent of thread order, as everywhere else.
@compute @workgroup_size(64)
fn bonds_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let bond = linear_index(gid, grid);
    if (bond >= params.n_bonds) {
        return;
    }

    // A bond is named by the site it leaves and the axis it runs along, so the
    // partner is that site's forward neighbor along that axis.
    let site = bond / D;
    let axis = bond % D;
    let partner = neighbors[site * N_NEIGHBORS + 2u * axis];

    let agree = labels[site] == labels[partner];
    let drawn = keyed_uniform(params.seed ^ BOND_KEY, bond, pc.sweep) < params.p;
    bonds[bond] = select(0u, 1u, agree && drawn);
}

// Stage 2. Every site starts as its own cluster.
@compute @workgroup_size(64)
fn label_init(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }
    atomicStore(&cluster[site], site);
}

// Stage 3, run repeatedly until nothing moves. Each pass lowers a site's cluster
// value to the smallest it can see: its own, its bonded neighbors', and one
// pointer jump past whichever of those won.
//
// Every value a site can take is a site index in its own cluster — its own to
// begin with, a bonded neighbor's or that neighbor's own value thereafter — so
// no pass can merge two clusters that are not connected. The values only ever
// decrease, so the iteration terminates; the pointer jump is what makes it
// terminate in about a logarithm rather than a lattice diameter.
//
// At the fixed point every bonded pair holds the same value, so the value is
// constant across a cluster and, being a site of that cluster, equals its own
// entry. That constant is the "root" the relabel stage keys on.
//
// The host is in the loop because a dispatch has no global barrier: a pass
// cannot see whether *another workgroup* moved, so convergence can only be
// observed between dispatches. `changed` is read back rather than branched on
// here for the same reason.
@compute @workgroup_size(64)
fn propagate(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }

    var best = atomicLoad(&cluster[site]);
    for (var axis = 0u; axis < D; axis = axis + 1u) {
        let forward = neighbors[site * N_NEIGHBORS + 2u * axis];
        if (bonds[site * D + axis] == 1u) {
            best = min(best, atomicLoad(&cluster[forward]));
        }
        // The backward bond is the neighbor's forward one, since a bond is named
        // by the end it points away from.
        let backward = neighbors[site * N_NEIGHBORS + 2u * axis + 1u];
        if (bonds[backward * D + axis] == 1u) {
            best = min(best, atomicLoad(&cluster[backward]));
        }
    }
    best = min(best, atomicLoad(&cluster[best]));

    // Only this thread writes this site, so `old` is the value the pass started
    // from and the comparison cannot miss a move it made itself. A move some
    // *other* thread has yet to make is caught by the next pass, which is why a
    // pass reporting no change means the state it started from was already the
    // fixed point.
    let old = atomicMin(&cluster[site], best);
    if (best < old) {
        atomicStore(&changed, 1u);
    }
}
