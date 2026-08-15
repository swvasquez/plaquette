// The relabel stage of a cluster sweep — the one kernel of the four that knows
// what a state is.
//
// Everything before it (`cluster_prelude.wgsl`) is graph work: which bonds are
// open, and which sites the open bonds connect. This is where the labels come
// back, under the two axes the host composes (`Params.seeded` and
// `Params.forced`, from the CPU `Extent` and `Relabel`):
//
//   - Swendsen–Wang (`seeded = 0`, `forced = 0`) relabels every cluster with a
//     fresh label drawn uniformly from all `q`, including the one the cluster
//     already carried.
//   - Wolff (`seeded = 1`, `forced = 1`) relabels only the cluster holding a
//     uniformly drawn seed site, onto a label forced to differ.
//
// The device gets Wolff's seeded extent for free from the full decomposition
// the prelude already built: picking the cluster under a uniformly random site
// *is* Wolff's size-biased cluster choice, so relabeling just that one is the
// same move, at the cost of labeling work the single-cluster CPU growth
// avoids. On a data-parallel device that labeling is the part that runs well,
// which is why the seeded extent is a filter here rather than a frontier walk.
//
// A *redraw* rather than a flip is worth stating for the Swendsen–Wang case,
// since at `q = 2` the textbook presentation says "flip each cluster with
// probability one half". The two are the same move written differently — a
// cluster that redraws its own label is the half of the time a flip does not
// fire — and redrawing is what generalizes past two states. The forced change
// is the other bookkeeping: uniform over the `q - 1` other labels, which at
// `q = 2` *is* the flip.

// Keys the label draw off a different stream from the bond draws. Same role as
// `BOND_KEY` in the prelude, and the same reasoning: the two must be independent,
// or a cluster's label would correlate with which of its bonds happened to open.
const ROOT_SALT: u32 = 0x85ebca6bu;

// Keys the seed-site draw off a third stream, for the same reason again: the
// seeded cluster choice must not correlate with the labels or the bonds.
const SEED_SALT: u32 = 0xc2b2ae35u;

@compute @workgroup_size(64)
fn relabel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) grid: vec3<u32>,
) {
    let site = linear_index(gid, grid);
    if (site >= params.n_sites) {
        return;
    }

    // Keyed on the cluster's root rather than on the site: every member of a
    // cluster reads the same key, so every member draws the same label with no
    // communication whatsoever. That is what makes this stage embarrassingly
    // parallel despite each cluster having to move as one.
    let root = atomicLoad(&cluster[site]);

    if (params.seeded == 1u) {
        // Every thread computes the same seed site from the same keyed draw,
        // then keeps going only if it shares the seed's cluster. The clamp
        // mirrors the one on the label draw below.
        var seed_site = u32(keyed_uniform(params.seed ^ SEED_SALT, 0u, pc.sweep) * f32(params.n_sites));
        if (seed_site >= params.n_sites) {
            seed_site = params.n_sites - 1u;
        }
        if (root != atomicLoad(&cluster[seed_site])) {
            return;
        }
    }

    if (params.forced == 1u) {
        // Uniform over the q - 1 other labels, stepped past the current one.
        // Every member holds the same current label — bonds open only between
        // agreeing sites — so the shared keyed draw lands them all together.
        var step = u32(keyed_uniform(params.seed ^ ROOT_SALT, root, pc.sweep) * f32(params.n_states - 1u));
        if (step >= params.n_states - 1u) {
            step = params.n_states - 2u;
        }
        labels[site] = (labels[site] + 1u + step) % params.n_states;
        return;
    }

    // The clamp is not paranoia: `keyed_uniform` returns the largest float below
    // one, and scaling it by `n_states` can round *up* to exactly `n_states`,
    // which would put the label one past the last one that exists. The host's
    // read-back would reject the word, so this would be a loud failure rather
    // than a silent one — but it would be a failure.
    var draw = u32(keyed_uniform(params.seed ^ ROOT_SALT, root, pc.sweep) * f32(params.n_states));
    if (draw >= params.n_states) {
        draw = params.n_states - 1u;
    }
    labels[site] = draw;
}
