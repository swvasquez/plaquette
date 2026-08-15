// The relabel stage of a Swendsen-Wang sweep for the q-state Potts model — the
// one kernel of the four that knows what a state is.
//
// Everything before it (`cluster_prelude.wgsl`) is graph work: which bonds are
// open, and which sites the open bonds connect. This is where the labels come
// back, and what it does is Potts's own business — draw a fresh label uniformly
// from all `q`, including the one the cluster already carried. A model whose
// move is not a redraw would write a different kernel here and reuse the three
// stages unchanged: an Ising cluster backend flips, and a clock model reflects
// about an axis rather than resampling at all.
//
// A *redraw* rather than a flip is worth stating, since at `q = 2` the textbook
// presentation says "flip each cluster with probability one half". The two are
// the same move written differently — a cluster that redraws its own label is
// the half of the time a flip does not fire — and redrawing is what generalizes
// past two states.

// Keys the label draw off a different stream from the bond draws. Same role as
// `BOND_KEY` in the prelude, and the same reasoning: the two must be independent,
// or a cluster's label would correlate with which of its bonds happened to open.
const ROOT_SALT: u32 = 0x85ebca6bu;

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
