# The Ising model

The Ising model is a lattice of two-state spins with a nearest-neighbour
interaction. Three pieces define it: the lattice that lays out the sites and says
which are neighbours, the spin that lives on each site, and the energy that scores
a whole configuration.

The lattice is a $D$-dimensional grid of $N$ sites with periodic boundaries, so
every site is equivalent and has the same neighbours — one forward and one
backward along each axis, $2D$ in all. The square lattice in two dimensions is the
usual case, but nothing in the model fixes the dimension. What the lattice
provides is the adjacency: the set of nearest-neighbour pairs, or bonds, that the
interaction acts across.

On each site sits a single spin taking one of two values,

$$s_i = \pm 1.$$

A choice of spin at every site is a configuration, and there are $2^N$ of them.

The energy assigns a number to each configuration, a nearest-neighbour coupling
together with a uniform external field,

$$H = -J \sum_{\langle i,j \rangle} s_i s_j \; - \; h \sum_i s_i,$$

where the first sum runs over each bond once and the second over all sites. The
coupling $J$ sets the interaction strength — $J > 0$ is ferromagnetic, so aligned
neighbours lower the energy — and the field $h$ favours one orientation, vanishing
for the field-free model.
