# Public API declarations for Julia 1.11+ (public but not exported).

# The composer node/leaf extension contract: a new node implements
# `child_nleaves` / `child_logpdf` / `child_rand!`, and a new leaf wrapper
# `free_leaf` / `rewrap_leaf`. `component_names` reads a node's child names.
public child_nleaves, child_logpdf, child_rand!
public free_leaf, rewrap_leaf, component_names

# The shared one_of-outcome supertype (`Resolve` / `Compete`).
public AbstractOneOf
