# Public API declarations for Julia 1.11+ (public but not exported).

# The composer node/leaf extension contract: a new node implements
# `child_nleaves` / `child_logpdf` / `child_rand!`, and a new leaf wrapper
# `free_leaf` / `rewrap_leaf`. `component_names` reads a node's child names.
public child_nleaves, child_logpdf, child_rand!
public free_leaf, rewrap_leaf, component_names

# The composer abstract-type hierarchy. `AbstractComposedDistribution` is the
# root the composer nodes subtype; `AbstractMultiChild` groups the positional
# multi-child composers (`Sequential` / `Parallel`); `AbstractOneOf` is the
# univariate one_of arm (`Resolve` / `Compete`). Downstream extension packages
# dispatch on these.
public AbstractComposedDistribution, AbstractMultiChild, AbstractOneOf

# The compact `(outcome, time)` pair view of a one_of draw. A bare `rand` of a
# `Resolve` / `Compete` returns the full named event record (the primary,
# self-describing draw); `rand_outcome` is the two-tuple view for when only the
# winning name and time are wanted. Public but not exported (the record-returning
# `rand` is the exported entry point).
public rand_outcome

# The reusable interface-conformance harness (`TestUtils.test_interface` and
# the per-family `test_*` checks).
public TestUtils

# The Mellin product type (`Z = X * Y`) re-exported from ConvolvedDistributions.
# Its `product` constructor is exported, but the `Product` type stays unexported
# because a bare `Product` would clash with Distributions' deprecated `Product`;
# reach it as `ComposedDistributions.Product`.
public Product

# The LogDensityProblems core codec: the flat-vector <-> nested-NamedTuple
# bijection (`flat_dimension`/`flatten`/`unflatten`) and the assembled
# PPL-neutral log-density (`ComposedLogDensity`/`as_logdensity`/`logdensity`).
# A weakdep `LogDensityProblems` extension wraps `ComposedLogDensity` as a
# standard problem; this core stays Turing/LogDensityProblems-free.
public flat_dimension, flatten, unflatten
public ComposedLogDensity, as_logdensity, logdensity
