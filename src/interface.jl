# ============================================================================
# Abstract type hierarchy for the composer nodes
# ============================================================================
#
# The composer nodes share one supertype, following the `AbstractOneOf` model
# (concrete types subtype the abstract; shared behaviour and the documented
# interface contract hang off the abstract):
#
#   AbstractComposedDistribution{F, S} — combine named child distributions into
#     an event tree (the `child_*` node interface). Spans both variate forms:
#     the multivariate event-tree composers (`Sequential`, `Parallel`, `Choose`)
#     and the univariate marginal one_of family (`AbstractOneOf`: `Resolve`,
#     `Compete`).
#
#   AbstractMultiChild{S} — the two positional multi-child composers
#     (`Sequential`, `Parallel`) the tree walkers dispatch over together.
#
# The abstract is parametric on variate form `F` (`Univariate` / `Multivariate`)
# so one supertype spans the univariate and multivariate members while
# preserving `Distribution{F, S}` — the `UnivariateDistribution{S}` alias for the
# univariate `AbstractOneOf` members stays intact, so existing dispatch is
# unchanged. Downstream extension packages (CensoredDistributions and its
# siblings) dispatch on these supertypes, so the names and shape match the shared
# contract.

@doc """
    AbstractComposedDistribution{F<:VariateForm, S<:ValueSupport}

Supertype of the composer nodes that combine named child distributions into an
event tree: the multivariate [`Sequential`](@ref) / [`Parallel`](@ref) /
[`Choose`](@ref) and the univariate one_of family
([`AbstractOneOf`](@ref): [`Resolve`](@ref) / [`Compete`](@ref)). Parametric on
variate form so the one supertype spans both.

Required methods a concrete subtype implements (the node interface):

- `child_nleaves(c)`, `child_logpdf(c, x, offset, n)`,
  `child_rand!(out, offset, rng, c)` — walk the flat event vector;
- `component_names(c)` — the child names;
- `params(c)` and `params_table(c)`;
- `event_names(c)` (flat) and `event_tree(c)` (nested);
- `Base.show(io, c)`.

Verify a subtype with
`ComposedDistributions.TestUtils.test_composed_interface`.
"""
abstract type AbstractComposedDistribution{F <: VariateForm,
    S <: ValueSupport} <: Distribution{F, S} end

@doc """
    AbstractMultiChild{S<:ValueSupport}

Supertype of the positional multi-child composers [`Sequential`](@ref) and
[`Parallel`](@ref) (subtype of
`AbstractComposedDistribution{Multivariate, S}`). These two store `.components` /
`.names` and are walked positionally by the tree machinery, so they share
dispatch on `::AbstractMultiChild` (the supertype the tree walkers key off).
[`Choose`](@ref) (disjoint alternatives) is a sibling, not a multi-child node.
"""
abstract type AbstractMultiChild{S <: ValueSupport} <:
              AbstractComposedDistribution{Multivariate, S} end
