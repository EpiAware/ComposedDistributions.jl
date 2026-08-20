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

Required methods a concrete subtype implements (the node interface; see
[Adding a valid composer node](@ref new-composer-node) for the full contract
and a worked example):

- [`node_children`](@ref)`(node)` and [`node_rebuild`](@ref)`(node, children)`
  — the node's children and how to rebuild around new ones;
- [`component_names`](@ref)`(node)` — the child names.

Those three alone are enough for a node to compose, table
([`composed_to_table`](@ref)), and `update`; [`params`](@ref),
`child_nleaves`/`child_logpdf`/`child_rand!` (the flat event-vector walk) and
the flat-vector codec (`flat_dimension`/`flatten`/`unflatten`/`reconstruct`)
all derive from them generically for a plain "concatenating" node (one whose
realisation is just its children's laid end to end, `Sequential`/`Parallel`'s
own semantics). A node with different combination semantics (a disjunction, a
mixture) overrides `child_nleaves`/`child_logpdf`/`child_rand!` directly, as
[`Choose`](@ref) does. Codec (flatten/unflatten/fit) support additionally
needs the node's own child names and children's types as its first two type
parameters — see [`node_children`](@ref)'s docstring for why.

Not yet part of the generic contract (open, tracked in follow-up issues): a
downstream node's `event_names`/`event_tree` (flat/nested event naming),
top-level `rand`/`logpdf` when the node is used as a standalone root
distribution, and `Base.show`.

Verify a subtype with
`ComposedDistributions.TestUtils.test_node_interface` (the node contract) and
`ComposedDistributions.TestUtils.test_composed_interface` (the fuller public
checklist too, for a node also exercising `rand`/`logpdf`/moments as a root).
"""
abstract type AbstractComposedDistribution{F <: VariateForm,
    S <: ValueSupport} <: Distribution{F, S} end

@doc """
    AbstractMultiChild{S<:ValueSupport}

Supertype of the positional multi-child composers [`Sequential`](@ref) and
[`Parallel`](@ref) (subtype of
`AbstractComposedDistribution{Multivariate, S}`). These two store `.components`
and carry their child names in a `names` type parameter (read with
[`component_names`](@ref)), and are walked positionally by the tree machinery,
so they share dispatch on `::AbstractMultiChild` (the supertype the tree
walkers key off). [`Choose`](@ref) (disjoint alternatives) is a sibling, not a
multi-child node.
"""
abstract type AbstractMultiChild{S <: ValueSupport} <:
              AbstractComposedDistribution{Multivariate, S} end
