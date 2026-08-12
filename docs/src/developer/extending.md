# [Extending ComposedDistributions](@id extending)

This page is the reference for what makes a type a _valid_ participant in composition: a leaf, a leaf wrapper, or a composer node.
It states the exact method contract the package relies on for each role, what each role costs to implement, and what today's contract does and does not yet cover.

The reusable interface-conformance suite `ComposedDistributions.TestUtils` checks these contracts over every built-in shape and the worked examples on this page, and the package runs it in `test/interfaces.jl`, so the prose here and the tests stay in sync.

## The type landscape

The composer nodes share one supertype, `AbstractComposedDistribution{F, S}`.
The named-child composers and the univariate one_of family sit under it; leaves and leaf wrappers are plain univariate distributions under no composer supertype.

| Type | Role |
|---|---|
| `AbstractComposedDistribution{F, S}` | root: named children combine into an event tree |
| `AbstractMultiChild{S}` | positional multi-child composers, tree-walked together |
| `Sequential` | named steps in series (a chain) |
| `Parallel` | named branches off one shared origin (fan-out) |
| `Choose` | data-selected disjoint alternatives (a sibling of `AbstractMultiChild`, not a member) |
| `AbstractOneOf` | one univariate time-to-event marginal |
| `Resolve` | a fixed-probability mixture |
| `Compete` | racing hazards (soonest cause fires) |
| `Shared` | a tied leaf (one free parameter across branches); plain univariate leaf, no composer supertype |
| any `Distributions.jl` `UnivariateDistribution` | a leaf, or the base of a leaf wrapper |

`AbstractComposedDistribution` is parametric on variate form `F` (`Univariate` / `Multivariate`), so one supertype spans the univariate one_of members and the multivariate event-tree composers while preserving `Distribution{F, S}`.
Downstream extension packages (CensoredDistributions and its siblings) dispatch on these supertypes, so the names and shape match the shared contract; `TestUtils.test_abstract_membership` pins the membership down as a test, so a type filed under the wrong supertype fails.

## Adding a leaf: nothing

Any `Distributions.jl` distribution is a valid leaf with no package-specific hooks.
It composes, ties, gets wrapped in `uncertain`, and scores through the flat codec, unchanged.
The only thing it does not get for free is a named parameter table row: `param_names(::Any) = ()` falls back to `:param_1, :param_2, ...` positional names unless the type overrides [`param_names`](@ref).

```@example extending
using ComposedDistributions, Distributions

leaf = Gamma(2.0, 1.0)
tree = compose((onset = leaf, admit = LogNormal(0.5, 0.4)))
composed_to_table(tree)
```

## Adding a leaf wrapper

A leaf wrapper carries fixed structure (censoring bounds, a shared tie, an attached prior) around an inner base distribution.
`inner_dist`/`rewrap_leaf` are the two mandatory hooks; everything else is optional, with a default that is correct for a wrapper that carries *nothing extra* through.
A wrapper that does carry something through and does not override the matching hook loses it silently, which is [issue #277](https://github.com/EpiAware/ComposedDistributions.jl/issues/277)'s whole subject and the reason [`TestUtils.test_leaf_protocol_completeness`](@ref) below exists.

| Method | Role | Default when unimplemented |
|---|---|---|
| [`inner_dist`](@ref) | peel one wrapper layer to the inner distribution | none — mandatory |
| [`rewrap_leaf`](@ref) | rebuild the wrapper around a new inner delay | none — mandatory |
| [`free_leaf`](@ref) | peel to the innermost free delay | recursion through `inner_dist` |
| [`node_attributes`](@ref) | the layer's own fixed, non-parameter structure | `(;)` (none) |
| [`param_names`](@ref) | the inner delay's native parameter names | positional (`:param_1`, ...) |
| [`leaf_param_names`](@ref) | the estimable names, native then extra | `param_names` alone |
| [`leaf_ctor`](@ref) | the constructor that rebuilds the inner delay | the type's own constructor |
| [`uncertain_specs`](@ref) | attached priors, or `nothing` when fixed | `nothing` (fixed) |
| [`shared_tag`](@ref) | the shared tie tag, or `nothing` | `nothing` (untied) |
| [`leaf_mean`](@ref), [`leaf_var`](@ref) | per-leaf moments | the inner free delay's moment |
| [`extra_leaf_params`](@ref) | modifier-owned free parameters and supports | `(;)` (none) |
| [`set_extra_leaf_params`](@ref) | rebuild with new extra values | identity |
| [`leaf_detail_lines`](@ref) | `inspect` rendering lines | the type's own `show` |

The peel and rebuild pair is the base of the protocol.
`inner_dist` peels a single layer, and the read-through hooks (`free_leaf`, `shared_tag`, `uncertain_specs`, `extra_leaf_params`) recurse through it, so one method covers all of them.
`free_leaf` reaches the inner free delay whose parameters are the leaf's free parameters, and `rewrap_leaf` re-applies the fixed structure around a rebuilt delay.
A plain leaf is its own inner distribution and `rewrap_leaf` returns the new inner delay, so the identity holds without a method.
`composed_to_table` folds the same peel into the wrapper layers it lists, one `:node` row each, so a wrapper appears in the table with no further method.

A worked example with a plain leaf, where every peel is the identity and the extra map is empty.

```@example extending
using ComposedDistributions, Distributions

leaf = Gamma(2.0, 1.0)
(free = ComposedDistributions.free_leaf(leaf),
    names = ComposedDistributions.leaf_param_names(leaf),
    specs = ComposedDistributions.uncertain_specs(leaf),
    tag = ComposedDistributions.shared_tag(leaf),
    mean = ComposedDistributions.leaf_mean(leaf),
    extras = ComposedDistributions.extra_leaf_params(leaf))
```

A truncated leaf peels its fixed structure to the inner delay, so the parameter table lists only the inner free parameters, and `leaf_mean` reports the truncated moment (not the inner delay's), because `Truncated` overrides it.
`truncated(Normal(0.0, 1.0); lower = 0.0)` has a Distributions.jl closed-form
truncated mean, so the difference from the untruncated `Normal(0.0, 1.0)`
mean (`0.0`) is visible below, unlike a family such as `Gamma` with no
closed-form truncated moment, where `leaf_mean` falls back to the untruncated
one and the two would look the same.

```@example extending
tr = truncated(Normal(0.0, 1.0); lower = 0.0)
inner = ComposedDistributions.free_leaf(tr)
rebuilt = ComposedDistributions.rewrap_leaf(tr, Normal(1.0, 2.0))
(inner = inner, rebuilt = rebuilt,
    mean = ComposedDistributions.leaf_mean(tr))
```

### Steps

1. Implement `inner_dist` and `rewrap_leaf` so the wrapper peels one layer to its inner distribution and rebuilds around a new inner delay. A tie or an attached prior under the wrapper then reaches the table with no further method, since the read-through hooks follow the same peel.
2. Add a `node_attributes` method when the wrapper carries fixed structure worth showing, as a truncation bound or a censoring window does. Each entry becomes one `:attribute` row of `composed_to_table`.
3. Override `leaf_mean` and `leaf_var` only when the wrapper's transform changes the moment, as an affine scale and shift does (or as `Truncated`/`Distributions.Censored` do — see the built-in overrides in `composed_moments.jl` for the fallback rule when the underlying family has no closed form).
4. Implement `extra_leaf_params` and `set_extra_leaf_params` only when the wrapper owns a free parameter beyond the inner delay's native ones.
5. Add a `leaf_detail_lines` method when the wrapper's raw struct dump would clutter an `inspect` tree.
6. Run [`TestUtils.test_leaf_protocol_completeness`](@ref) over a wrapper constructor built from steps 1-4: it names the specific step you skipped.

Every method here dispatches on an instance, and the flat-vector codec (`flat_dimension`, `unflatten`, `flatten`, `reconstruct`) reads the same instance-level hooks at call time, so there is no separate type-level table to keep in step and no registration step.

## [Writing a new composer node](@id new-composer-node)

A realisation of a composed tree is one flat vector of leaf values laid out depth-first, and each node reads and writes only its own contiguous slice by an offset.
Three methods carry this, reached by the qualified name (`ComposedDistributions.child_nleaves` and friends, which are `public` but not exported):

- `child_nleaves(node)` returns a positive `Int`, the flat-slot count (one per leaf below the node);
- `child_logpdf(node, x, offset, n)` returns a finite scalar over the node's `n`-wide slice `x[offset + 1 : offset + n]`, independent of the surrounding padding;
- `child_rand!(out, offset, rng, node)` fills exactly that slice in place and returns `nothing`, leaving the padding either side untouched.

A node delegates to each child by the same three methods, passing each child its own offset, so it nests inside any other node automatically: the built-ins call these functions polymorphically over each child rather than switching on the child's concrete type, so a node implementing them nests as a named child of `Sequential`/`Parallel`/`Choose`/`Resolve`/`Compete` with no registration, once it subtypes `AbstractComposedDistribution` (composability is checked structurally against that root, not a closed list of the built-in types).

A univariate leaf is the base case: it occupies one slot (`child_nleaves == 1`), `child_rand!` writes its single draw, and `child_logpdf` scores `x[offset + 1]`.
Any `Distributions.jl` distribution is therefore a valid leaf with no package-specific hooks, as the leaf section above says.

Add `node_children(node)` (returning the node's immediate children as a `Tuple`) and the node also gets [`has_varying`](@ref) and [`has_uncertain`](@ref) for free — both walk `any(f, node_children(node))` generically, dispatching against the public `AbstractComposedDistribution` root.
Add [`node_attributes`](@ref) when the node carries fixed structure that is not a free parameter, as a `Choose`'s selector is; its `node` label and its children's rows need no method of their own.

```@example extending
using ComposedDistributions, Distributions, Random
import ComposedDistributions: child_nleaves, child_logpdf, child_rand!,
                              node_children, AbstractComposedDistribution

# A minimal node combining two branches side by side. Subtyping
# `AbstractComposedDistribution{Multivariate, Continuous}` (rather than the
# `AbstractMultiChild` intermediate, which assumes a `.components` field) is
# the smallest correct root for a bespoke multivariate node.
struct Both{A, B} <: AbstractComposedDistribution{Multivariate, Continuous}
    first::A
    second::B
end

child_nleaves(b::Both) = child_nleaves(b.first) + child_nleaves(b.second)

function child_logpdf(b::Both, x, offset, ::Int)
    n1 = child_nleaves(b.first)
    n2 = child_nleaves(b.second)
    return child_logpdf(b.first, x, offset, n1) +
           child_logpdf(b.second, x, offset + n1, n2)
end

function child_rand!(out, offset, rng::AbstractRNG, b::Both)
    n1 = child_nleaves(b.first)
    child_rand!(out, offset, rng, b.first)
    child_rand!(out, offset + n1, rng, b.second)
    return nothing
end

node_children(b::Both) = (b.first, b.second)

node = Both(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
out = zeros(child_nleaves(node))
child_rand!(out, 0, Random.default_rng(), node)
lp = child_logpdf(node, out, 0, child_nleaves(node))

# `Both` nests as a named child of a built-in — the point of subtyping
# `AbstractComposedDistribution` rather than keeping it a bare struct.
tree = sequential(:pair => node, :tail => Gamma(1.0, 1.0))
(node_output = out, node_logpdf = lp, tree_length = length(tree),
    tree_flat_logpdf = logpdf(tree, [1.0, 2.0, 0.5]),
    has_varying = has_varying(tree))
```

### What this does not (yet) cover

The three `child_*` methods plus `node_children` are enough for flat-vector scoring and sampling (`child_logpdf`/`child_rand!` directly, and `logpdf(tree, flat_vector)` on any tree nesting the node) and for the `has_varying`/`has_uncertain` predicates.
They are not yet enough for a genuinely new multi-leaf node type to participate in the *named* surface: `rand(tree)` returning a labelled `NamedTuple`, `logpdf(tree, nt::NamedTuple)`, `composed_to_table`, `event_names`/`event_tree`, or the [`flat_dimension`](@ref)/[`flatten`](@ref)/[`unflatten`](@ref)/[`reconstruct`](@ref) codec used by [`uncertain`](@ref) parameters.
Those walks are still closed over the built-in node types, tracked separately (the codec walk is [issue #332](https://github.com/EpiAware/ComposedDistributions.jl/issues/332); the named-output walk is undocumented today).
A node with exactly one flat slot (occupying a single named position, like the one_of family) is unaffected by this gap; a node with more than one is not yet a drop-in replacement for `Sequential`/`Parallel` beyond flat scoring.

## The one_of-outcome family: `AbstractOneOf`

The two one_of-outcome nodes share the supertype `AbstractOneOf`: [`Resolve`](@ref) (the fixed-probability mixture, cause and timing independent) and [`Compete`](@ref) (racing hazards, with the winning probability derived from the hazards).
`AbstractOneOf` subtypes `AbstractComposedDistribution{Univariate, Continuous}`, so the one_of family is the univariate arm of the composer hierarchy.
Both are univariate marginals, so each occupies a single flat slot and satisfies the node contract through the univariate-leaf base case, and is not subject to the "what this does not yet cover" gap above.

A valid member subtypes `AbstractOneOf`, stores its outcome `names`, and implements the standard univariate interface (`logpdf`, `rand`, and the moments it can compute) so the marginal is a proper distribution.

```@example extending
using ComposedDistributions, Distributions

r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
(r isa ComposedDistributions.AbstractOneOf, c isa ComposedDistributions.AbstractOneOf)
```

## The introspection contract

A composed tree exposes its structure through name introspection, and every built-in node keeps these in agreement:

- `component_names(node)` — the `Tuple` of immediate child names;
- [`event_names`](@ref) — the flat per-event name tuple (one entry per leaf edge, plus the origin);
- [`event_tree`](@ref) — the same names as a nested record;
- [`event`](@ref) — fetch a child or descend a name path;
- [`composed_to_table`](@ref) — the full node/attribute/parameter inventory (one row per composer node, leaf wrapper layer, fixed-structure attribute and free parameter); a composed distribution is itself a Tables.jl source over this table. Filter its `role` column to `:param` for the free-parameter-only rows;
- [`node_attributes`](@ref) — a node or leaf layer's own fixed, non-parameter structure, one `:attribute` row each. This is the only method a downstream type defines to control its rows in that table: the `node` label is read off the type name, and a wrapped leaf's layers are peeled through `inner_dist`.

```@example extending
using ComposedDistributions, Distributions
import ComposedDistributions: component_names

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
(names = component_names(tree), onset = event(tree, :onset_admit))
```

## Conformance suite reference

The reusable suite lives in the `ComposedDistributions.TestUtils` submodule.
`test_interface` runs the public checklist over the fixture set; `test_node_interface` runs the node-extension checklist; `test_composed_interface` wraps both and asserts the `AbstractComposedDistribution` membership; `test_abstract_membership` asserts the whole hierarchy; `test_leaf_protocol_completeness` closes [#277](https://github.com/EpiAware/ComposedDistributions.jl/issues/277) (a wrapper that silently drops what it wraps); and `test_sampling_consistency` closes [#278](https://github.com/EpiAware/ComposedDistributions.jl/issues/278) (`rand` and `logpdf`/`cdf` silently describing different distributions).
Drop the same suite into your own tests to verify a custom leaf, wrapper, or composer conforms, and run it after adding a type to a family.

```@docs
ComposedDistributions.TestUtils
ComposedDistributions.TestUtils.test_interface
ComposedDistributions.TestUtils.test_composed_interface
ComposedDistributions.TestUtils.test_node_interface
ComposedDistributions.TestUtils.test_abstract_membership
ComposedDistributions.TestUtils.test_rejects_invalid
ComposedDistributions.TestUtils.test_leaf_protocol_completeness
ComposedDistributions.TestUtils.test_sampling_consistency
ComposedDistributions.TestUtils.test_ad_safety
ComposedDistributions.TestUtils.test_registry_coverage
ComposedDistributions.TestUtils.registry_types
ComposedDistributions.TestUtils.example_fixtures
```
