# [The leaf protocol](@id leaf-protocol)

This page is the reference for the leaf protocol, the stable public contract a downstream leaf-wrapper package implements so its wrapped delays are transparent to composition.
A leaf is a univariate distribution at the tip of a composed tree.
Any `Distributions.jl` distribution is a valid leaf with no extra work, so a plain leaf needs none of these methods.
A wrapper leaf, a type that carries fixed structure or extra parameters around an inner delay (censoring in CensoredDistributions, the modifiers in ModifiedDistributions), implements the methods below so the introspection and reconstruction layers see through the wrapper to the inner free delay.

The names are `public` but not exported, so a downstream package reaches them by the qualified name (`ComposedDistributions.free_leaf` and friends) and adds methods dispatching on its own wrapper type.
Extending only `inner_dist` and `rewrap_leaf` is enough for a fixed-structure wrapper; a wrapper that attaches priors or owns extra parameters extends the rest so those reach `composed_to_table` and `build_priors`.
Every method here dispatches on an instance, and the flat-vector codec (`flat_dimension`, `unflatten`, `flatten`, `reconstruct`) reads the same instance-level hooks at call time, so there is no separate type-level table to keep in step and no further registration step.

## The methods

The protocol splits into peel and rebuild, names, reconstruction, uncertainty, the shared tag, moments, and extra parameters.

| Method | Role |
|---|---|
| `inner_dist(leaf)` | peel one wrapper layer to the inner distribution |
| `free_leaf(leaf)` | peel to the innermost free delay |
| `rewrap_leaf(leaf, inner)` | rebuild the wrapper around a new inner delay |
| `node_attributes(leaf)` | the layer's own fixed, non-parameter structure |
| `component_names(node)` | a node's child names |
| `param_names(leaf)` | the inner delay's native parameter names |
| `leaf_param_names(leaf)` | the estimable names, native then extra |
| `leaf_ctor(leaf)` | the constructor that rebuilds the inner delay |
| `uncertain_specs(leaf)` | attached priors, or `nothing` when fixed |
| `shared_tag(leaf)` | the shared tie tag, or `nothing` |
| `leaf_mean(leaf)`, `leaf_var(leaf)` | per-leaf moments |
| `extra_leaf_params(leaf)` | modifier-owned free parameters and supports |
| `set_extra_leaf_params(leaf, vals)` | rebuild with new extra values |
| `leaf_detail_lines(leaf)` | `inspect` rendering lines |

The peel and rebuild pair is the base of the protocol.
`inner_dist` peels a single layer, and the read-through hooks (`free_leaf`, `shared_tag`, `uncertain_specs`, `extra_leaf_params`) recurse through it, so one method covers all of them.
`free_leaf` reaches the inner free delay whose parameters are the leaf's free parameters, and `rewrap_leaf` re-applies the fixed structure around a rebuilt delay.
A plain leaf is its own inner distribution and `rewrap_leaf` returns the new inner delay, so the identity holds without a method.
`composed_to_table` folds the same peel into the wrapper layers it lists, one `:node` row each, so a wrapper appears in the table with no further method.

Names and reconstruction fix the coordinates the parameter table and the codec work in.
`param_names` labels the native family parameters, `leaf_param_names` appends any extra names, and `leaf_ctor` rebuilds the inner delay from a positional tuple of native values.

## Parameter names

`param_names(leaf)` derives its answer from the leaf's own type, with no method required for the common case.
The names are the first `N` fieldnames of `typeof(free_leaf(leaf))`, transliterated (Greek letters to their English spelling), where `N` is the arity of `params(leaf)`.
This works whenever a family's fields line up 1:1 with its `params`, in order, which is true of every Distributions.jl-conforming family whose author declared its fields in `params` order — the common case, and the reason steer 1 holds for names too.
A leaf whose fields fall short of `N`, or whose declared field type at some slot disagrees with the matching `params` slot, or whose transliterated names collide, falls back to positional `:param_1, :param_2, ...` rather than guessing.

A leaf type whose fields do not line up with its `params`, in order, overrides `param_names` explicitly, in step with `leaf_ctor`.
The motivating case is a moment-parameterised wrapper naming a mean and a standard deviation rather than a shape and a scale — see the worked example below.
`uncertain(...)` checks the alignment at construction time for any leaf whose derivation used the fieldname branch, restricted to `Real`-valued parameter slots, and throws naming `ComposedDistributions.param_names` and `leaf_ctor` as the fix when a field's value does not match its corresponding `params` slot.
`TestUtils.test_interface`/`test_node_interface` run the same check over every real leaf of a tree, so a conformance run catches the mismatch too.

A leaf's `params` must be type-stable: the derivation reads `typeof(params(leaf))`, so a type-unstable `params` widens every codec return type built from that leaf.

```@example leaf-protocol
using ComposedDistributions, Distributions

struct MomentLeaf{D} <: ContinuousUnivariateDistribution
    vals::Tuple{Float64, Float64}
end
Distributions.params(d::MomentLeaf) = d.vals
ComposedDistributions.param_names(::MomentLeaf) = (:mean, :sd)
function ComposedDistributions.leaf_ctor(::MomentLeaf{D}) where {D}
    return (vals...) -> MomentLeaf{D}((vals[1], vals[2]))
end

ComposedDistributions.leaf_param_names(MomentLeaf{LogNormal}((8.0, 2.0)))
```

## Extra parameters

Most wrappers carry only fixed structure, so their extra-parameter map is empty and `composed_to_table`'s `:param` rows show just the inner delay's parameters.
A wrapper that owns a free parameter which is not one of the inner delay's native parameters reports it through `extra_leaf_params`, a `NamedTuple` mapping each extra name to a `(value, support)` pair.
The thinning factor of `thin(d, p)` is the first instance, reported as a `:thin` entry with support `(0.0, 1.0)`.
The support drives the default prior, so a `:thin` factor picks up a `Uniform(0, 1)` default the same way a `branch_probs` row does.
`set_extra_leaf_params` is the dual that rebuilds the leaf from new extra values by name.

A worked example with a plain leaf, where every peel is the identity and the extra map is empty.

```@example leaf-protocol
using ComposedDistributions, Distributions

leaf = Gamma(2.0, 1.0)
(free = ComposedDistributions.free_leaf(leaf),
    names = ComposedDistributions.leaf_param_names(leaf),
    specs = ComposedDistributions.uncertain_specs(leaf),
    tag = ComposedDistributions.shared_tag(leaf),
    mean = ComposedDistributions.leaf_mean(leaf),
    extras = ComposedDistributions.extra_leaf_params(leaf))
```

A censored or truncated leaf peels its fixed structure to the inner delay, so the parameter table lists only the inner free parameters.

```@example leaf-protocol
inner = ComposedDistributions.free_leaf(truncated(Gamma(2.0, 1.0); upper = 10.0))
rebuilt = ComposedDistributions.rewrap_leaf(
    truncated(Gamma(2.0, 1.0); upper = 10.0), Gamma(3.0, 1.5))
(inner = inner, rebuilt = rebuilt)
```

## Adding a wrapper leaf

1. Implement `inner_dist` and `rewrap_leaf` so the wrapper peels one layer to its inner distribution and rebuilds around a new inner delay. A tie or an attached prior under the wrapper then reaches the table with no further method, since the read-through hooks follow the same peel.
2. Add a `node_attributes` method when the wrapper carries fixed structure worth showing, as a truncation bound or a censoring window does. Each entry becomes one `:attribute` row of `composed_to_table`.
3. Override `leaf_mean` and `leaf_var` only when the wrapper's transform changes the moment, as an affine scale and shift does.
4. Implement `extra_leaf_params` and `set_extra_leaf_params` only when the wrapper owns a free parameter beyond the inner delay's native ones.
5. Add a `leaf_detail_lines` method when the wrapper's raw struct dump would clutter an `inspect` tree.

The docstrings for each method, with runnable examples, are in the public API reference.
Steps 1-5 are all a leaf-wrapper package needs: `composed_to_table`, `build_priors` and the generated flat-vector codec (`flat_dimension`/`unflatten`/`flatten`/`reconstruct`) all read the same instance-level hooks, so there is no separate codec-specific registration step.
