# [Extending ComposedDistributions](@id extending)

This page is the reference for what makes a type a _valid_ participant in composition: a leaf, a leaf wrapper, or a composer node.
It states the exact method contract the package relies on for each role, what each role costs to implement, and what today's contract does and does not yet cover.

The reusable interface-conformance suite `ComposedDistributions.TestUtils` checks these contracts over every built-in shape and the worked examples on this page, and the package runs it in `test/interfaces.jl`, so the prose here and the tests stay in sync.

## The type landscape

The composer nodes share one supertype, `AbstractComposedDistribution{F, S}`.
The named-child composers and the univariate one_of family sit under it; a leaf or leaf wrapper is any type implementing the `Distributions.jl` univariate interface (see "Adding a leaf: nothing" below), under no composer supertype of its own.

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
| any type implementing the `Distributions.jl` univariate interface | a leaf, or the base of a leaf wrapper |

`AbstractComposedDistribution` is parametric on variate form `F` (`Univariate` / `Multivariate`), so one supertype spans the univariate one_of members and the multivariate event-tree composers while preserving `Distribution{F, S}`.
Downstream extension packages (CensoredDistributions and its siblings) dispatch on these supertypes, so the names and shape match the shared contract; `TestUtils.test_abstract_membership` pins the membership down as a test, so a type filed under the wrong supertype fails.

## Adding a leaf: nothing

Any type conforming to the `Distributions.jl` univariate interface is a valid leaf with no `ComposedDistributions` method of its own.
Conformance here means implementing `Distributions.params`, `logpdf`, `rand`, and `minimum`/`maximum`: methods `Distributions.jl` itself requires of a univariate distribution, not a `ComposedDistributions`-specific hook.
`composed_to_table`'s `support` column reads `minimum`/`maximum` directly, so a leaf missing either composes and scores but cannot table.
It composes, ties, gets wrapped in `uncertain`, and scores through the flat codec, unchanged.
Its estimable parameter names come from its own type, so a named parameter table row needs no method either; see [Parameter names](@ref parameter-names) below for the one case that does.

Subtyping `UnivariateDistribution` remains the recommended shape for a leaf: `Distributions.jl`'s own `truncated`/`censored`, and anything else dispatching on that supertype, require it.
A leaf that only implements the interface above, without subtyping, still composes, tables, scores, flattens and fits — it is the escape hatch for a type that cannot subtype — but it cannot be wrapped by `Distributions.jl`'s own `truncated`/`censored`.

```@example extending
using ComposedDistributions, Distributions

leaf = Gamma(2.0, 1.0)
tree = compose((onset = leaf, admit = LogNormal(0.5, 0.4)))
composed_to_table(tree)
```

A user-defined type makes the same claim concrete.
`Zilch` below is not a `Distributions.jl` built-in, and implements exactly those interface methods, no more.

```@example extending
using Random
using ComposedDistributions.TestUtils: test_node_interface

struct Zilch <: ContinuousUnivariateDistribution
    m::Float64
    s::Float64
end
Distributions.params(d::Zilch) = (d.m, d.s)
Distributions.logpdf(d::Zilch, x::Real) = logpdf(Normal(d.m, d.s), x)
Base.rand(rng::AbstractRNG, d::Zilch) = rand(rng, Normal(d.m, d.s))
Base.minimum(d::Zilch) = -Inf
Base.maximum(d::Zilch) = Inf

ComposedDistributions.param_names(Zilch(1.0, 2.0))
```

`param_names` derives `(:m, :s)` from `Zilch`'s own fields, with no method of its own (see [Parameter names](@ref parameter-names) below).
It composes, tables, scores and samples, and takes an attached prior through `uncertain`, all with no `ComposedDistributions` method:

```@example extending
zilch_tree = compose((onset = Zilch(1.0, 2.0), admit = LogNormal(0.5, 0.4)))
composed_to_table(zilch_tree)
```

```@example extending
uncertain_zilch = compose((onset = uncertain(
    Zilch(1.0, 2.0); m = Normal(0.0, 1.0)),))
ComposedDistributions.flat_dimension(uncertain_zilch)
```

```@example extending
test_node_interface(Zilch(1.0, 2.0); name = "Zilch")
nothing # hide
```

## Adding a leaf wrapper

A leaf wrapper carries fixed structure (censoring bounds, a shared tie, an attached prior) around an inner base distribution.
A wrapper costs exactly two methods.
`inner_dist` peels one layer to the inner distribution and `rewrap_leaf` rebuilds the wrapper around a new one; everything below them is optional, with a default that is correct for a wrapper carrying *nothing extra* through.
Defining `inner_dist` without `rewrap_leaf` is refused with an error naming the missing method, because rebuilding such a wrapper would otherwise drop the structure it carries and leave every later `logpdf` and `rand` wrong with no warning.
A wrapper that does carry something through and does not override the matching hook loses it silently, which is [issue #277](https://github.com/EpiAware/ComposedDistributions.jl/issues/277)'s whole subject and the reason [`TestUtils.test_leaf_protocol_completeness`](@ref) below exists.

The two required methods:

| Method | Role |
|---|---|
| [`inner_dist`](@ref) | peel one wrapper layer to the inner distribution |
| [`rewrap_leaf`](@ref) | rebuild the wrapper around a new inner delay |

Everything else is optional, and each default is correct for a wrapper that
carries nothing of that kind. Override one only when the wrapper does.

| Optional method | Role | Default when unimplemented |
|---|---|---|
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

## [Parameter names](@id parameter-names)

`param_names(leaf)` derives its answer from the leaf's own type, with no method required for the common case.
The names are the first `N` fieldnames of `typeof(free_leaf(leaf))`, transliterated (Greek letters to their English spelling), where `N` is the arity of `params(leaf)`.
This works whenever a family's fields line up 1:1 with its `params`, in order, which is true of every Distributions.jl-conforming family whose author declared its fields in `params` order — the common case, and the reason a plain leaf costs nothing for names too.
A leaf whose fields fall short of `N`, or whose declared field type at some slot disagrees with the matching `params` slot, or whose transliterated names collide, falls back to positional `:param_1, :param_2, ...` rather than guessing.

A leaf type whose fields do not line up with its `params`, in order, overrides `param_names` explicitly, in step with `leaf_ctor`.
The motivating case is a moment-parameterised wrapper naming a mean and a standard deviation rather than a shape and a scale.
`uncertain(...)` checks the alignment at construction time for any leaf whose derivation used the fieldname branch, restricted to `Real`-valued parameter slots, and throws naming [`param_names`](@ref) and [`leaf_ctor`](@ref) as the fix when a field's value does not match its corresponding `params` slot.
`TestUtils.test_interface`/`test_node_interface` run the same check over every real leaf of a tree, so a conformance run catches the mismatch too.

A leaf's `params` must be type-stable: the derivation reads `typeof(params(leaf))`, so a type-unstable `params` widens every codec return type built from that leaf.

```@example extending
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

## [Writing a new composer node](@id new-composer-node)

A node combines named children into a bigger structure: a table row, a slice of the flat estimated-parameter vector, a slice of the flat event vector.
Three methods carry the whole contract, reached by the qualified name (`public`, not exported):

- [`node_children`](@ref)`(node)` returns the node's children as a `Tuple`, positionally matching `component_names`;
- [`node_rebuild`](@ref)`(node, children)` rebuilds a node of the same type and own fixed structure around a new children tuple;
- `component_names(node)` returns a `Tuple` of the child names.

Those three are all a plain node needs.
`composed_to_table`, nested `params`, `update`, [`has_varying`](@ref) and [`has_uncertain`](@ref) all walk them generically, dispatching against the public `AbstractComposedDistribution` root rather than a closed list of the built-in types, so a node needs no registration.

A realisation of a composed tree is also one flat vector of leaf values laid out depth-first, and each node reads and writes only its own contiguous slice by an offset.
Three further methods carry that walk:

- `child_nleaves(node)` returns a positive `Int`, the flat-slot count (one per leaf below the node);
- `child_logpdf(node, x, offset, n)` returns a finite scalar over the node's `n`-wide slice `x[offset + 1 : offset + n]`, independent of the surrounding padding;
- `child_rand!(out, offset, rng, node)` fills exactly that slice in place and returns `nothing`, leaving the padding either side untouched.

They default generically off `node_children` for a *concatenating* node, one whose realisation is just its children's laid end to end, exactly `Sequential`/`Parallel`'s own semantics.
A node with different combination semantics (a disjunction like `Choose`, a mixture) overrides the three directly.

A univariate leaf is the base case for that walk: it occupies one slot (`child_nleaves == 1`), `child_rand!` writes its single draw, and `child_logpdf` scores `x[offset + 1]`.
Any type implementing the `Distributions.jl` univariate interface is therefore a valid leaf with no package-specific hooks, as the leaf section above says.
Its estimable parameter names are derived from its own fields (see [Parameter names](@ref parameter-names)); a leaf whose fields do not line up with its `params`, in order, is the one case that needs a method of its own.

Add [`node_attributes`](@ref) when the node carries fixed structure that is not a free parameter, as a `Choose`'s selector is; its `node` label and its children's rows need no method of their own.

### Codec (flatten/unflatten/fit) support

A tree's estimated-parameter codec ([`flat_dimension`](@ref)/[`flatten`](@ref)/[`unflatten`](@ref)/[`reconstruct`](@ref), the primitive a sampler-driven fit routes through) is generated once per distinct tree *type*, from a compile-time walk over that type alone — no `Dict`, no per-call tree walk, so the reverse-mode AD backends differentiate through it (see the module banner on `src/composers/codec_gen.jl`).
That compile-time walk cannot call `node_children`/`component_names`, or any other method a downstream package defines: a `@generated` function calling a method a *different*, downstream package supplies is a Julia world-age hazard, verified by a two-package harness that reproduces a `MethodError: ... world age` even after both packages are fully loaded.
`node_children` itself, an ordinary *runtime* dispatch called from the code the generator emits rather than from the generator itself, carries no such risk, which is exactly why it is fine to require but a compile-time equivalent is not.

So a node wanting codec support declares its own child names (matching `component_names`) and its children's types (matching `node_children`'s return type) as its **first two type parameters**, in that order — exactly the shape `Sequential`/`Parallel`/`Choose`/`Compete` already use.
A node subtyping `AbstractComposedDistribution` without that shape gets a clear, actionable `ArgumentError` naming the type the first time the codec is asked to walk it, never a silent miscount.

### Steps

1. Subtype `AbstractComposedDistribution{F, S}`, with your own child names and children's types as the first two type parameters if you want codec support (see above).
2. Implement `node_children`, `node_rebuild` and `component_names`.
3. Override the three `child_*` methods only if the node's combination semantics are not a plain concatenation.
4. Implement `node_attributes` if the node carries fixed structure that is not a free parameter.
5. Verify against the suite ([`TestUtils.test_node_interface`](@ref) covers the node contract; run it, don't just read the list above).

```@example extending
using ComposedDistributions, Distributions
using ComposedDistributions.TestUtils: test_node_interface
import ComposedDistributions: node_children, node_rebuild, component_names,
                              AbstractComposedDistribution

# A minimal node combining two named branches side by side. `names` and the
# children's types as the first two type parameters (mirroring
# `Sequential`/`Parallel`) is what lets the codec below read the tree's
# layout with no method call of its own.
struct Both{names, C <: Tuple} <:
       AbstractComposedDistribution{Multivariate, Continuous}
    children::C
    function Both{names}(children::C) where {names, C <: Tuple}
        length(names) == length(children) ||
            throw(ArgumentError("names/children length mismatch"))
        new{names, C}(children)
    end
end
Both(children::C, names::NTuple{N, Symbol}) where {N, C <: Tuple} =
    Both{names}(children)

node_children(d::Both) = d.children
node_rebuild(d::Both, children::Tuple) = Both(children, component_names(d))
component_names(::Both{names}) where {names} = names
nothing # hide
```

That is the whole contract — `child_nleaves`/`child_logpdf`/`child_rand!` are not overridden, so they fall back to the generic concatenating-node default, and the harness passes:

```@example extending
node = Both((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:first, :second))
test_node_interface(node)
nothing # hide
```

It genuinely composes, tables, flattens and fits, with no further method — nested as a child of a built-in composer, or built directly with an `uncertain` leaf of its own:

```@example extending
both = Both((uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)),
        Gamma(1.5, 1.0)), (:leg, :tail))
tbl = composed_to_table(both)  # one estimated row, for the uncertain leg
(node = tbl.node[1], estimated_rows = count(!isnothing, tbl.prior))
```

```@example extending
# `Both` nests like any built-in composer node.
tree = Sequential((both, Gamma(1.0, 1.0)), (:branch, :tail2))
ComposedDistributions.flat_dimension(tree)  # 1: only the leg is uncertain
```

```@example extending
# reconstruct/update is the fit primitive: collapse at an estimated draw.
fitted = ComposedDistributions.reconstruct(tree, [4.0])
node_children(node_children(fitted)[1])[1]  # the leg, collapsed at x[1] = 4.0
```

### What this does not (yet) cover

`component_names`, `composed_to_table` and `node_attributes` are generic over any node satisfying the contract above, and so is the codec once the node carries the type-parameter layout.
The *named event* surface is not: `rand(tree)` returning a labelled `NamedTuple`, `logpdf(tree, nt::NamedTuple)`, and `event_names`/`event_tree`/`event` are exercised only over the five built-in node kinds, as are top-level `rand`/`logpdf`/moments when the node is a standalone root.
A downstream node's own event naming and root-level sampling are open, tracked in [issue #332](https://github.com/EpiAware/ComposedDistributions.jl/issues/332), not part of this contract.
A node with exactly one flat slot (occupying a single named position, like the one_of family) is unaffected by this gap.

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

A composed tree exposes its structure through name introspection.
`component_names`, `composed_to_table` and `node_attributes` are generic over any node satisfying the node contract above; `event_names`/`event_tree`/`event` are, for now, exercised only over the five built-in node kinds.

- `component_names(node)` — the `Tuple` of immediate child names;
- [`composed_to_table`](@ref) — the full node/attribute/parameter inventory (one row per composer node, leaf wrapper layer, fixed-structure attribute and free parameter); a composed distribution is itself a Tables.jl source over this table. Filter its `role` column to `:param` for the free-parameter-only rows;
- [`node_attributes`](@ref) — a node or leaf layer's own fixed, non-parameter structure, one `:attribute` row each. This is the only method a downstream type defines to control its rows in that table: the `node` column holds the type itself, and a wrapped leaf's layers are peeled through `inner_dist`;
- over the built-ins: [`event_names`](@ref) (the flat per-event name tuple, one entry per leaf edge plus the origin), [`event_tree`](@ref) (the same names nested) and [`event`](@ref) (fetch a child or descend a name path).

### Reading a table back

`compose(table)` is the inverse of `composed_to_table`: it rebuilds a tree from the rows, so `compose(composed_to_table(d)) == d`.
It works by dispatching on the table's `node` column, which holds the type itself rather than the type's name, so there is no name-to-type registry for a downstream package to register with.

Three hooks say how a type rebuilds itself, and two of them are defaulted so the extension cost does not move:

- [`from_table`](@ref)`(::Type{T}, values, attrs)` rebuilds a leaf. The default is `T(values...)`, right for any Distributions.jl family whose `params` are its constructor arguments, so **a leaf still costs zero methods**. A leaf whose free parameters are not its constructor arguments (one overriding `param_names`/`rebuild_leaf`) needs a method; the default detects that case and throws rather than building a wrong leaf silently.
- [`node_from_table`](@ref)`(::Type{T}, names, children, attrs)` rebuilds a composer node. The default is `T(children, names)` — the shape `Sequential`, `Parallel` and the `Both` example above already use — so **a node built that way still costs only the three methods of the node contract**. A node whose constructor takes something else needs a method; `Choose`, `Compete` and `Resolve` each have one.
- [`rewrap_from_table`](@ref)`(::Type{T}, inner, attrs)` re-applies a leaf-wrapper layer. This one has **no default**: a wrapper's fixed structure is its own, so a wrapper type needs a method here to round-trip. It receives back exactly what that layer's `node_attributes` reported. An `uncertain` layer needs none — its specs ride the `prior` column.

Anything without an applicable method throws, naming the type and the method to define, rather than losing structure quietly.

```@example extending
tbl = composed_to_table(both)
compose(tbl) == both
```

```@example extending
using ComposedDistributions, Distributions
import ComposedDistributions: component_names

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
(names = component_names(tree), onset = event(tree, :onset_admit))
```

## Conformance suite reference

The reusable suite lives in the `ComposedDistributions.TestUtils` submodule.
`test_interface` runs the public checklist over the fixture set; `test_node_interface` runs the node-extension checklist, including the `flat_dimension` invariant (`test_estimation_dimension`: the codec's estimated-parameter count must match `composed_to_table`'s estimated-row count, so a node that silently drops an uncertain leaf from the codec fails the harness instead of shipping green); `test_composed_interface` wraps both and asserts the `AbstractComposedDistribution` membership; `test_abstract_membership` asserts the whole hierarchy; `test_leaf_protocol_completeness` closes [#277](https://github.com/EpiAware/ComposedDistributions.jl/issues/277) (a wrapper that silently drops what it wraps); and `test_sampling_consistency` closes [#278](https://github.com/EpiAware/ComposedDistributions.jl/issues/278) (`rand` and `logpdf`/`cdf` silently describing different distributions).
Drop the same suite into your own tests to verify a custom leaf, wrapper, or composer conforms, and run it after adding a type to a family.

`test_node_interface` and `test_interface` check two different bars.
`test_node_interface` checks only what `ComposedDistributions` itself requires of a leaf or node: `child_nleaves`, `child_rand!`, `child_logpdf`, and the `flat_dimension` invariant above.
`test_interface` runs the fuller public checklist over the fixture set, and for a leaf called with `univariate = true` it additionally calls `mean`, `var`, `std` and `cdf` (in `TestUtils.jl`'s `_check_moments_and_rand` and `_check_cdf`): real `Distributions.jl` surface that `ComposedDistributions` never calls on a leaf itself.
Run `test_node_interface` to verify a custom leaf or node satisfies what this package needs.
Run `test_interface` too when the leaf is also meant to behave as a fully featured standalone `Distributions.jl` distribution.

```@docs
ComposedDistributions.TestUtils
ComposedDistributions.TestUtils.test_interface
ComposedDistributions.TestUtils.test_composed_interface
ComposedDistributions.TestUtils.test_node_interface
ComposedDistributions.TestUtils.test_estimation_dimension
ComposedDistributions.TestUtils.test_abstract_membership
ComposedDistributions.TestUtils.test_rejects_invalid
ComposedDistributions.TestUtils.test_leaf_protocol_completeness
ComposedDistributions.TestUtils.test_sampling_consistency
ComposedDistributions.TestUtils.test_ad_safety
ComposedDistributions.TestUtils.test_registry_coverage
ComposedDistributions.TestUtils.registry_types
ComposedDistributions.TestUtils.example_fixtures
```
