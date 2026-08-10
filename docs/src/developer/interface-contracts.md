# [Interface contracts: valid nodes and leaves](@id interface-contracts)

This page is the reference for what makes a type a _valid_ participant in composition, stated as the exact method contract the package relies on.
A composer combines named child distributions into an event tree; the tree walkers reach every node and leaf through a small set of methods, so any type that implements those methods composes with the built-ins with no extra work.

The reusable interface-conformance suite `ComposedDistributions.TestUtils` checks these contracts over every built-in node shape and a user-defined node, and the package runs it in `test/interfaces.jl`, so the prose here and the tests stay in sync.
To add a valid member, subtype the right abstract, implement the methods listed for its role, and run `test_node_interface` (the node contract alone — composing, tabling, flattening) or `test_composed_interface` (the fuller public checklist too, for a node also exercising `rand`/`logpdf`/moments as a standalone root) over an instance.

## The type landscape

The composer nodes share one supertype, `AbstractComposedDistribution{F, S}`.
The named-child composers and the univariate one_of family sit under it; leaves and leaf wrappers are plain univariate distributions under no composer supertype.

```text
Distribution{F, S}
└── AbstractComposedDistribution{F, S}   named children → an event tree
    ├── AbstractMultiChild{S}            positional, tree-walked together
    │   ├── Sequential                   named steps in series (a chain)
    │   └── Parallel                     named branches off one origin (fan-out)
    ├── Choose                           data-selected disjoint alternatives
    └── AbstractOneOf                    one univariate time-to-event marginal
        ├── Resolve                      a fixed-probability mixture
        └── Compete                      racing hazards (soonest cause fires)

plain univariate leaves (no composer supertype):
    Shared    a tied leaf (one free parameter across branches)
    NoEvent   an absorbing no-event branch
    any Distributions.jl UnivariateDistribution
```

`AbstractComposedDistribution` is parametric on variate form `F` (`Univariate` / `Multivariate`), so one supertype spans the univariate one_of members and the multivariate event-tree composers while preserving `Distribution{F, S}`.

- `AbstractMultiChild` is an intermediate that groups the two positional multi-child composers (`Sequential`, `Parallel`) the tree walkers dispatch over together.
- `Choose` is a sibling of `AbstractMultiChild`, not a multi-child node itself.
- `AbstractOneOf` re-roots the univariate one_of family under the composed supertype, so it stays a `UnivariateDistribution` while sharing the composed abstract; the tree walkers dispatch on `AbstractOneOf` wherever the two one_of nodes behave alike (one event slot per outcome, the shared origin, the per-outcome draw) and on the concrete type only where the scoring arithmetic differs.
- Downstream extension packages (CensoredDistributions and its siblings) dispatch on these supertypes, so the names and shape match the shared contract; `test_abstract_membership` pins the membership down as a test, so a type filed under the wrong supertype fails.

## The composer-node contract

A node combines named children into a bigger structure: a table row, a slice of the flat estimated-parameter vector, a slice of the flat event vector.
Three methods carry the whole contract, reached by the qualified name (`public`, not exported):

- [`ComposedDistributions.node_children`](@ref)`(node)` returns the node's children as a `Tuple`, positionally matching `component_names`;
- [`ComposedDistributions.node_rebuild`](@ref)`(node, children)` rebuilds a node of the same type and own fixed structure around a new children tuple;
- `component_names(node)` returns a `Tuple` of the child names.

Those three are all a plain node needs.
`composed_to_table`, nested `params`, and `update` all walk `node_children`/`node_rebuild`/`component_names` generically, with no further method from the node.
So does the flat *event*-vector walk (`child_nleaves`/`child_logpdf`/`child_rand!`, below) for a "concatenating" node — one whose realisation is just its children's laid end to end, exactly `Sequential`/`Parallel`'s own semantics.
A node with different combination semantics (a disjunction like `Choose`, a mixture) overrides the three `child_*` methods directly:

- `child_nleaves(node)` returns a positive `Int`, the flat-slot count (one per leaf below the node);
- `child_logpdf(node, x, offset, n)` returns a finite scalar over the node's `n`-wide slice `x[offset + 1 : offset + n]`, independent of the surrounding padding;
- `child_rand!(out, offset, rng, node)` fills exactly that slice in place and returns `nothing`, leaving the padding either side untouched.

A univariate leaf is the base case for the `child_*` walk: it occupies one slot (`child_nleaves == 1`), `child_rand!` writes its single draw, and `child_logpdf` scores `x[offset + 1]`.
Any `Distributions.jl` distribution is therefore a valid leaf with no package-specific hooks — steer 1 of the package's design, unaffected by anything below.

### Codec (flatten/unflatten/fit) support

A tree's estimated-parameter codec (`flat_dimension`/`flatten`/`unflatten`/`reconstruct`, the primitive a sampler-driven fit routes through) is generated once per distinct tree *type*, from a compile-time walk over that type alone — no `Dict`, no per-call tree walk, so the reverse-mode AD backends differentiate through it (see the module banner on `src/composers/codec_gen.jl`).
That compile-time walk cannot call `node_children`/`component_names` (or any other method a downstream package defines): a `@generated` function calling a method a *different*, downstream package supplies is a Julia world-age hazard, verified by a two-package harness that reproduces a `MethodError: ... world age` even after both packages are fully loaded — `node_children` itself, an ordinary *runtime* dispatch called from the code the generator emits (not from the generator itself), carries no such risk, which is exactly why it is fine to require but a compile-time equivalent is not.

So a node wanting codec support declares its own child names (matching `component_names`) and its children's types (matching `node_children`'s return type) as its **first two type parameters**, in that order — exactly the shape `Sequential`/`Parallel`/`Choose`/`Compete` already use.
A node subtyping `AbstractComposedDistribution` without that shape gets a clear, actionable `ArgumentError` naming the type the first time the codec is asked to walk it, never a silent miscount.

### [Adding a valid composer node](@id new-composer-node)

1. Subtype `AbstractComposedDistribution{F, S}`, with your own child names and children's types as the first two type parameters if you want codec support (see above).
2. Implement `node_children`, `node_rebuild` and `component_names`.
3. Override the three `child_*` methods only if the node's combination semantics are not a plain concatenation.
4. Implement `node_attributes` if the node carries fixed structure that is not a free parameter, as a `Choose`'s selector is. Its `node` label and its children's rows need no method of their own.
5. Verify against the suite (`test_node_interface` covers the node contract; run it, don't just read the list above).

```@example new-composer-node
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

That is the whole contract — `child_nleaves`/`child_logpdf`/`child_rand!` are not overridden, so they fall back to the generic "concatenating node" default, and the harness passes:

```@example new-composer-node
node = Both((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:first, :second))
test_node_interface(node)
nothing # hide
```

It genuinely composes, tables, flattens and fits, with no further method — nested as a nameless child of a built-in composer, or built directly with an `uncertain` leaf of its own:

```@example new-composer-node
both = Both((uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        Gamma(1.5, 1.0)), (:leg, :tail))
tbl = composed_to_table(both)  # one estimated row, for the uncertain leg
(node = tbl.node[1], estimated_rows = count(!isnothing, tbl.prior))
```

```@example new-composer-node
# `Both` nests like any built-in composer node.
tree = Sequential((both, Gamma(1.0, 1.0)), (:branch, :tail2))
ComposedDistributions.flat_dimension(tree)  # 1: only the leg is uncertain
```

```@example new-composer-node
# reconstruct/update is the fit primitive: collapse at an estimated draw.
fitted = ComposedDistributions.reconstruct(tree, [4.0])
node_children(node_children(fitted)[1])[1]  # the leg, collapsed at x[1] = 4.0
```

## The one_of-outcome family: `AbstractOneOf`

The two one_of-outcome nodes share the supertype `AbstractOneOf`: [`Resolve`](@ref) (the fixed-probability mixture, cause and timing independent) and [`Compete`](@ref) (racing hazards, with the winning probability derived from the hazards).
`AbstractOneOf` subtypes `AbstractComposedDistribution{Univariate, Continuous}`, so the one_of family is the univariate arm of the composer hierarchy.
Both are univariate marginals, so each occupies a single flat slot and satisfies the node contract through the univariate-leaf base case.

A valid member subtypes `AbstractOneOf`, stores its outcome `names`, and implements the standard univariate interface (`logpdf`, `rand`, and the moments it can compute) so the marginal is a proper distribution.

```julia
using ComposedDistributions, Distributions

r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
r isa ComposedDistributions.AbstractOneOf
c isa ComposedDistributions.AbstractOneOf
```

## The introspection contract

A composed tree exposes its structure through name introspection. `component_names`, `composed_to_table` and `node_attributes` are generic over any node satisfying the node contract above; `event_names`/`event_tree`/`event` (and top-level `rand`/`logpdf`/moments when the node is a standalone root) are, for now, exercised only over the five built-in node kinds — a downstream node's own event naming and root-level sampling are open, tracked in a follow-up issue, not part of this contract.

- `component_names(node)` — the `Tuple` of immediate child names;
- [`composed_to_table`](@ref) — the full node/attribute/parameter inventory (one row per composer node, leaf wrapper layer, fixed-structure attribute and free parameter); a composed distribution is itself a Tables.jl source over this table. Filter its `role` column to `:param` for the free-parameter-only rows;
- `node_attributes` — a node or leaf layer's own fixed, non-parameter structure, one `:attribute` row each. This is the only method a downstream type defines to control its rows in that table: the `node` label is read off the type name, and a wrapped leaf's layers are peeled through `inner_dist`;
- over the built-ins: [`event_names`](@ref) (the flat per-event name tuple, one entry per leaf edge plus the origin), [`event_tree`](@ref) (the same names nested) and [`event`](@ref) (fetch a child or descend a name path).

```julia
using ComposedDistributions, Distributions
import ComposedDistributions: component_names

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
component_names(tree)          # (:onset_admit, :admit_death)
event(tree, :onset_admit)      # Gamma(2.0, 1.0)
composed_to_table(tree)        # the full node/attribute/param table
```

## The leaf-wrapper contract

A leaf wrapper wraps one inner base distribution and stays transparent to the prior and parameter surface, through the `free_leaf` / `rewrap_leaf` pair (`public`, not exported).
See [The leaf protocol](@ref leaf-protocol) for the full contract, the round-trip guarantee, and a worked example.

## Keeping the hierarchy honest

The reusable `ComposedDistributions.TestUtils` suite is the machine-checkable statement of these contracts, and the package runs it in `test/interfaces.jl`.
`test_interface` runs the public checklist over the fixture set (`example_fixtures`); `test_node_interface` runs the node-extension checklist, including the `flat_dimension` invariant (`test_estimation_dimension`: the codec's estimated-parameter count must match `composed_to_table`'s estimated-row count, so a node that silently drops an uncertain leaf from the codec fails the harness instead of shipping green); `test_composed_interface` wraps both and asserts the `AbstractComposedDistribution` membership; and `test_abstract_membership` asserts the whole hierarchy (every composer under `AbstractComposedDistribution`, `Sequential` / `Parallel` under `AbstractMultiChild`, the one_of family under `AbstractOneOf`, `Choose` a sibling, and plain leaves and `Shared` standalone).
Drop the same suite into your own tests to verify a custom leaf or composer conforms, and run it after adding a type to a family:

```julia
using ComposedDistributions.TestUtils: test_composed_interface, test_abstract_membership

test_abstract_membership()
```

## Conformance suite reference

The reusable suite lives in the `ComposedDistributions.TestUtils` submodule.

```@docs
ComposedDistributions.TestUtils
ComposedDistributions.TestUtils.test_interface
ComposedDistributions.TestUtils.test_composed_interface
ComposedDistributions.TestUtils.test_node_interface
ComposedDistributions.TestUtils.test_estimation_dimension
ComposedDistributions.TestUtils.test_abstract_membership
ComposedDistributions.TestUtils.test_rejects_invalid
ComposedDistributions.TestUtils.test_ad_safety
ComposedDistributions.TestUtils.test_registry_coverage
ComposedDistributions.TestUtils.registry_types
ComposedDistributions.TestUtils.example_fixtures
```
