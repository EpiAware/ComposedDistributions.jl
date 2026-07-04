# [Interface contracts: valid nodes and leaves](@id interface-contracts)

This page is the reference for what makes a type a _valid_ participant in composition, stated as the exact method contract the package relies on.
A composer combines named child distributions into an event tree; the tree walkers reach every node and leaf through a small set of methods, so any type that implements those methods composes with the built-ins with no extra work.

The interface-conformance suite in `test/interfaces.jl` checks these contracts over every built-in node shape and a user-defined node, so the prose here and the tests stay in sync.
To add a valid member, implement the methods listed for its role and run the suite over an instance.

## The type landscape

Only the one_of-outcome family shares a supertype.
The named-child composers are sibling multivariate distributions; leaves and leaf wrappers are univariate.

```text
Distribution{Multivariate, Continuous}
├── Sequential   named steps in series (a chain)
├── Parallel     named branches off one shared origin (a fan-out)
└── Choose       data-selected disjoint alternatives (a row picks one)

UnivariateDistribution{Continuous}
├── AbstractOneOf   one univariate time-to-event marginal
│   ├── Resolve     a fixed-probability mixture over competing outcomes
│   └── Compete     racing hazards (the soonest cause fires)
├── Shared          a tied leaf (one free parameter across branches)
└── NoEvent         an absorbing no-event branch

plain leaves: any Distributions.jl UnivariateDistribution
```

`AbstractOneOf` is the only shared abstract: the tree walkers dispatch on it wherever the two one_of nodes behave alike (one event slot per outcome, the shared origin, the per-outcome draw) and on the concrete type only where the scoring arithmetic differs.
`Sequential`, `Parallel` and `Choose` are concrete siblings with no common composer abstract; membership is a structural fact the suite pins down rather than a type relation.

## The composer-node contract

A realisation of a composed tree is one flat vector of leaf values laid out depth-first, and each node reads and writes only its own contiguous slice by an offset.
Three methods carry this, reached by the qualified name (`ComposedDistributions.child_nleaves` and friends, which are `public` but not exported):

- `child_nleaves(node)` returns a positive `Int`, the flat-slot count (one per leaf below the node);
- `child_logpdf(node, x, offset, n)` returns a finite scalar over the node's `n`-wide slice `x[offset + 1 : offset + n]`, independent of the surrounding padding;
- `child_rand!(out, offset, rng, node)` fills exactly that slice in place and returns `nothing`, leaving the padding either side untouched.

A node delegates to each child by the same three methods, passing each child its own offset, so it nests inside any other node automatically.
`component_names(node)` returns a `Tuple` of the child names.

A univariate leaf is the base case: it occupies one slot (`child_nleaves == 1`), `child_rand!` writes its single draw, and `child_logpdf` scores `x[offset + 1]`.
Any `Distributions.jl` distribution is therefore a valid leaf with no package-specific hooks.

### Adding a valid composer node

1. Implement the three `child_*` methods so they read and write only the node's own slice, delegating to each child by the same methods.
2. Implement `component_names` if the node carries named children.
3. Verify against the suite by running the same conformance checks the built-ins pass.

```julia
using ComposedDistributions, Distributions, Random
import ComposedDistributions: child_nleaves, child_logpdf, child_rand!

# A minimal node combining two branches side by side.
struct Both{A, B}
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

node = Both(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
out = zeros(child_nleaves(node))
child_rand!(out, 0, Random.default_rng(), node)
child_logpdf(node, out, 0, child_nleaves(node))
```

## The one_of-outcome family: `AbstractOneOf`

The two one_of-outcome nodes share the supertype `AbstractOneOf`: [`Resolve`](@ref) (the fixed-probability mixture, cause and timing independent) and [`Compete`](@ref) (racing hazards, with the winning probability derived from the hazards).
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

A composed tree exposes its structure through name introspection, and every built-in node keeps these in agreement:

- `component_names(node)` — the `Tuple` of immediate child names;
- [`event_names`](@ref) — the flat per-event name tuple (one entry per leaf edge, plus the origin);
- [`event_tree`](@ref) — the same names as a nested record;
- [`event`](@ref) — fetch a child or descend a name path;
- [`params_table`](@ref) — the free parameters flattened to a Tables.jl table, one row per parameter.

```julia
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
component_names(tree)          # (:onset_admit, :admit_death)
event(tree, :onset_admit)      # Gamma(2.0, 1.0)
params_table(tree)             # a Tables.jl table of the free parameters
```

## The leaf-wrapper contract

A leaf wrapper wraps one inner base distribution and stays transparent to the prior and parameter surface.
The package reaches the inner leaf through two methods (`public`, not exported):

- `free_leaf(d)` returns the free inner leaf (a `Distribution`), peeling any wrapping;
- `rewrap_leaf(d, inner)` reconstructs an equivalent wrapper around a new inner leaf.

Together they must round-trip: `rewrap_leaf(d, free_leaf(d))` reproduces a node whose density matches `d`.
A plain leaf is its own free leaf (`free_leaf(d) == d`, `rewrap_leaf(d, inner) == inner`); `Truncated` peels to its untruncated base and rebuilds the bounds; [`Shared`](@ref) peels through to its inner leaf and rebuilds the tie.

```julia
using ComposedDistributions, Distributions
import ComposedDistributions: free_leaf, rewrap_leaf

d = shared(:inc, Gamma(2.0, 1.0))
free_leaf(d)                                 # Gamma(2.0, 1.0)
free_leaf(rewrap_leaf(d, Gamma(3.0, 1.5)))   # Gamma(3.0, 1.5)
```

## Keeping the hierarchy honest

The conformance suite in `test/interfaces.jl` is the machine-checkable statement of these contracts.
It runs the node interface over a plain leaf, `Sequential`, `Parallel`, `Resolve`, `Compete`, `Choose`, a nested mix and a user-defined node; asserts the abstract membership (the one_of family under `AbstractOneOf`, the named-child composers as sibling multivariate distributions, plain leaves and `Shared` standalone); checks the introspection names agree; and round-trips the leaf-wrapper contract.
Run it after adding a type to a family, and extend it with the new member.
