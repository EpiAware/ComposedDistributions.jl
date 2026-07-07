# [Concepts: composition verbs and primitives](@id concepts)

A composed distribution is a multi-state event process: named events linked by delays, wired into a tree.
This page maps the modelling concepts to the verbs that build them, so you can find the right primitive by intent before reading the worked examples in [Composing distributions](@ref composing-distributions).

## The layers

The package has four layers, each building on the one before.

- **Leaves** are any `Distributions.jl` `UnivariateDistribution`, used directly as the per-event delays.
  A leaf can also resolve later: a [`Varying`](@ref) leaf maps an observed covariate to a distribution, and an [`Uncertain`](@ref) leaf carries distribution-valued parameters.
- **Composers** wire named leaves into an event tree ([`compose`](@ref) and the five composers).
- **Combination and lowering** join or collapse whole delays ([`convolved`](@ref), [`difference`](@ref), [`observed_distribution`](@ref)).
- **Parameters and edits** read and reshape an assembled tree ([`params_table`](@ref), [`build_priors`](@ref), [`update`](@ref), [`prune`](@ref), [`splice`](@ref)).

## The verb map

The verbs fall into five families, listed here as verb, what it does, and what it returns.

**Structural composition** wires named branches into a tree.

| Verb | What it does | Returns |
|---|---|---|
| `compose` | lowers a NamedTuple, table or matrix to the stack | a composer |
| `sequential` | a conjunctive chain, where steps add up | [`Sequential`](@ref) |
| `parallel` | independent branches off one shared origin | [`Parallel`](@ref) |
| `resolve` | one outcome occurs by a fixed probability | [`Resolve`](@ref) |
| `compete` | racing hazards, the first to fire wins | [`Compete`](@ref) |
| `choose` | a data field picks the branch | [`Choose`](@ref) |
| `shared` | tags a leaf as a tied parameter group at build time | a tied leaf |
| `tie` | ties leaves at named paths into one parameter group | a tied tree |

**Combination and lowering** joins or collapses whole delays.

| Verb | What it does | Returns |
|---|---|---|
| `convolved` | the sum `X + Y` | [`Convolved`](@ref) |
| `difference` | the dual `X - Y` | [`Difference`](@ref) |
| `as_mixture` | the mixture view of a one_of node | a `MixtureModel` |
| `observed_distribution` | collapses a chain to its convolved total | a convolved leaf |

**Parameters** read and prior the free parameters.

| Verb | What it does | Returns |
|---|---|---|
| `params_table` | the flat free-parameter inventory | a Tables.jl table |
| `build_priors` | support-derived default priors from that table | a nested prior `NamedTuple` |
| `default_prior` | the default prior for one parameter row | a `Distribution` |

**Reading and editing** inspect or reshape an assembled tree.

| Verb | What it does | Returns |
|---|---|---|
| `event` / `event_names` / `event_tree` | fetch a child or the record key names | a node, leaf or names |
| `mean` / `var` | the composed marginal moments | a number or `NamedTuple` |
| `update` | replaces parameter values or whole nodes | a same-shape tree |
| `prune` / `splice` | drops or inserts a branch | an edited tree |

**Deferred leaves** hold a distribution that resolves later.
`varying` and `uncertain` are the two cases of one idea: a leaf that is a *map to
a distribution* rather than a fixed distribution, delegating silently to a
fallback until it is resolved.
They differ only in what indexes the map — `varying` an **observed** covariate
(time, stratum) resolved by `instantiate`, `uncertain` a **latent** parameter
draw (with a prior) resolved by `rand` or collapsed by `update` — and they share
one resolution walk, so a leaf can be both at once.

| Verb | What it does | Returns |
|---|---|---|
| `varying` / `Context` / `instantiate` | an observed covariate picks the leaf | a resolved tree |
| `has_varying` | whether any un-instantiated leaf remains | a `Bool` |
| `uncertain` | distribution-valued parameters that act as priors | an uncertain leaf |
| `has_uncertain` | whether any uncertain leaf remains | a `Bool` |

The deferred-leaf verbs are worked through in [Multi-strata trees and parameter uncertainty](@ref strata-uncertainty).

## Concept to primitive

| Modelling concept | Primitive | What it builds |
|---|---|---|
| Steps in series (a chain) | `sequential` / `compose` with a `Vector` value | [`Sequential`](@ref) |
| Branches off one shared origin | `parallel` / `compose` with a `NamedTuple` | [`Parallel`](@ref) |
| One outcome by fixed probability | `resolve` | [`Resolve`](@ref) |
| Racing outcomes (first to fire wins) | `compete` | [`Compete`](@ref) |
| A data field selects the sub-model | `choose` | [`Choose`](@ref) |
| Tie a leaf across branches | `shared` / `tie` | one shared parameter group |
| Sum of two independent delays | `convolved` | [`Convolved`](@ref) |
| Difference of two delays | `difference` | [`Difference`](@ref) |
| Mixture view of a one_of node | `as_mixture` | a `MixtureModel` |
| Collapse a chain to its total | `observed_distribution` | the convolved marginal |
| The free-parameter inventory | `params_table` | a Tables.jl table |
| Support-derived priors | `build_priors` | a nested prior `NamedTuple` |
| Read a child or descend a path | `event` | a node or leaf |
| Flat / nested event names | `event_names` / `event_tree` | the record key names |
| Replace values or whole nodes | `update` | a same-shape tree |
| Drop or insert a branch | `prune` / `splice` | an edited tree |
| A leaf that varies with a covariate | `varying`, resolved by `instantiate` | [`Varying`](@ref) |
| A leaf with parameter uncertainty | `uncertain`, collapsed by `update` | [`Uncertain`](@ref) |
| Guard a fitting loop against unresolved leaves | `has_varying` / `has_uncertain` | a `Bool` |

## One object, both directions

A composed object is a `Distributions.jl` distribution: it scores an observed record with `logpdf` and simulates a new one with `rand`, so a model is built once and used in both directions.

```@example concepts
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))

params_table(tree)
```

Read [Composing distributions](@ref composing-distributions) for each verb worked through end to end.
