# [Concepts: composition verbs and primitives](@id concepts)

A composed distribution is a multi-state event process: named events linked by delays, wired into a tree.
This page maps the modelling concepts to the verbs that build them, so you can find the right primitive by intent before reading the worked examples in [Composing distributions](@ref composing-distributions).

## The layers

The package has four layers, each building on the one before.

- **Leaves** are any `Distributions.jl` `UnivariateDistribution`, used directly as the per-event delays.
  A leaf can also resolve later: a [`Varying`](@ref) leaf maps an observed covariate to a distribution, and an [`Uncertain`](@ref) leaf carries distribution-valued parameters.
- **Composers** wire named leaves into an event tree ([`compose`](@ref) and the five composers).
- **Combination and lowering** join or collapse whole delays ([`convolve_distributions`](@ref), [`difference`](@ref), [`observed_distribution`](@ref)).
- **Parameters and edits** read and reshape an assembled tree ([`params_table`](@ref), [`build_priors`](@ref), [`update`](@ref), [`prune`](@ref), [`splice`](@ref)).

## The verb map

Every operator falls into one of four families.

```text
Structural composition (wire named branches into a tree)
├─ compose      lower a NamedTuple / table / matrix to the stack
├─ sequential   conjunctive chain (steps add up)
├─ parallel     independent branches off one shared origin
├─ resolve      one outcome occurs by fixed probability
├─ compete      racing hazards, first to fire wins
├─ choose       a data field picks the branch
├─ shared       tag a leaf as a tied parameter group at build time
└─ tie          tie leaves at paths into one parameter group

Combination and lowering (join or collapse whole delays)
├─ convolve_distributions   the sum X + Y (Convolved)
├─ difference               the dual X - Y (Difference)
├─ as_mixture               the MixtureModel view of a one_of node
└─ observed_distribution    collapse a chain to its convolved total

Parameters (read and prior the free parameters)
├─ params_table    the flat free-parameter inventory (a Tables.jl table)
├─ build_priors    support-derived default priors from that table
└─ default_prior   the default prior for one parameter row

Reading and editing (inspect or reshape an assembled tree)
├─ event / event_names / event_tree  fetch a child / the record key names
├─ mean / var                        the composed marginal moments
├─ update                            replace values or whole nodes
└─ prune / splice                    drop or insert a branch (topology)

Deferred leaves (a leaf whose distribution resolves later)
├─ varying / Context / instantiate   observed covariate picks the leaf
├─ has_varying                       guard: un-instantiated leaves remain
├─ uncertain                         distribution-valued parameters (priors)
└─ has_uncertain                     guard: uncertain leaves remain
```

## Concept to primitive

| Modelling concept | Primitive | What it builds |
|---|---|---|
| Steps in series (a chain) | `sequential` / `compose` with a `Vector` value | [`Sequential`](@ref) |
| Branches off one shared origin | `parallel` / `compose` with a `NamedTuple` | [`Parallel`](@ref) |
| One outcome by fixed probability | `resolve` | [`Resolve`](@ref) |
| Racing outcomes (first to fire wins) | `compete` | [`Compete`](@ref) |
| A data field selects the sub-model | `choose` | [`Choose`](@ref) |
| Tie a leaf across branches | `shared` / `tie` | one shared parameter group |
| Sum of two independent delays | `convolve_distributions` | [`Convolved`](@ref) |
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
