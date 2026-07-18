# [Frequently asked questions](@id faq)

Short answers to the questions this package tends to raise.
For a full walkthrough see [Composing distributions](@ref composing-distributions), and for the verb map see the [Concepts](@ref concepts) page.

## Where did censoring go?

ComposedDistributions is the generic composition layer, split out from [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl).
It composes any `Distributions.jl` `UnivariateDistribution` and adds no censoring of its own.
If you need primary-event censoring, interval censoring, or truncation of the observation process, use CensoredDistributions.jl, which builds its censored leaves on top of this composition algebra.
Right-truncation of a plain leaf is still available through `truncated()` from Distributions.jl, since any leaf is an ordinary distribution.

## What does `rand` of a `Resolve` or `Compete` node return?

A one_of node draws the named event record of the outcome that fired: a `NamedTuple` keyed by [`event_names`](@ref), a positional origin slot then one slot per outcome, with the fired outcome's time present and the others `missing`.
The record names which outcome occurred, so it feeds straight back into `logpdf`.

```@example faq
using ComposedDistributions, Distributions, Random

node = resolve(:death => (Gamma(1.5, 1.0), 0.3), :recover => Gamma(2.0, 1.5))
rand(Xoshiro(1), node)
```

For the compact `(outcome, time)` pair pass [`rand`](@ref)`(node; outcome = true)`; for the marginal time to resolution alone (discarding which outcome fired) sample [`as_mixture`](@ref)`(node)`.

```@example faq
rand(Xoshiro(1), node; outcome = true),
rand(Xoshiro(1), as_mixture(node))
```

## I'm migrating from CensoredDistributions' `compose`: what changed?

CensoredDistributions' `compose` took varargs pairs, `compose(:a => d1, :b => d2)`.
This package's `compose` takes a `NamedTuple`, `compose((a = d1, b = d2))`, as its primary spelling, because a `NamedTuple` also carries recursively-nested branch names for tables and matrices, where a flat pairs list cannot.
The varargs-pairs spelling still works: `compose(:a => d1, :b => d2, ...)` is a thin convenience method that lowers to the `NamedTuple` form and returns the identical stack, so existing CensoredDistributions-style call sites keep working unmodified.
Prefer the `NamedTuple` form in new code.

## How do I get the total-delay distribution of a chain?

A [`Sequential`](@ref) chain models each step separately; to collapse it to the single distribution of its origin-to-final gap, use [`observed_distribution`](@ref), which convolves the steps into one delay.
For two standalone delays that are not part of a tree, [`convolved`](@ref) gives their sum `X + Y` directly.
See [Delay chains and the linear chain trick](@ref linear-chain) for a worked example.

## How do I fix a parameter across branches?

Use [`shared`](@ref) or [`tie`](@ref) to make several leaves one free parameter: `shared` tags a leaf where it is built, `tie` walks an assembled tree to named leaves and ties them.
Either way [`params_table`](@ref) then inventories the tied occurrences once under the shared tag rather than as separate parameters.
See [Composing distributions](@ref composing-distributions) for a worked example.

## Why is my `logpdf` `-Inf`?

A composed `logpdf` is `-Inf` when a value falls outside a leaf's support.
The commonest cause is a negative or zero value scored against a positive-support delay, such as a `Gamma` or `LogNormal`.
When scoring a whole record, check that every value sits in its own leaf's support and that the record is laid out in the tree's event order (see [`event_names`](@ref)); a value in the wrong slot can land outside the leaf that scores it.

## How do I fit a composed distribution to data?

Mark the parameters to estimate by building a leaf with [`uncertain`](@ref), or promote a leaf/subtree/whole tree already in place with [`uncertain`](@ref)`(tree, ...)` / `uncertain(tree)`, then package the tree and data into a log-density with [`as_logdensity`](@ref), or wrap it as a `DynamicPPL` model with [`as_turing`](@ref) for direct sampling with Turing.jl.
See [Fitting a composed distribution](@ref inference) for the full pipeline.

## Is a composed distribution really a `Distribution`?

Yes.
Every composer subtypes `Distributions.Distribution`, so `logpdf`, `rand`, `mean`, `var`, `cdf` (where defined) and the rest of the interface work unchanged, and a composed object drops into any code that expects a distribution.
