# [Time-, strata-, and covariate-varying distributions](@id varying-distributions)

A composed tree is *stationary* by default: every leaf is a fixed `Distributions.jl` distribution.
Real delays are often **non-stationary** — an onset→admission delay shortens over a wave, a case-fatality ratio drifts, a delay differs by region.
`ComposedDistributions` models this by generalising the **leaf**, not by adding a new composer verb: a leaf becomes a map from a *context* to a distribution, and [`instantiate`](@ref) resolves a whole tree against a context.

!!! note "Two cases of one concept"
    [`Varying`](@ref) and [`Uncertain`](@ref) are the two *deferred leaves*.
    See [Concepts](@ref concepts) for how they differ (observed covariate vs latent parameter) and share one resolution walk.

## The three pieces

- [`Varying`](@ref) — a leaf holding a map `covariate ↦ Distribution`, the covariate it reads (default `:time`), and a `reference` distribution it behaves as when no context is supplied.
- [`Context`](@ref) — an open bag of covariates (`Context(time = 5.0)`, `Context(region = :north)`).
- [`instantiate`](@ref)`(tree, ctx)` — resolves every varying leaf against the context and returns the same tree made concrete.
  It is the identity on fixed leaves and on a `nothing` context, so existing stationary trees are untouched.

!!! warning "Always `instantiate` before scoring or sampling"
    A [`Varying`](@ref) leaf behaves as its `reference` distribution (e.g. the `t = 0` delay) until you call [`instantiate`](@ref).
    Scoring or sampling a raw tree that still holds a `Varying` leaf — `logpdf(tree, x)`, `rand(tree)` — does **not** error; it silently uses the reference, which is a wrong answer against real per-record times.
    Always resolve first (`resolved = instantiate(tree, Context(time = t))`) and score the resolved tree.
    In a fitting loop, guard the call with [`has_varying`](@ref): `@assert !has_varying(resolved)`.

## A time-varying delay

```@example varying
using ComposedDistributions, Distributions

# An onset→admission delay whose scale grows with calendar time.
d = varying(t -> Gamma(2.0, 1.0 + 0.02t))

mean(d)                                    # the reference (t = 0)
```

```@example varying
mean(instantiate(d, Context(time = 10.0))) # the delay at t = 10
```

A `Varying` leaf drops into any composer as an ordinary leaf, because it *is* a `UnivariateDistribution`:

```@example varying
chain = sequential(:onset_admit => varying(t -> Gamma(2.0, 1.0 + 0.02t)),
    :admit_death => LogNormal(0.5, 0.4))

instantiate(chain, Context(time = 10.0))   # a concrete, stationary chain
```

## Strata (a categorical covariate)

Time is just one covariate; a stratum is another.
Name the covariate and pass a `reference` (there is no meaningful `f(0.0)` for a categorical index):

```@example varying
by_region = varying(r -> r === :north ? Gamma(2.0, 1.0) : Gamma(3.0, 1.5);
    covariate = :region, reference = Gamma(2.0, 1.0))

instantiate(by_region, Context(region = :south))
```

## Node-level variation (a time-varying CFR)

Because a [`Resolve`](@ref) is itself univariate, a whole node can vary — e.g. a case-fatality ratio that rises over time — with no new machinery:

```@example varying
cfr(t) = 0.2 + 0.02t
node = varying(t -> resolve(:death => (Gamma(1.5, 1.0), cfr(t)),
    :disch => Gamma(2.0, 1.5)))

instantiate(node, Context(time = 10.0))    # a concrete Resolve, CFR = 0.4
```

## `Choose` resolves the same way

A [`Choose`](@ref) already selects an alternative by an observed data field.
That is the categorical case of covariate indexing, so it resolves the same way: give the selector in the context and `instantiate` collapses to the chosen branch.

```@example varying
disj = choose(:index => Gamma(2.0, 1.0), :sourced => Gamma(4.0, 1.5))

instantiate(disj, Context(kind = :index))
```

## Latent parameters (the uncertain-distributions bridge)

An *observed* covariate (time, region) and a *latent* parameter (one a sampler draws) are the same covariate channel — only who fills the slot differs.
[`with_covariates`](@ref) threads a latent value into the context alongside the observed covariates, so a leaf keyed on a parameter name resolves the same way:

```@example varying
latent = varying(θ -> Gamma(θ, 1.0); covariate = :inc_shape,
    reference = Gamma(2.0, 1.0))

ctx = with_covariates(Context(time = 4.0); inc_shape = 2.5)  # sampler adds θ
instantiate(latent, ctx)
```

This is the same generalisation as [`uncertain`](@ref), along a different index:

- A `Varying` leaf keyed on a sampled parameter is the bare bridge, resolved once the sampler fills the slot.
- [`Uncertain`](@ref) is the richer latent leaf: it also carries each parameter's **prior** (so [`params_table`](@ref) rides it on the `prior` column), draws the marginal with `rand`, and collapses via [`update`](@ref).
- Both are deferred leaves resolved by one machinery, so a leaf keyed on an observed covariate whose per-level parameter is itself `uncertain` is both cases at once.

## Feeding a recurrent / renewal operator

A recurrent (renewal) operator sweeps a delay **kernel** across a time series, one step per index — its "time" is the length of that vector.
That operator is a *consumer* of the composed stack, not a composer itself, and it is not part of this package; it lives in the renewal/time-series layer and calls `kernel_at` (below) once per step.
`instantiate` gives it exactly what it needs, a **kernel per time step**:

```@example varying
# Resolve the chain at each t, then collapse to its convolution kernel.
kernel_at(t) = observed_distribution(instantiate(chain, Context(time = t)))

kernels = [kernel_at(t) for t in 0:4]
mean.(kernels)                             # the kernel mean drifts with time
```

- Each `kernel_at(t)` is an ordinary, stationary `Convolved` distribution: non-stationarity is resolved *before* convolution.
- A `ConvolvedDistributions`-based renewal step convolves it exactly as it would a fixed kernel.
- `ComposedDistributions` answers *"what is the kernel at time `t`?"*; the renewal layer answers *"apply it across the time axis."*

## Learning more

- The full interface: [`Varying`](@ref), [`varying`](@ref), [`Context`](@ref), [`with_covariates`](@ref), [`instantiate`](@ref) in the [Public API](@ref public-api).
