# Design note 0001 — Time-, strata-, and uncertainty-varying distributions

**Status:** draft for discussion
**Author:** design review (opened for @seabbs)
**Relates to:** a future `ConvolvedDistributions` convolution of a composed
stack; the parallel *uncertain distributions* work (distributions with
distributions on their parameters); the "recurrent operator" idea.

## The question being reviewed

> We have a recurrent operator which must have some concept of time. We will
> also want this to convolve a composed stack against a vector whose length is
> time. Distributions and delays can evolve over time. Do we represent that in
> the composed stack or in `ConvolvedDistributions`? I lean towards the composed
> stack. There is a relationship to distributions that vary by strata (not just
> time), and to the *uncertain distributions* work (priors on parameters), which
> overlaps. Is this its own package with an extension, or is it a composed dist
> so it fits here?

This note argues for a specific answer, grounds it in the code as it stands
today, and sketches the concrete seam to build. The short version:

1. **Two different things are both being called "time" and they must be kept
   apart.** One is the *output axis of a convolution* (a vector indexed by
   time). The other is *non-stationarity of the distribution itself* (its
   parameters change with an index). They are orthogonal and belong in
   different layers.
2. **Non-stationarity is a property of the leaf, not of the composer.** Time,
   strata, and parameter-uncertainty are three instances of one generalisation:
   *a leaf is no longer a fixed `Distribution` but a map from a context to a
   `Distribution`.* This should be modelled by generalising the leaf via a
   resolution seam, **not** by adding new composer verbs.
3. **The seam belongs in `ComposedDistributions`; the concrete varying-leaf
   types can live in a companion package.** The seam threads a context through
   the hot path, so it has to be core. Concrete leaves (a time-varying leaf, an
   uncertain leaf) plug in through it, exactly as censored and modified leaves
   already plug into `free_leaf` / `rewrap_leaf` from their own packages.
4. **`ConvolvedDistributions` stays a stationary-kernel convolver.** The
   composed stack resolves a kernel *per time step* and hands it over already
   concrete. Non-stationarity is resolved *before* convolution, not inside it.
5. **The recurrent operator is a renewal-layer citizen, not a new composer.** It
   consumes a per-step kernel from the composed stack; its "concept of time" is
   the index over the output vector, which is convolution-axis time, i.e. point 1.

## Two axes that are both called "time"

The request conflates two concepts. Separating them is the single most
important move in this design, because they resolve to different packages.

### Axis A — the convolution / renewal output axis (time-as-index)

When you "convolve a composed stack against a vector whose length is time", the
vector is a time series (an incidence curve indexed by day, say). Time here is
the discretised *index* over which outputs are produced; the length of the
vector is the horizon. This is renewal/convolution machinery: take a delay
*kernel* and apply it across a time series.

`ComposedDistributions` already meets this layer at exactly one point:
`observed_distribution` (`src/composers/observed.jl`) collapses a `Sequential`
chain to the single scalar it observes — the convolution of its steps, returned
as a `Convolved` — and `convolve_distributions(::Sequential)` extends the
`ConvolvedDistributions` verb to a composed stack. That kernel is then the
thing a convolution/renewal step applies across the vector. **The vector-as-time
axis is not, and should not become, a concept inside the composer types.** A
composer produces a kernel; something downstream sweeps it over time.

### Axis B — non-stationarity of the distribution (parameter-as-function)

"Distributions and delays can evolve over time" is a different statement. It
says the delay *itself* is not fixed: the onset→admit delay might be 5 days in
one wave and 3 in the next, or the CFR of a `Resolve` node drifts, or a delay
differs by region. The distribution is no longer a single object; it is a map

```
index ↦ Distribution
```

where the index is calendar time, a wave, an age band, a region — a covariate.
This is *non-stationarity*, and it is a property of the leaf (or of a node's
branch probabilities), independent of whether anything is ever convolved.

### Why the split matters

The two axes are orthogonal:

- Axis B without A: a strata-varying delay you score against a line list and
  never convolve.
- Axis A without B: a perfectly stationary kernel swept over a renewal process.
- Both together: a non-stationary kernel `kernel(t)` swept over the renewal axis
  — the interesting epidemiological case, where today's convolution weight uses
  today's delay distribution.

The user's instinct — "it should be part of composed distributions" — is
**correct for Axis B** and **not the right home for Axis A**. Axis B is the
leaf-varies-with-a-covariate story and belongs here. Axis A is the
sweep-a-kernel-over-time story and belongs in the convolution/renewal layer,
which merely *asks* the composed stack for the right kernel at each step.

## The real abstraction: a context-indexed leaf

The user already sees the generalisation — "not just time but strata etc." Name
it: a leaf becomes a **context-indexed distribution family**. Today a leaf is a
`UnivariateDistribution`. Generalise it to something that, given a *context*,
yields a `UnivariateDistribution`:

```
instantiate(leaf, context) -> UnivariateDistribution
```

- Time-varying leaf: `instantiate(l, ctx) = l.f(ctx.time)` — parameters a
  function of calendar time.
- Strata-varying leaf: `instantiate(l, ctx) = l.by_stratum[ctx.region]` — a
  categorical lookup.
- Fixed leaf (everything today): `instantiate(d::UnivariateDistribution, _) = d`
  — ignores the context. This identity default is what makes the whole change
  backward compatible.

Time is not special; it is one continuous covariate. Strata is a categorical
covariate. Both are the *same* seam with different index types. **This is the
key claim: do not add a `TimeVarying` composer and a separate `Strata` composer.
Add one seam and let leaves vary however they like.**

### The package already has this pattern twice

This is not a foreign idea grafted on — the codebase already indexes behaviour by
per-record data in two places, which is strong evidence the seam belongs here:

- **`Choose`** (`src/composers/Choose.jl`) is a *data-selected* disjunction:
  a record carries a `selector` field (e.g. `:kind`) and that value picks which
  alternative scores/samples. That is exactly "the distribution depends on an
  observed covariate", for a categorical covariate with a small fixed set of
  outcomes. A context-indexed leaf is the continuous / open generalisation of
  the same idea. `Choose` and a strata-varying leaf are two points on one
  spectrum, and the design should make them share the covariate-threading
  machinery rather than reinvent it.
- **Reserved row fields** (`src/composers/tree_events.jl`, `_RESERVED_ROW_FIELDS`)
  already thread per-record, non-event covariates through scoring —
  `obs_time` (a per-record observation horizon), `obs_window`, per-record
  `branch_probs`. The infrastructure to carry a per-record context to the
  scoring path *exists*; a context-indexed leaf extends that channel to reach the
  leaves, not just the truncation/observation logic.

So the seam is a natural extension of contracts already in the package
(`free_leaf`/`rewrap_leaf` for leaf structure, the reserved-field channel for
per-record covariates, `Choose`'s selector for data-driven dispatch), not a new
paradigm.

## Where uncertain distributions fit (and why to coordinate now)

The parallel *uncertain distributions* effort — distributions with distributions
on their parameters — is the **same generalisation of the leaf, along a different
index**:

| | leaf becomes | index is | map is | resolved by |
|---|---|---|---|---|
| Non-stationary (this note) | `ctx ↦ Distribution` | an **observed** covariate (time, stratum, region) | deterministic | looking up the covariate |
| Uncertain (other effort) | `θ ↦ Distribution`, `θ ~ P` | a **latent** parameter | random | sampling/marginalising `θ` |

Both replace "a fixed distribution" with "a distribution whose parameters vary".
They differ only in whether the index is **observed** (deterministic function of
data) or **latent** (a random variable to be sampled or integrated out). That is
a real distinction with modelling consequences — one is conditioning, the other
is marginalising — but it is a distinction *within a single leaf protocol*, not a
reason for two unrelated mechanisms.

Concretely they should share `instantiate(leaf, context)` where `context`
carries both observed covariates *and* any sampled parameter values:

- an uncertain leaf reads its sampled `θ` from `context` (placed there by the
  Turing/`LogDensityProblems` layer — issues #9, #13 — during sampling);
- a non-stationary leaf reads `context.time` (or `.region`);
- **a leaf can be both** — a time-varying delay whose time-slope carries a prior.
  That composition is only possible if the two efforts share one context object
  and one resolution seam. If they ship two separate mechanisms, "an uncertain,
  time-varying leaf" becomes impossible or requires a bridge no one wants to
  write.

**Recommendation: agree the `context` object and the `instantiate` signature
jointly with the uncertain-distributions work before either lands.** This is the
highest-leverage coordination point and the main reason to write the seam down
now rather than build time-varying leaves in isolation.

## Package boundaries: what is core vs. companion

The user's either/or — "its own package with an extension, or a composed dist so
it fits here" — resolves to **both, split along the seam**:

- **The seam is core `ComposedDistributions`.** Threading a `context` through
  `logpdf` / `rand` / `observed_distribution` touches the hot-path signatures of
  the composer types. That cannot be a weak-dependency extension; extensions add
  *methods for their own types*, they cannot change the core call signatures.
  The abstract protocol (`instantiate`, the `context` type, the identity default
  for fixed leaves, and the context-threading through the composers) must live
  here, alongside `free_leaf`/`rewrap_leaf` and `child_logpdf`/`child_rand!` in
  the public extension contract (`src/public.jl`).
- **Concrete varying-leaf *types* can live in a companion package.** A
  `TimeVaryingDelay` / `StrataDelay` type, and the uncertain-leaf type, are
  ordinary leaves that implement the seam. They can sit in their own package(s)
  and register through the protocol, precisely mirroring how `Affine`/`Weighted`/
  `Transformed` live in `ModifiedDistributions` and plug in via the
  `ComposedDistributions × ModifiedDistributions` extension
  (`ext/ComposedDistributionsModifiedDistributionsExt.jl`), and how censored
  leaves stay in `CensoredDistributions`. The no-piracy invariant is preserved:
  each companion owns its own leaf type and implements a
  `ComposedDistributions`-owned function on it.

So: **build the seam here now; ship concrete time/strata leaves in a companion
package (or as a weak-dep extension) later.** "Part of composed distributions"
and "its own package with an extension" are both right — they are answering
about the two halves.

There is one judgement call worth surfacing: whether the *simplest* time-varying
leaf (a closure `t ↦ Distribution`) is lightweight enough to live in core behind
the seam as a convenience, with only the heavier, dependency-carrying variants
(spline bases, GP-driven parameters, Catalyst-coupled forms) pushed to
companions. That is a reasonable middle path and can be decided when the seam
lands; it does not change the architecture.

## `ConvolvedDistributions` stays stationary

Do **not** teach `ConvolvedDistributions` about time-variation. It should remain
a convolution + quadrature layer over *concrete* distributions. When leaves are
non-stationary, the composed stack resolves the kernel at each step and hands a
concrete `Convolved` to the convolver:

```
observed_distribution(stack, ctx_at_t)  ->  a concrete Convolved kernel for time t
```

The renewal/convolution sweep asks the composed stack for `kernel(t)` at each
`t` and convolves as it does today. Non-stationarity is resolved on the
`ComposedDistributions` side of the boundary; `ConvolvedDistributions` never sees
a varying object. This keeps the convolver simple, keeps AD flowing through
already-concrete kernels, and means the existing `convolve_distributions(::Sequential)`
path generalises to `convolve_distributions(::Sequential, ctx)` without any change
to the convolution engine itself.

## The recurrent operator

Reading the request literally — "a recurrent operator which must have some
concept of time" — there are two possible meanings, and the design says
different things about each:

- **Recurrence over the time axis (renewal).** An operator that applies a kernel
  at each step and feeds outputs back as inputs across the output vector. Its
  "concept of time" is Axis A: the index over the vector. This is a *renewal-layer
  operator*, not a `ComposedDistributions` verb. It consumes the per-step kernel
  the composed stack exposes (`observed_distribution(stack, ctx_at_t)`). Keep it
  in the time-series/renewal layer; give it a clean `kernel_at(stack, t)` seam
  into the composed stack. It is a *consumer* of the grammar, not a member of it.
- **Recurrence over composition (a repeated stack).** If "recurrent" means "the
  same composed sub-stack repeated N times", that is sugar over `Sequential`
  (an N-fold chain of a shared sub-tree) and *can* be a composer-level
  convenience. But note this still has no intrinsic notion of time — it is
  structural repetition — so the "concept of time" it needs is again supplied by
  the context seam (Axis B) if the repeated step is non-stationary.

Either way, the operator does not motivate a new time concept *inside* the
composer types. It motivates (a) the context seam for non-stationary kernels and
(b) a clean kernel-per-step accessor for the renewal layer to call.

## Sketch of the seam (illustrative, not final)

Enough concreteness to argue about; the exact spelling is for review.

```julia
# --- core: ComposedDistributions ---

# A per-record / per-step context. Carries observed covariates (time, strata)
# and any sampled parameter values (for uncertain leaves). Shared with the
# uncertain-distributions work — this type is the coordination surface.
abstract type AbstractContext end

# The resolution seam. Identity for a fixed leaf, so all existing code is
# unchanged and no context need ever be passed unless a leaf actually varies.
instantiate(d::UnivariateDistribution, ::AbstractContext) = d
instantiate(d::UnivariateDistribution, ::Nothing)        = d

# Composers thread the context to the leaves. logpdf / rand gain an optional
# context; the default (nothing) reproduces today's behaviour exactly.
logpdf(d::Sequential, x::AbstractVector; context = nothing) = ...  # forwards context to leaves
rand(rng, d::Sequential; context = nothing)                 = ...  # forwards context to leaves

# The kernel accessor the renewal / convolution layer calls per step.
observed_distribution(d::Sequential, context) = ...  # resolve leaves, then convolve

# --- companion package (or weak-dep extension): concrete varying leaves ---

struct TimeVaryingDelay{F} <: UnivariateDistribution{Continuous}
    f::F            # time -> UnivariateDistribution
end
instantiate(l::TimeVaryingDelay, ctx::AbstractContext) = l.f(ctx.time)

# Introspection/edits reuse the existing leaf contract:
free_leaf(l::TimeVaryingDelay)        = ...   # its free parameters
rewrap_leaf(l::TimeVaryingDelay, new) = ...
```

Key properties this sketch is chosen to have:

- **Backward compatible.** The identity `instantiate` and the `nothing` context
  default mean nothing changes for fixed-leaf trees; the context is opt-in.
- **Type-stable hot path.** Context threading is mechanical forwarding down the
  same flat-slice recursion the composers already use (`nesting.jl`); it bottoms
  out at `instantiate(leaf, ctx)`. No runtime dispatch on covariate values beyond
  what `Choose` already does with its type-stable `_pick`.
- **One seam, many index kinds.** Time, strata, and uncertainty are leaf types
  over one `instantiate`, not three parallel mechanisms.
- **Convolver untouched.** `ConvolvedDistributions` receives concrete kernels.

## What this PR implements

This PR ships a first, deliberately minimal cut of recommendation 1–3 so the
seam is concrete enough to build on and review against real code, while the
larger questions stay open:

- **`AbstractContext` + `Context`** (`src/composers/varying.jl`) — the covariate
  bag. `Context` is an open `NamedTuple` (`Context(time = 5.0)`,
  `Context(region = :north)`), so it already has room for the uncertain-work's
  sampled parameters. `AbstractContext` is the extension point for that work.
- **`Varying` leaf** — a `UnivariateDistribution` holding a map
  `covariate ↦ Distribution`, a `covariate` name (default `:time`), and a
  `reference` distribution. It drops into `Sequential` / `Parallel` / `compose`
  as an ordinary leaf and delegates every `Distributions` method to its
  reference, so existing trees are unaffected and a varying tree is fully usable
  at its reference by default.
- **`instantiate(tree, ctx)`** — the resolution seam, implemented as a
  structure-preserving tree transform: it rebuilds the composer against the
  context with every varying leaf resolved, and is the identity on fixed leaves
  and on a `nothing` context (so it is backward compatible and opt-in). This is
  the "resolve the kernel, then convolve/score" layering the note argues for:
  `observed_distribution(instantiate(stack, ctx_at_t))` is the kernel at `t`.
- **Introspection transparency** — `free_leaf` / `rewrap_leaf` / `_shared_tag`
  peel and rebuild through a `Varying` leaf, treating the varying map as fixed
  structure and the reference's parameters as the free parameters, exactly as
  censoring bounds and modifiers already do.

Also in this PR, extending the seam past bare leaves:

- **Node-level variation** — a whole univariate node varies with the context. A
  time-varying `Resolve` CFR is `varying(t -> resolve(:death => (d, cfr(t)), …))`:
  because a `Resolve` is itself a `UnivariateDistribution`, node-level variation
  falls out of the leaf seam with no new type, and `instantiate` resolves the node
  in place inside a chain.
- **`Choose` on the seam** — `Choose`'s `selector` is the categorical instance of
  covariate indexing, so it now joins `instantiate`: if the context carries the
  selector, `instantiate` SELECTS that alternative and resolves it (collapsing the
  disjunction); without it, every alternative is resolved and the `Choose` kept.
  One mechanism now spans categorical selection and continuous covariate indexing.
- **The uncertain-work contract** — `Context` is the shared covariate bag and
  `with_covariates(ctx; …)` threads the sampler's LATENT parameters into it, so an
  uncertain leaf is just a `Varying` leaf keyed on a sampled-parameter name.
  Observed and latent indices are one covariate channel; only who fills the slot
  differs. `AbstractContext` is the type the uncertain work extends.

Deliberately **not** in this PR, and tracked by the open questions below: full
inventory of a varying map's own coefficients in `params_table` (a varying node
is still opaque to introspection behind its reference), node-level variation for
MULTIVARIATE nodes (a `Parallel`'s structure, as opposed to its leaves, which are
already resolved), the renewal/recurrent operator itself, and ratifying the
`AbstractContext` contract jointly with the uncertain-distributions author.

## Recommendation

1. **Model non-stationarity as a context-indexed leaf, not a new composer.** Add
   an `instantiate(leaf, context)` seam to core `ComposedDistributions` with an
   identity default, and thread an optional `context` through the composer hot
   path. Fold `Choose`'s selector into the same covariate machinery.
2. **Keep the convolution/renewal axis out of the composer types.** The composed
   stack exposes a per-step kernel (`observed_distribution(stack, ctx_at_t)`);
   the renewal/recurrent operator sweeps it. `ConvolvedDistributions` stays a
   stationary-kernel convolver.
3. **Put the seam in core; ship concrete time/strata leaves in a companion
   package** registered through the seam (mirroring `ModifiedDistributions` /
   `CensoredDistributions`). "Fits in composed distributions" (the seam) and "its
   own package with an extension" (the leaves) are both correct.
4. **Coordinate the `context` object and `instantiate` signature with the
   uncertain-distributions work now.** Observed-covariate variation and
   latent-parameter uncertainty are the same generalisation of the leaf over an
   observed vs. latent index; they must share one context so a leaf can be both.
5. **Treat the recurrent operator as a renewal-layer consumer** of the composed
   stack's per-step kernel, not a `ComposedDistributions` verb.

## Open questions for review

- **Context shape (partly settled).** This PR uses `Context`, an open
  `NamedTuple` bag under `AbstractContext`, with no mandatory field —
  `with_covariates` merges observed covariates and sampled parameters into one
  channel. Still to ratify jointly with the uncertain-distributions author:
  whether sampled parameters want their own reserved namespace inside the bag
  (e.g. `ctx.params.…`) rather than sitting flat alongside `time`/`region`, and
  whether `AbstractContext` needs a formal accessor protocol (`_covariate` /
  `_has_covariate`) so a non-`Context` subtype can plug in.
- **`Choose` unification (done here).** `Choose` now rides the seam: with the
  selector in the context, `instantiate` selects and resolves that alternative;
  without it, all alternatives resolve and the `Choose` is kept. Open: whether the
  hot-path `logpdf(::Choose, x; kind)` should also be re-expressed on the context
  channel, or stay the specialised fast path it is today.
- **Node-level variation (done for univariate nodes).** A time-varying `Resolve`
  CFR works as `varying(t -> resolve(…))` because a `Resolve` is univariate; the
  node resolves in place inside a chain. Open: a MULTIVARIATE node whose own
  structure (not just its leaves) varies — a `Parallel` cannot wrap in the
  univariate `Varying`, though its leaves already resolve via descent.
- **Core convenience vs. companion-only.** The bare closure `covariate ↦ Distribution`
  (`Varying`) lives in core behind the seam; heavier variants (splines, GPs,
  Catalyst-coupled) still belong in companions.
- **Interaction with `params_table` / priors (#4).** Still open. A varying leaf's
  free parameters are the coefficients of its `f` (e.g. spline weights), not the
  parameters of any one realised `Distribution`. This PR treats the varying map as
  fixed structure and inventories the reference's parameters; a fuller answer
  (introspecting `f`'s own coefficients) is future work.
