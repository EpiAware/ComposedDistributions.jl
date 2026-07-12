## Unreleased

- **feat:** `to_constrained(prob, z)` completes the PPL-neutral codec's HMC
  surface: given an assembled `ComposedLogDensity` and an unconstrained flat
  vector, it returns the constrained ESTIMATED parameters and the
  log-determinant Jacobian a sampler needs
  (`logdensity(prob, x) + logjac` is the unconstrained-space target). The
  transform is built per row from each row's prior via `Bijectors.bijector`
  (a stick-breaking `Beta` row, a positive-support prior, a non-centred
  pooled latent/hyperparameter), or, for a centred-pooled row, from its
  population's family. It lives in a new `ComposedDistributionsBijectorsExt`
  weakdep extension, so the core codec stays free of a `Bijectors`
  dependency.

- **Breaking (upstream-driven):** adopt the ConvolvedDistributions AD-seam move
  (#137): ConvolvedDistributions 0.2 relocated its AD-safe hook family out to the
  new `EpiAwareADTools` package under underscore-free names, so the racing-hazard
  node now calls and extends `EpiAwareADTools.logccdf_ad_safe` /
  `ccdf_ad_safe` (was `ConvolvedDistributions._logccdf_ad_safe` /
  `._ccdf_ad_safe`). `EpiAwareADTools` is a new dependency; it is unregistered, so
  it is git-pinned in the root and isolated (`test/ad`, `test/jet`, `benchmark`)
  environments until it registers. No user-facing API change.

- **Breaking (upstream-driven):** adopt ConvolvedDistributions 0.2, which makes
  the bare-distribution `convolve_series(delay, series)` discrete-only — a
  continuous delay now throws, because discretising it is an explicit modelling
  choice (single- vs double-interval censoring) upstream will not make silently
  (ConvolvedDistributions #31/#47). The composed-tree convenience is preserved:
  `convolve_series(::Sequential, series; events)` and
  `convolve_series(::Resolve/::Compete, series)` collapse to their continuous
  observed total and discretise it for you with the interval-censored-secondary
  scheme (`discretise_pmf` over lags `0:(length(series) - 1)`) before convolving,
  so the composed output is unchanged from before. For day-binned
  (double-interval-censored) primaries, discretise the total yourself and pass
  the PMF to `convolve_series(pmf, series)`. `discretise_pmf` and `DelayPMF` are
  now re-exported. Compat bumped to `0.2`; because 0.2 is unregistered the source
  is git-pinned (re-adding what #107 removed) until it registers.

- **test:** end-to-end continuous delay-stack scenarios. A committed
  `test/composers/stack_scenarios.jl` testset drives a handful of named,
  epi-flavoured continuous stacks (an onset→admission→death `Sequential` chain,
  a `Parallel` of independent reporting branches, a death-vs-discharge
  `Resolve`, a competing-causes `Compete`, nested composes mixing them, a
  renewal `convolve_series`, a `tie` group and a `difference`) through the whole
  verb surface together — construction via both spellings, `rand`/`logpdf`
  round-trip, a seeded large-N Monte-Carlo moment check against the analytic /
  quadrature values, the introspection / edit / prior surface, and a ForwardDiff
  gradient per stack.

- Overall moments of a composed tree now honour an `affine` modifier: a chain
  with an `affine(delay; scale, shift)` step reports the scale/shift-adjusted
  mean/var (matching what `rand` draws) instead of peeling the affine off to the
  inner delay's moment. A hazard-modified (`Modified`) leaf has no analytic
  moment yet, so a chain containing one now errors informatively rather than
  silently returning the unmodified free-leaf moment, pending
  ModifiedDistributions#44's numeric cumulative-hazard path (#120).

- **Breaking (upstream-driven):** following ConvolvedDistributions' rename, the
  re-exported `convolve_distributions` is split into two verbs —
  `convolved(dists...; method)` for the distribution form (the sum `X + Y`, a
  chain's observed total) and `convolve_series(delay, series; interval)` for the
  timeseries form (convolving a numeric series through a delay). The composed-tree
  methods follow suit: `convolved(::Sequential)` collapses a chain to its total,
  and `convolve_series(::Sequential, series; events)` drives the renewal / latent
  series. No alias is kept (the package is unreleased).

- `sequential`, `parallel`, `resolve`, `compete` and `choose` now accept a
  positional `NamedTuple` spelling
  (`resolve((death = (Gamma(1.5, 1.0), 0.3), disch = Gamma(2.0, 1.5)))`) as the
  equivalent of the `name => value` Pairs, for hand-written children;
  `choose`'s `selector` stays a keyword. The one_of constructors also build their
  outcome tuples with `map` rather than a generator comprehension, so
  constructing a `Resolve` / `Compete` inside a differentiated function is
  Enzyme-safe (no `collect_to!` `Array` temporary Enzyme's type analysis rejects).

- `rand_outcome` is now a documented `public` binding (previously an undocumented
  internal), matching CensoredDistributions. It stays unexported — the
  record-returning `rand` is the exported entry point — so reach it
  module-qualified as `ComposedDistributions.rand_outcome`.

- **Breaking:** `rand` of a standalone `Resolve` or `Compete` node now returns
  the named event record of the outcome that fired — a `NamedTuple` keyed by
  `event_names(node)` (a positional origin slot then one slot per outcome, the
  fired outcome's time present and the others `missing`) — instead of the scalar
  marginal time-to-resolution (#96, syncing to CensoredDistributions' #639). The
  record names which outcome occurred, so `logpdf(node, rand(node))` round-trips
  and identifies the outcome. To recover the old scalar draw, sample the
  marginal `rand(as_mixture(node))`; the `(outcome, time)` pair view stays
  `rand_outcome`. A one_of node nested inside a `Sequential` / `Parallel` is
  unchanged (it stays one scalar value slot, its marginal); a new
  `logpdf(node, ::NamedTuple)` scores a standalone record.

- `params_table` is now a superset schema carrying both the uncertain-first
  `prior` column and CensoredDistributions' `:thin` rows via the `_thin_factor`
  / `_set_thin_factor` hooks (no-op here, so no `:thin` row appears; the hooks
  let a thinning modifier layer plug in). `_leaf_detail_lines` becomes the
  per-leaf `inspect` detail extension point, and the racing-hazard moment /
  winning-probability / cause-cdf quadratures thread a shared 64-node
  Gauss-Legendre rule (#96).

- Partial pooling across strata (#78). A new `pool(group, population)` spec,
  placed inside `uncertain` where a prior would go, declares a parameter
  partially pooled across the leaves that name the same group: each member's
  parameter is drawn from one shared `population` distribution whose own free
  parameters are the estimated hyperparameters (carrying their priors through
  the ordinary `uncertain` spec machinery). It is the middle of the pooling
  spectrum between `shared`/`tie` (complete pooling, one value everywhere) and
  independent `uncertain` specs (no pooling, K unlinked values). A location-scale
  population (`Normal`/`LogNormal`) is reparameterised non-centred (one
  `Normal(0, 1)` latent per member, member `k` reconstructed as `mu + tau*z_k`
  or `exp(mu + tau*z_k)`), keeping the CensoredDistributions-compatible
  `[hyper..., z...]` flat layout; a general population takes the centred path
  (each member's parameter scored directly against the population). The
  hyperparameters flatten as ordinary uncertain-spec rows on the population.

- Node-level uncertainty: a `Resolve`'s branch probabilities can now be
  estimated (#89). Attach a simplex-valued `Distributions.Dirichlet` prior with
  `update(node, (branch_probs = Dirichlet(α),))`. The `Dirichlet` is what you
  write; the node is estimated through its K-1 stick-breaking coordinates
  (`:stick_1 … :stick_{K-1}`, each a `Beta` in (0, 1)), so `params_table`, the
  uncertain-first codec (`flatten` / `unflatten` / `flat_dimension` /
  `as_logdensity`) and chain readback all carry the sticks, and the
  probabilities are recovered from any draw (they always sum to one and the
  gradient is well-defined on every AD backend). Promote
  (`update(tree, param_priors(tree))`) attaches a flat `Dirichlet(ones(K))` per
  `Resolve`. `Compete`'s winning probability is derived from the hazards and
  `Choose`'s alternative is data-selected, so neither has a node-level free
  parameter (documented, no change).

- `convolve_series(chain, series; events)` convolves a timeseries to a
  named INTERIM event of a `Sequential` chain, not just its endpoint. The
  cumulative delay to an event is the observed collapse of the chain prefix up
  to it, so a single event name returns that event's count series and a tuple or
  vector of names returns a `NamedTuple` of series (the endpoint reproduces the
  whole-chain result). Only a plain continuous chain (every step a delay leaf)
  has per-event cumulative delays; a branching step is rejected, and an unknown
  event name errors listing the valid events. The discrete-event and
  thinning/branch-probability variants stay in CensoredDistributions.

- See-through fitting of `Convolved` / `Difference` leaf component parameters,
  replacing the previous fixed-composite treatment. `params_table` now
  inventories each component's scalar parameters under a `component_i` path
  segment (e.g. `total.component_1.shape`), and `update` rebuilds the composite
  from the updated components (preserving the solver method). A component may be
  made `uncertain` in place, so the uncertain-first codec
  (`flatten` / `unflatten` / `flat_dimension` / `as_logdensity`) estimates a
  spec'd component parameter like any other leaf parameter. The composite joins
  the shared `_node_children`/`_rebuild` deferred-leaf walk, so `has_uncertain`,
  `has_varying` and `instantiate` all see through a composite carrying an
  uncertain or varying component; it stays a single flat scored slot and an
  atomic node to the structural edits.

- Reconciled the `Varying`/`instantiate` seam with the `Uncertain` machinery
  (#47): `Varying` and `Uncertain` are now presented as the two cases of one
  *deferred leaf* concept — a leaf that maps to a distribution and resolves
  later, `Varying` from an observed covariate (via `instantiate`) and
  `Uncertain` from a latent parameter draw (via `rand`/`update`). `instantiate`
  now rebuilds through the shared `_node_children`/`_rebuild` reconstruction
  machinery that `update` and the structural edits already use, and the
  `has_varying`/`has_uncertain` guards share one node walk, so resolution is no
  longer a separate hand-rolled tree traversal. No user-facing API change;
  `instantiate`, `update`, `has_varying`/`has_uncertain`, and the codec's
  rejection of an un-`instantiate`d `Varying` leaf are unchanged.

## 0.1.0 — initial release

- The generic composition algebra ported from CensoredDistributions.jl:
  `compose` and the five composers (`Sequential`, `Parallel`, `Resolve`,
  `Compete`, `Choose`), `shared`/`tie`, structural edits
  (`update`/`prune`/`splice`), introspection (`params_table`,
  `build_priors`, `event`/`event_names`/`event_tree`), moments, and the
  convolution bridge (`observed_distribution`, `convolved`).
- Added `Varying` / `Context` / `instantiate`: leaves whose distribution
  depends on an observed covariate (time, stratum), resolved by
  `instantiate(tree, ctx)`; `has_varying(tree)` guards fitting loops.
- `Resolve`/`Compete` outcome probabilities read via `Distributions.probs`
  (following the CensoredDistributions rename).
- Docs site (overview, concepts, three tutorials, FAQ, interface
  contracts), benchmarks (core + AD) with a docs page, and a Mooncake AD
  extension.
- Typed composer hierarchy: `AbstractComposedDistribution{F, S}` roots the
  composers, with `AbstractMultiChild{S}` grouping `Sequential`/`Parallel` and
  `AbstractOneOf` the univariate one_of family; downstream extension packages
  dispatch on these supertypes. The reusable
  `ComposedDistributions.TestUtils` harness (`test_interface`,
  `test_composed_interface`, `test_node_interface`, `test_abstract_membership`,
  ...) verifies a custom leaf or composer conforms to the interface.

- Added `uncertain` / `Uncertain`: leaf distributions whose parameters are
  themselves distributions, nestable to any depth. `rand` draws the marginal so
  uncertain leaves compose everywhere, and the rest of the univariate surface
  (scalar `logpdf`/`cdf`/..., the moments, including a composed tree's overall
  moment) delegates to the template's central values. `has_uncertain(tree)`
  flags a tree that still holds an uncertain leaf, for a scoring/fitting loop
  to guard against a forgotten collapse. Collapse an uncertain leaf to its
  concrete template by pinning the parameters with `update(tree, params)`.
  Build one with a concrete template
  (`uncertain(Gamma(2.0, 1.0); shape = LogNormal(...))`), a positional family
  form (`uncertain(Gamma, LogNormal(...), 1.0)`), or the keyword family form.
  `truncated(uncertain(...))` pushes inside the template (conditional
  per-draw semantics). `params_table` gained a `prior` column carrying an
  uncertain parameter's spec, which `build_priors` now uses ahead of its
  per-row default.

- Extended the ConvolvedDistributions verbs to composed trees:
  `convolve_series(chain, series)` convolves a timeseries (e.g. expected
  infections) through a `Sequential` chain's observed total delay (the
  renewal / latent observation layer), and `difference(a, b)` forms the
  difference of two chains' observed totals; a `Parallel` / `Choose` (no single
  observed delay) errors with guidance. A `Convolved` / `Difference` node used
  as a leaf inside a tree scores, samples, and reports moments as a plain
  univariate leaf, and is treated as fixed structure by `params_table` /
  `build_priors` / `update` (fit its components by composing them as explicit
  chain steps).

- Added the PPL-neutral LogDensityProblems core codec: the flat-vector <->
  nested-NamedTuple bijection (`flatten` / `unflatten` / `flat_dimension`) and
  the assembled `ComposedLogDensity` (`as_logdensity` / `logdensity`), with no
  DynamicPPL / Turing dependency. A `ForwardDiff` gradient flows through
  `logdensity`.

- Added the inference-readback verbs `chain_to_params` / `param_draws` /
  `strip_prefix` (and `update(template, chain)`): read a fitted chain's
  parameters back onto a composed-distribution template. Turing-free until both
  `DynamicPPL` and `FlexiChains` are loaded, when the extension supplies the
  methods.

- Uncertain-first estimation: the `uncertain` specs set the estimation
  boundary. `flatten` / `unflatten` / `flat_dimension` / `as_logdensity` /
  `logdensity` target EXACTLY the spec'd parameters, so a fixed leaf contributes
  no estimated dimension and a tree with no uncertain leaves estimates nothing
  (`flat_dimension == 0`; `logdensity` is then the data likelihood at the fixed
  tree). `update` introduces uncertainty through the prior interface: a
  distribution in a parameter slot makes just that parameter uncertain (a
  partial update), and `update(tree, param_priors(tree))` promotes a whole tree
  to uncertainty over its free parameters with default priors (the explicit
  estimate-everything path). `params_table` / `build_priors` are derived views
  over the object's specs; the readback verbs label exactly the spec'd
  parameters.

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
