## Unreleased

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

- `convolve_distributions(chain, series; events)` convolves a timeseries to a
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
  convolution bridge (`observed_distribution`, `convolve_distributions`).
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
  `convolve_distributions(chain, series)` convolves a timeseries (e.g. expected
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
