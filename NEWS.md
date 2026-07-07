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
