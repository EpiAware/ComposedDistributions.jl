## Unreleased

- **chore:** renamed the two `public`-declared centred-pooling internals from
  `_centred_pool_rows`/`_pool_centred_logprior` to `centred_pool_rows`/
  `pool_centred_logprior` (org naming convention: a leading underscore marks
  internal-only, and #212 had declared these `public`). `_centred_pool_rows`
  and `_pool_centred_logprior` remain as `public` transitional aliases for a
  caller already qualifying the old names (DistributionsInference.jl's
  fit-protocol extension); the aliases are removed once that extension moves
  onto the renamed functions.
- **fix:** scoring a named record with an unobserved (`missing`) step or
  branch on a `Sequential`/`Parallel` composer no longer throws (#271). The
  named-record path previously built a `Vector{Float64}` and called `Float64`
  on every field, so a record such as `(first = 3.2, second = missing)`
  raised `MethodError: no method matching Float64(::Missing)`. `Sequential`
  and `Parallel` now carry a `Missing`-admitting `logpdf` that scores the
  observed values and integrates out the rest (each unobserved leaf's own
  marginal contributes zero log density); the named-record builder always
  produces a `Missing`-admitting vector, even for an all-observed draw, so one
  `logpdf` method is selected regardless of which fields happen to be
  present. `missing` in a slot means the value was not observed, the same
  convention already used by `Resolve`/`Compete`'s event records.
- **fix:** a `Resolve` node with a `NoEvent` branch now reports a well-defined
  defective marginal survival instead of throwing (#254). `cdf`/`ccdf`/`mean`
  previously errored for any node holding a no-event branch, so no consumer
  reading a component's survival could see through it. `cdf` now sums only
  the occurring branches, rising to `occurrence_probability` rather than one;
  `ccdf` (the generic `1 - cdf` fallback) comes along for free and flattens
  at the no-event mass instead of decaying to zero; `mean` reports the
  conditional-on-occurrence mean of the observed branches (there is no
  unconditional mean — the marginal has an atom at "never", no finite time),
  and throws when `occurrence_probability` is zero (no branch can occur, so
  no conditional mean exists) rather than silently returning `NaN` from a
  `0/0` division. `logpdf` and `as_mixture` are unchanged and still reject a
  no-event node (there is no proper `MixtureModel` over a marker with no
  density); a non-terminal node (a composer-valued outcome) still rejects
  `cdf`/`ccdf`/`mean`, no-event branch or not.
- **fix:** a racing-hazard (`Compete`) node's `probs`/`occurrence_probability`
  no longer raise a `MethodError` when a cause is itself a composite/
  convolved distribution with no `quantile` method (#259). The shared
  quadrature window's `_hazard_quad_window` called `quantile(cause, 0.9999)`
  directly on every cause; it now falls back to a moment-based
  (`mean + 10*std`) window when a cause's `quantile` is unavailable or
  non-finite, and ignores a cause with no usable window (neither a finite
  quantile nor finite moments) rather than letting it poison the shared
  integral. The support floor/ceiling feeding the window (previously the
  public `minimum`/`maximum(::Compete)`, which themselves throw if any
  cause's own `minimum`/`maximum` throws) are now built the same
  fallback-robust way. This fix is scoped to the crash only: the shared
  64-node quadrature can still return a **badly wrong** (not merely
  imprecise) split when the resolved window is wide — for a genuinely
  heavy-tailed or high-variance cause the 64-node answer can be off by
  100% (including the wrong cause winning), while still looking plausible
  (finite, in `[0, 1]`, summing to `<= 1`); see #294, filed from this PR's
  review, for the tracked accuracy gap and worked examples.
- **breaking:** removed the `ComposedDistributionsFlexiChainsExt` weakdep
  extension and its `chain_to_params`/`param_draws`/`strip_prefix`/
  `update(template, chain)` surface (#221). DistributionsInference.jl already
  hosts a generic, tested replacement (`readback`/`readback_draws`, built on
  its own fit-protocol extension) that round-trips a composed tree — pooled,
  shared-tag, or Dirichlet-`branch_probs` — through a real chain with no
  ComposedDistributions-specific code; this package carrying its own 388-line
  parallel tree-walk duplicated that machinery rather than adding anything.
  Drops the `FlexiChains` and `DynamicPPL` weakdeps entirely (neither has any
  other user left in this package). Use `DistributionsInference.readback`/
  `readback_draws` instead; see the [fitting guide](@ref inference).
- **breaking:** `update` is now `public`, not `export`ed (#221). Several
  ecosystem packages (and plenty outside it) have their own `update`-shaped
  verb; exporting a name this generic risked the same ambiguous-binding
  clash #233 hit with `as_turing` when two packages both export a same-named
  generic function. Reach it as `ComposedDistributions.update` or with
  `using ComposedDistributions: update`.
- **fix:** the `ComposedDistributionsMooncakeExt` `xlogy`/`xlog1py` import now
  uses `Mooncake.@from_chainrules` (both AD directions) rather than
  `@from_rrule` (reverse only), so Mooncake forward mode no longer derives a
  silently wrong (zero) gradient at a Gamma-family shape landing exactly on
  `1.0` (#214). A new `test/ad/scenarios.jl` scenario exercises `shape == 1.0`
  under Mooncake forward and reverse against a ForwardDiff reference; the import
  stays in place pending an upstream Mooncake rule (see #99).
- **test:** three new AD gradient scenarios close coverage gaps in the
  `ADFixtures` registry (`test/ADFixtures/src/ADFixtures.jl`): a
  `Shared`-tagged uncertain leaf driven through the full `logdensity` codec
  (tag-dedup's reverse-mode gradient accumulation was untested), a
  `Truncated`-wrapped uncertain leaf through the #216 leaf-wrapper registry's
  codec path, and a `Censored` leaf marginal (#215 landed with value-level
  tests only for both wrappers). Also ledgers a real, currently-CI-red
  regression: Enzyme reverse has crashed with an internal LLVM compiler
  error (`EnzymeInternalError` in `nodecayed_phis!`) on the "Uncertain-leaf
  logdensity codec" scenario since #190, reproducing on every CI run but not
  locally — declared broken in `backend_broken_scenarios()` so CI reflects
  the real state instead of failing outright (#223).
- **docs:** corrected `docs/benchmarks.md`'s claim that the performance-
  history timeline updates "on every push to `main` and on tagged releases"
  (#231) — `benchmark-history.yaml` is currently parked to
  `workflow_dispatch`-only pending #41 (an unregistered, chained dependency
  the kit's benchmark scratch-registry bootstrap does not yet resolve), so
  the automatic trigger the page described is not actually running.
  `docs/benchmarks_notes.md` (the file that exists for exactly this kind of
  note) now explains the parked trigger and links #41, so a reader seeing
  "not enough comparable revisions to compute ratios yet" understands why
  rather than concluding the page is broken.
- **breaking:** removed the `ComposedDistributionsLogDensityProblemsExt`
  weakdep extension and the `as_turing`/`ComposedDistributionsDynamicPPLExt`
  surface (#220, #233). Both duplicated machinery DistributionsInference.jl
  already provides generically over any fit-protocol object (`as_logdensity`
  implementing `LogDensityProblems` directly; `as_turing` building the
  DynamicPPL model), via its `ComposedDistributions` fit-protocol extension —
  `export as_turing` also collided with DistributionsInference's own exported
  `as_turing` when both packages were loaded together. Use
  `DistributionsInference.as_logdensity`/`as_turing` instead; this package's
  own Turing-free `ComposedLogDensity`/`as_logdensity`/`logdensity` core is
  unchanged (still `public`, not exported) and is what DistributionsInference
  builds on.

- **breaking:** stopped re-exporting ConvolvedDistributions' surface
  (`convolved`, `product`/`Product`, `discretise_pmf`, `DelayPMF`,
  `AnalyticalSolver`, `NumericSolver`, `GaussLegendre`, `AbstractSolverMethod`,
  `integrate`, `gl_integrate`; #228) — a caller now reaches these with its own
  `using ConvolvedDistributions`. `convolve_series`/`difference`/`Convolved`/
  `Difference` stay reachable: this package extends/constructs them itself
  for composed tree types.

- **breaking:** `convolve_series(chain, series)` (and the `Resolve`/`Compete`
  marginal form) no longer discretises a continuous observed delay for you
  with an implicit interval-censored-secondary scheme; it collapses the tree
  and delegates straight to `ConvolvedDistributions.convolve_series`, which
  throws for a continuous delay and asks you to discretise first (#226) — the
  same contract a bare distribution already has under ConvolvedDistributions
  0.2. Discretise explicitly with `ConvolvedDistributions.discretise_pmf(delay,
  maxlag)` (or a CensoredDistributions.jl double-interval-censored PMF for a
  day-binned primary) and pass the result to `convolve_series(pmf, series)`.
  The `interval` keyword is dropped from the chain-argument methods to match.
  A replacement composed-chain convenience is tracked as a
  CensoredDistributions.jl extension
  (EpiAware/CensoredDistributions.jl#886) rather than carried here, since the
  discretisation scheme is a censoring choice.

- **feat:** `register_leaf_wrapper!` is a new public hook so a leaf-wrapper
  package extension (ModifiedDistributions' `Affine`/`Weighted`/`Transformed`/
  `Modified`) can tell the generated flat-vector codec (`flat_dimension`,
  `unflatten`, `flatten`, `reconstruct`) how to peel its wrapper types and
  what extra parameters they own, without adding a direct dispatch method to
  `_leaf_free_type`/`_extra_names_of` — the codec's `@generated` functions
  cannot reliably see such a method if it is added after the generator has
  already compiled, a Julia `@generated`-function semantics gap confirmed by
  direct experiment (#188/#189). The hook takes plain data (a type-parameter
  index and a fixed extra-names tuple), never a callable, since even a stored
  closure hits the same world-age wall when called from a generator. A core
  (in-module) leaf wrapper (`Truncated`, `Distributions.Censored`) keeps its
  own direct-dispatch method but now routes its recursion through the same
  registry-aware resolver, so a core wrapper placed directly around a
  registered extension leaf (e.g. `truncated(thin(Gamma(...)))`) peels
  correctly too, not just the reverse nesting.
- **chore:** removed the `design/` folder (#224) — its one note, the
  time-/covariate-varying design rationale, is superseded by the landed
  `Varying`/`instantiate` implementation and its docstrings and the
  [Time-, strata-, and covariate-varying distributions](@ref
  varying-distributions) guide; dangling pointers to the removed file in
  `src/ComposedDistributions.jl`, `src/composers/varying.jl`, and that guide
  are cleaned up alongside it.
- **chore:** renamed `src/composers/hazard_one_of.jl` to
  `src/composers/Compete.jl` (#230), matching the file-per-type convention
  the other composers already follow (`Resolve.jl`, `Sequential.jl`, ...);
  no code changes, only the file name and its in-comment references.
- **chore:** renamed the public `_param_names`/`_leaf_ctor` leaf-protocol
  hooks to `param_names`/`leaf_ctor` (#229) — a leading underscore reads as
  "internal" in Julia, the opposite of what `public.jl` was declaring, and
  `docs/src/developer/leaf-protocol.md` already documented the clean names.
  The old underscored names remain as `const` aliases (the same function
  object, following the existing `uncertain_specs`/`_uncertain_specs`
  pattern), so an existing override such as
  `ComposedDistributions._param_names(::MyLeaf) = ...` keeps working.

- **test:** added a guard against `params_table`/codec ordering drift (#192,
  the #190 review follow-up): the runtime `params_table` walk and the
  generated `unflatten`/`flatten` codec are two independent, hand-maintained
  implementations of the same walk-order and dedup rule, and
  `ext/ComposedDistributionsDynamicPPLExt.jl` assumes their orderings
  coincide with nothing checking it. A new consistency test
  (`test/composers/codec_consistency.jl`) covers every composer/leaf shape;
  a new `as_turing`/NUTS round trip on a `shared(...)`-tagged tree
  (`test/composers/turing_ext.jl`) covers the one path that actually depends
  on the coupling. Full unification into one shared walker was assessed and
  set aside as disproportionate at this point (see the test file's header for
  the reasoning); `_param_names`/`_param_names_of`, the other duplicated
  piece the review flagged, turned out to serve genuinely different
  purposes (a public, instance-dispatched extension point vs. an internal,
  type-dispatched table the generated codec needs) and are not safely
  mergeable, so they get a direct comparison test instead.

- **breaking:** the `ComposedDistributionsLoweredDistributionsExt` extension
  and the `LoweredDistributions` weakdep are removed (LD#51, the #22
  hub-owned decision). `LoweredDistributions` now hosts the `lower` bridge
  for the composer types itself, in its own
  `LoweredDistributionsComposedDistributionsExt` (moved verbatim from this
  package). Anyone who imported the extension module directly from this
  package must load `LoweredDistributions` and rely on its extension
  instead; functionality is otherwise unchanged when both packages are
  loaded together. This is the last source-bridge weakdep this package
  carries — the remaining `Bijectors`/`DynamicPPL`/`FlexiChains`/
  `LogDensityProblems`/`Mooncake` inference extensions are separately
  staged to move out to DistributionsInference.jl (#185).

- **chore:** dropped the root `[sources]` pins for `ConvolvedDistributions`
  and `EpiAwareADTools` now that both are registered in General
  (ConvolvedDistributions 0.2.0, EpiAwareADTools 0.1.0) — both root
  dependencies resolve from the registry again, so this package is
  registrable (#41).
- **feat:** `reconstruct(d, x)` is a new generated primary for the flat-vector
  codec — a composed distribution rebuilt straight from an estimated flat
  vector in one call, with no intermediate `Dict`-typed accumulation (#178,
  PR 2 of the type-domain codec design). `unflatten`/`flatten`/`flat_dimension`
  are now themselves `@generated`: each walks the tree's TYPE once per
  distinct shape (names, tags and groups are compile-time since PR 1) and
  emits the slot indices as literals, producing a concretely-typed
  (`@inferred`-stable) nested `NamedTuple` instead of the old runtime
  `Dict{Symbol, Any}` walk. `logdensity` now routes through the generated
  `unflatten`, and this **fixes #162**: Enzyme reverse could not compile the
  old walk's type-unstable, heap-boxed reconstruction, and now differentiates
  it like every other backend. Measured on four baseline tree shapes (a plain
  chain, a tag shared across branches, a non-centred pooling group, and a
  Resolve stick-breaking node), the generated codec is 4.7-11.9x faster and
  allocates 4-14x less than the two-step `update(d, unflatten(d, x))` it
  replaces on the hot path; first-call compile latency for a fresh
  pooled+Resolve tree shape is unchanged (~0.6-0.7s, dominated by
  `Distributions`/`Dirichlet` compilation, not the generated function itself).
  The modifier-leaf codec path (`ModifiedDistributions`' `Affine`/`Weighted`/
  `Transformed`/`Modified`, e.g. `thin(...)`) is deferred to PR 4 (#189): a
  `@generated` function's generator can compile against a world snapshot
  taken before a later-loaded package extension adds its type-level leaf-
  protocol methods, a genuine Julia semantics gap around generated functions
  and extensions, not a bug in this PR's scope.

- **breaking:** the `ComposedDistributionsModifiedDistributionsExt` reverse
  extension and the `ModifiedDistributions` weakdep are removed (#170 step 2),
  ending the extension cycle between the two packages (Julia 1.12 fails on
  cross-module method overwrites when both packages' extensions activate
  together). ModifiedDistributions' own
  `ModifiedDistributionsComposedDistributionsExt` now hosts the full leaf
  protocol for its modifier leaves (`Affine` / `Weighted` / `Transformed` /
  `Modified`), reading it through the leaf-protocol public API published in
  #174. Anyone who imported the extension module directly from this package
  must load ModifiedDistributions and rely on its extension instead;
  functionality is otherwise unchanged when both packages are loaded together.

- **refactor!:** the composer/wrapper structs carry their layout-affecting
  names and tags in TYPE parameters rather than runtime fields, lifting
  `Sequential`/`Parallel`/`Choose`/`Resolve`/`Compete`'s outcome/step/branch
  names, `Shared`'s tag, and `Pool`'s group/non-centred flag into the type
  domain (part of #178, PR 1 of the type-domain codec design). The public
  constructors (`Sequential(components, names)`, `shared(tag, dist)`,
  `pool(group, population; noncentred)`, ...) are UNCHANGED, and every
  consumer that already read through the accessors
  (`component_names`/`shared_tag`, plus the new `pool_group`/
  `pool_noncentred`) needs no change. This is breaking only for code that
  constructed these structs directly and read the moved fields by name (e.g.
  `d.names`, `shared_leaf.tag`, `pool_spec.group`); such code should switch to
  the accessors. This groundwork is a step towards a generated, allocation-free
  flat-vector codec that fixes #162.

- **fix:** `as_logdensity` and the chain readback (`chain_to_params` /
  `update(template, chain)` / `param_draws`) now reject a tree where a
  `pool` group, a `shared` tag, and a root-level edge name are not
  disjoint (#177). All three land in the same root-lifted NamedTuple
  namespace at readback, so a same-named pair previously clobbered each
  other silently instead of erroring; the collision is caught once at
  construction or readback entry, not on the gradient hot path. Reusing
  one tag to tie a parameter across branches, or one group across several
  pooled members, is unaffected.

- **feat:** `update(d, x::AbstractVector)` is a flat-vector shorthand for
  `update(d, unflatten(d, x))`, collapsing an uncertain tree at a sampler draw
  in a single call (#178).

- **feat:** the composer leaf protocol is now published as considered public API
  (#170). The downstream contract a leaf-wrapper package implements is
  de-underscored and documented as the stable surface: `uncertain_specs`,
  `shared_tag`, `leaf_param_names`, `leaf_mean`, `leaf_var` and
  `leaf_detail_lines` join the already-public `free_leaf` / `rewrap_leaf` /
  `component_names` / `param_names` / `leaf_ctor` (the underscored spellings
  stay as internal aliases, so nothing breaks). The scalar `thin`-factor hook is
  generalised to a map-based `extra_leaf_params` / `set_extra_leaf_params`
  protocol, each extra parameter carrying its own value and support, which also
  fixes a latent `BoundsError` when a leaf carries an extra parameter alongside
  an uncertain native one. A new developer page documents the protocol. This
  release is non-breaking; the removal of the ModifiedDistributions reverse
  extension and its weakdep (ending the MD-CD extension cycle) follows in a
  later release.

- **fix:** `minimum` and `maximum` on a `Parallel` now return a per-branch
  NamedTuple of support bounds, matching how `mean` / `var` report per-endpoint
  moments, and the other composed types raise a clear `ArgumentError` naming the
  generic instead of the opaque `MethodError` about `iterate`. Also tightens the
  `Parallel` `event_names` test so it genuinely asserts the event tuple.

- **feat:** `@events` declares an event-tree TOPOLOGY as a readable operator
  diagram, structure only with no distributions attached (#156). `→` (`\to`)
  chains events into a `Sequential`, `|` branches into a one_of outcome, `&`
  runs branches in `Parallel`, and parentheses group for precedence; a bare
  identifier is an event name and becomes a named hole. `update(skeleton;
  name = dist, ...)` fills the holes and builds the concrete composed tree
  through the existing verbs, so one delay topology is reused across pathogens
  or settings. Whether a `|` node becomes a fixed-probability `Resolve` or a
  racing-hazard `Compete` is decided at fill time by the fill value type
  (`(dist, prob)` tuples versus bare distributions, the last branch alone free
  to take the residual), so `|` stays one syntax. A fill value is any valid
  leaf, including an `uncertain` / `@uncertain` leaf or a ModifiedDistributions
  modifier leaf, which composes through the existing extension with no
  MD-specific code in `@events`. The fill validates that every hole is filled
  and no unknown key is passed.

- **docs:** a "Fitting a composed distribution" guide walks the inference
  tooling in one place: the `as_logdensity` log-density over a tree's estimated
  parameters, sampling it through the `LogDensityProblems` interface with
  LogDensityProblemsAD, sampling with Turing through `as_turing`, and reading a
  fitted chain back onto the tree with `chain_to_params` / `update`.

- **feat:** `@uncertain expr` reads a distribution-valued constructor argument
  as that parameter's prior, so an uncertain composed tree reads naturally
  (#155). It rewrites syntax only, leaving the type sorting to the runtime
  positional `uncertain(D, args...)` method: each call `D(pos_args...)` whose
  head is a distribution type and one of whose positional arguments is a
  distribution literal becomes `uncertain(D, pos_args...)`, so
  `@uncertain Gamma(Normal(0.7, 0.2), 1.0)` makes `shape` uncertain and fixes
  `scale`. The walk is recursive, so it composes with `compose`, the verbs and
  the ModifiedDistributions wrappers (a modifier wraps the rewritten uncertain
  leaf); an all-literal constructor and a keyword-carrying or qualified call are
  left unchanged. Pure `Expr` rewriting, so no new dependency.

- **test:** extend the AD gradient scenarios (`test/ADFixtures`) across the
  ForwardDiff / ReverseDiff / Enzyme / Mooncake matrix. A new `:latent`
  category differentiates the full `as_logdensity`/`logdensity` codec over an
  uncertain-leaf tree (the flat-vector to nested-NamedTuple `unflatten`/
  `update` path) and a centred pool (the `_pool_centred_logprior` population
  term), and the `:marginal` group gains a `Choose` scored at a selected
  alternative. The centred pool and the `Choose` differentiate on all four
  backends; the uncertain-leaf codec differentiates on ForwardDiff, ReverseDiff
  and Mooncake reverse but is marked broken on Enzyme reverse, which cannot
  compile its mixed fixed/active heap reconstruction (an opaque-pointer LLVM
  error).

- **feat:** `as_turing(dist, data; prefix, loglik)` builds a `DynamicPPL`
  model over a composed distribution's estimated parameters, so a composed
  posterior is sampleable with `sample(as_turing(dist, data), NUTS(), ...)`
  (#9). It is a light wrapper on the `as_logdensity` codec. Each estimated
  parameter is a named `~` site drawn from its own prior and the data
  likelihood is added with `@addlogprob!` from the codec's reconstruction, so
  the model's total log-density equals `logdensity(as_logdensity(dist, data),
  x)` by construction. The `~` site names match the inference readback exactly
  (`d.onset_admit.shape`, an uncertain node's `d.<edge>.branch_probs.stick_k`,
  a shared leaf once under its tag), so a fitted chain reads back through
  `chain_to_params` / `update(dist, chain)` unchanged. Supported rows carry a
  concrete prior (ordinary uncertain leaves and stick-breaking branch
  probabilities); a pooled tree is rejected for now (a centred pool has no
  fixed prior, and the readback does not yet consume a pooled chain), with a
  pointer to the `as_logdensity` + LogDensityProblemsAD path. The model lives
  in a new `ComposedDistributionsDynamicPPLExt` extension triggered by
  `DynamicPPL` alone.

- **feat:** a `LogDensityProblems` weak-dependency extension exposes a
  `ComposedLogDensity` (from `as_logdensity`) as a standard
  `LogDensityProblems` problem, so a composed distribution's posterior over its
  estimated parameters is sampleable by any LogDensityProblems consumer
  (AdvancedHMC, DynamicHMC, Pathfinder, Turing's `externalsampler`) with
  gradients supplied by LogDensityProblemsAD (#13). The extension implements
  `dimension` (the estimated flat-parameter count), `logdensity` (the codec's
  evaluator), and the zeroth-order `capabilities`. This is the Turing-free
  inference substrate that complements the DynamicPPL path, and it needs no new
  hard dependency.

- **feat:** a `LoweredDistributions` weak-dependency extension lowers a
  composed distribution to a backend-agnostic dynamical-systems
  representation, so `lower(compose(...))` yields a phase-type or
  continuous-time Markov chain for the whole delay structure that a Catalyst /
  ODE / Petri / Jump backend can consume (#149). The scalar composers lower
  exactly, since composition of phase-types is closed. `Sequential` convolves
  its steps into a series phase-type, `Resolve` mixes its outcomes into a
  hyper-phase-type weighted by the branch probabilities, `Compete` races its
  causes through the competing-risks Kronecker sum, and `Shared` lowers its
  wrapped leaf. The vector composers lower to a joint `CTMC`, `Parallel`
  through the Kronecker sum of its independent branches and `Choose` through
  the block-diagonal union of its selector alternatives. Nesting a `Parallel`
  or `Choose` inside a scalar composer raises a clear error rather than a
  silent misrepresentation.

- **fix:** `logdensity`/`unflatten` (`src/composers/logdensity.jl`) now
  differentiate under Mooncake, both forward and reverse. `unflatten` calls
  the `Symbol` path-splitter `_split_edge` unconditionally on every row, and
  the codec's length guards build their `DimensionMismatch` message by
  interpolating the tree object; both recurse into Base's UTF-8
  string-indexing continuation machinery, for which Mooncake's whole-program
  rule derivation has no rule (a `sub_ptr` intrinsic hit), on every call for
  `_split_edge`, and on any reachable branch for the guards' messages,
  regardless of whether it is taken. `_split_edge` and the four
  `DimensionMismatch`-throwing call sites (`logdensity.jl`,
  `named_outputs.jl`, `Parallel.jl`, `Sequential.jl`) are shielded from
  Mooncake with `@zero_derivative` in `ComposedDistributionsMooncakeExt`
  (#146).


- **feat:** `compose` gains a varargs-pairs spelling,
  `compose(:a => d1, :b => d2, ...)` (#145), a thin convenience over the
  primary `NamedTuple` form so call sites migrating from
  CensoredDistributions' pairs-based `compose` keep working unmodified. See
  the FAQ for the migration note.

- **feat:** `_uncertain_specs` and `_leaf_detail_lines` are now `public`
  (#142), sanctioning the leaf-introspection recursion a leaf-wrapper package
  extends alongside `free_leaf`/`rewrap_leaf`. Without extending
  `_uncertain_specs`, an uncertain prior attached to a wrapped leaf was
  silently dropped by `build_priors`; without `_leaf_detail_lines`, a wrapped
  leaf's `inspect` detail fell back to its raw struct dump.

- **test:** pin three seams now that ConvolvedDistributions 0.2 is adopted: a
  Modified-wrapped chain step (affine / weight / thin) lowering through
  `observed_distribution` / `convolve_series` (#117); the recurrent-operator
  seam end to end (a time-varying chain resolved per step, collapsed,
  discretised, and driving the vector convolution surface, plus one hand-rolled
  renewal step, refs #82); and the restored real `cdf` assertion on a
  `difference` of two chains that the #122 workaround stood in for while
  ConvolvedDistributions #45 was open (#137).

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

- **feat:** re-export ConvolvedDistributions 0.2's Mellin product family (the
  `product` constructor for `Z = X * Y`), so the convolution surface is complete
  through ComposedDistributions alone. The `product` constructor is exported;
  the `Product` type stays unexported (a bare `Product` would clash with
  Distributions' deprecated `Product`) but is public and reachable as
  `ComposedDistributions.Product`, mirroring ConvolvedDistributions. Composing a
  `Product` leaf into a tree is not yet wired (#139).

- **fix:** `probs` / `occurrence_probability` on a racing-hazard (`Compete`)
  node no longer return winning probabilities that sum slightly above one. The
  per-cause split is mathematically sub-stochastic (sums to `1 - ∏ S_k(∞) ≤ 1`),
  but Gauss-Legendre quadrature could overshoot to e.g. 1.0000322 for proper
  causes; the split is now rescaled to a valid probability vector when it
  exceeds one, leaving a genuine sub-one defective deficit intact (#115).

- **refactor:** the racing-hazard (`Compete`) moment, winning-probability and
  cause-cdf quadratures now call the public
  `integrate(::GaussLegendre, f, lo, hi)` rather than reaching into
  ConvolvedDistributions' internal `GaussLegendre(; n).rule` and `gl_integrate`.
  Results are identical (the same fixed 64-node rule); this drops the coupling
  to an unexported upstream type that could change without a breaking bump
  (#109).

- **refactor:** renamed the internal `src/composers/intervene.jl` to
  `structural_edits.jl`, naming the `update` / `prune` / `splice` verbs it holds
  (the `intervene` verb is gone from the public API). No user-facing change
  (#114).

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

- **Behaviour change:** the standalone `(name, time)` view of a one_of draw is
  now a keyword on `rand`, `rand(node; outcome = true)`, rather than the
  separate public `rand_outcome` verb (now removed). A keyword-free `rand(node)`
  is unchanged (it still returns the full named event record), so only call
  sites that reached for the compact pair need updating from
  `ComposedDistributions.rand_outcome(rng, node)` to
  `rand(rng, node; outcome = true)`.

- **Breaking:** `rand` of a standalone `Resolve` or `Compete` node now returns
  the named event record of the outcome that fired — a `NamedTuple` keyed by
  `event_names(node)` (a positional origin slot then one slot per outcome, the
  fired outcome's time present and the others `missing`) — instead of the scalar
  marginal time-to-resolution (#96, syncing to CensoredDistributions' #639). The
  record names which outcome occurred, so `logpdf(node, rand(node))` round-trips
  and identifies the outcome. To recover the old scalar draw, sample the
  marginal `rand(as_mixture(node))`; the `(outcome, time)` pair view is
  `rand(node; outcome = true)`. A one_of node nested inside a `Sequential` /
  `Parallel` is
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
