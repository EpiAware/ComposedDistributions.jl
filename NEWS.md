## Unreleased

- **breaking:** `param_names`/`leaf_param_names` derive a leaf's parameter
  names from the leaf's own type (the first `N` fieldnames of `typeof(leaf)`,
  transliterated, where `N` is the arity of `params(leaf)`) instead of a
  curated table of six families. Any Distributions.jl-conforming leaf whose
  fields line up 1:1 with its `params`, in order, now gets real names with no
  method of its own; a leaf whose fields do not line up falls back to
  `:param_1, :param_2, ...` as before. This moves 57 families off positional
  names and renames four: Gamma/Weibull `(:shape, :scale)` →
  `(:alpha, :theta)`, Exponential `(:scale,)` → `(:theta,)`, Uniform
  `(:lower, :upper)` → `(:a, :b)`. Normal/LogNormal are unchanged. Every
  dotted flat coordinate keyed on the old names moves with them (e.g.
  `onset_admit.shape` → `onset_admit.alpha`, a pooled stratum's
  `<stratum>.shape.z` → `<stratum>.alpha.z`). **A chain or posterior stored
  against a 0.1.x template cannot be read back onto 0.2.0.** A leaf type
  whose fields do not line up with its `params` in order now either derives
  wrong labels (a type-only rule cannot see a field/params order mismatch on
  its own) or, if it reaches `uncertain(...)`, is rejected there with an
  error naming `ComposedDistributions.param_names` and `leaf_ctor` as the
  fix; define `param_names` explicitly for such a type (#372, #377).
- **feature:** the composer-node extension contract is minimal and public
  now (`public`, not exported).
  A downstream node implements `node_children(node)`, `node_rebuild(node,
  children)` and `component_names(node)` (promoted from the private
  `_node_children`/`_rebuild`), and that alone is enough to compose, table
  (`composed_to_table`) and `update`.
  `child_nleaves`/`child_logpdf`/`child_rand!` now default generically off
  `node_children` for a plain "concatenating" node, so a node whose
  realisation is just its children laid end to end (`Sequential`/`Parallel`'s
  own semantics) needs none of the three; a node with different combination
  semantics still overrides them directly, as `Choose` does.
  A node additionally wanting the flat-vector codec
  (`flat_dimension`/`flatten`/`unflatten`/`reconstruct`, the primitive a fit
  routes through) declares its own child names and children's types as its
  first two type parameters, matching `Sequential`/`Parallel`/`Choose`/
  `Compete`'s existing shape — a `@generated` function cannot call a method a
  downstream package defines (a Julia world-age hazard), so this is the only
  design that lets the codec read a downstream node's layout at compile time.
  `composed_to_table`, `params`, `update`, `has_uncertain`, `has_varying` and
  `compose(...)` nesting are now generic over any `AbstractComposedDistribution`
  subtype rather than closed to the five built-in node kinds; a node
  subtyping `AbstractComposedDistribution` without the required methods now
  fails with a clear error naming the missing method or the exact
  type-parameter shape needed, rather than silently being treated as a leaf
  with zero estimated parameters (#374).
  `TestUtils.test_estimation_dimension` is a new conformance check (wired
  into both `test_node_interface` and `test_interface`) asserting
  `flat_dimension(d)` matches `composed_to_table(d)`'s estimated-row count,
  so a node that silently drops an uncertain leaf from the codec fails the
  harness rather than shipping green.
  A downstream node's own `event_names`/`event_tree`/`Base.show`, and
  top-level `rand`/`logpdf`/moments when the node is used as a standalone
  root distribution, are not yet part of this contract and remain built-in
  only; tracked as a follow-up issue.
- **breaking:** `params_table(d)` is removed. `composed_to_table(d)` is now
  the single table surface: it returns the full node/attribute/parameter
  inventory of a composed tree — one row per composer node, per leaf
  (wrapper) layer, per fixed-structure attribute, and per scalar free
  parameter. The parameter-only view `params_table(d)` used to give is a
  filter over the full table: `filter(row -> row.role == :param,
  Tables.rows(composed_to_table(d)))`. `build_priors` and `update` both
  accept a `composed_to_table`-shaped table directly, filtering to its
  `:param` rows internally, so `build_priors(composed_to_table(tree))` and
  `update(tree, composed_to_table(tree))` work without filtering by hand.
  The wrapper type `composed_to_table` returns is renamed from
  `ParamsTable` to `ComposedTable`. This is a breaking change folded into
  the same unregistered 0.2.0 window as the rest of the #227 table work,
  not a deprecation cycle.
- **feature:** `composed_to_table(d)` is built from one shared pre-order
  walk. A composed distribution is also a Tables.jl source in its own
  right, forwarding to `composed_to_table`, so `DataFrame(tree)` gives the
  full table. `update` filters a role-carrying table to its parameter rows
  first, and `update(a, b)` with `b` a tree now throws a clear error naming
  `composed_to_table` as the explicit way to copy rows, rather than
  silently bulk-writing `b` into `a`. `node_attributes` is the one new public
  hook a downstream node or leaf-wrapper type defines to control its own
  rows, reporting the fixed, non-parameter structure it carries. A row's
  `node` label is read off the type name and a wrapped leaf's layers are
  peeled through `inner_dist`, so neither is asked of a downstream type
  (#227).
- **breaking:** removed `default_prior`, `build_priors` and `param_priors`.
  These guessed a prior family from a parameter's name or its leaf's variate support.
  A parameter's name does not reliably say what family of prior it needs (an `InverseGaussian`'s `mu` is positive, a `GEV`/`SkewNormal`'s `shape` is signed), so the guess mis-classified real families.
  Choosing priors is DistributionsInference.jl's job, not this package's: it describes structure, supports and which parameters are free.
  `uncertain(tree)` (bare) still means "estimate every free parameter".
  It now marks each one with the `no_prior()` marker instead of guessing a prior, including a `Resolve`'s branch-probability simplex (`branch_prob_prior` now also accepts `no_prior()` alongside a `Dirichlet`).
  `no_prior()` is a spec value everywhere a prior or a `pool(...)` spec is accepted (`uncertain`, `update`, `composed_to_table`'s `prior` column).
  Attach a real prior afterwards with a targeted `uncertain(tree; param = prior, ...)` call, or by editing `composed_to_table(tree)`'s `prior` column and calling `update(tree, table)`.
  `rand` on an `Uncertain` leaf still carrying a `no_prior()` marker refuses, naming the parameter, rather than silently drawing from nothing.
  A `Resolve` node's `rand` carries the same guard on its `branch_prob_prior`: it refuses, naming the node, when `branch_prob_prior` is marked `no_prior()`, rather than silently drawing from its current fixed `branch_probs` (#366).

GitHub releases (auto-generated from merged PRs) cover every release.
