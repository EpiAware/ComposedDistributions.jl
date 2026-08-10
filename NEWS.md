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
  fix; define `param_names` explicitly for such a type. `default_prior`'s
  support-derived defaults shift for a handful of families as a consequence
  (Cauchy `sigma`, Laplace/Logistic `theta`, TDist `nu` move to a
  positive-truncated default; InverseGaussian `mu`, SkewNormal `alpha` and
  NormalInverseGaussian `beta` are now misclassified by the unchanged
  `_is_positive_param`/`_is_location_param` heuristic) — tracked as a
  follow-up, not re-engineered here (#372).
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

GitHub releases (auto-generated from merged PRs) cover every release.
