## Unreleased

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

GitHub releases (auto-generated from merged PRs) cover every release.
