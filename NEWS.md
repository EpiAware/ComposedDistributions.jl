## Unreleased

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
  silently bulk-writing `b` into `a`. `node_kind`, `node_children`,
  `node_attributes` and `leaf_layers` are the new public hooks a downstream
  node/leaf-wrapper type overrides to control its own rows (#227).

GitHub releases (auto-generated from merged PRs) cover every release.
