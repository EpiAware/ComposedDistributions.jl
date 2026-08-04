## Unreleased

- **feature:** `composed_to_table(d)` returns the full node/attribute/
  parameter inventory of a composed tree — one row per composer node, per
  leaf (wrapper) layer, per fixed-structure attribute, and per scalar free
  parameter — alongside the existing `params_table(d)`, which is now exactly
  the `role == :param` projection of it, from one shared pre-order walk (no
  extra traversal on `params_table`'s existing call sites). A composed
  distribution is also a Tables.jl source in its own right, forwarding to
  `composed_to_table`, so `DataFrame(tree)` gives the full table. `update`
  filters a role-carrying table to its parameter rows first, and
  `update(a, b)` with `b` a tree now throws a clear error naming
  `composed_to_table`/`params_table` as the explicit way to copy rows,
  rather than silently bulk-writing `b` into `a`. `node_kind`,
  `node_children`, `node_attributes` and `leaf_layers` are the new public
  hooks a downstream node/leaf-wrapper type overrides to control its own
  rows (#227).

GitHub releases (auto-generated from merged PRs) cover every release.
