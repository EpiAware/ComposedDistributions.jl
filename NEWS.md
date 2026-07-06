## Unreleased

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

This file tracks notes for major releases and significant milestones; GitHub
Releases (auto-generated from merged PRs) cover every release in between.
