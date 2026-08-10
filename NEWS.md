## Unreleased

- **breaking:** removed the name/support-based default-prior guessing from
  `build_priors`/`param_priors`, and the `default_prior` function and its
  `_is_location_param`/`_is_positive_param` name heuristics entirely. A
  parameter's name does not reliably say what family of prior it needs (an
  `InverseGaussian`'s `mu` is positive, a `GEV`/`SkewNormal`'s `shape` is
  signed), so guessing from the name mis-classified real families, and a
  correct per-family support table is not something this package maintains.
  `build_priors`/`param_priors` now require a prior for every row — via
  `priors = (...)`, a `default` function, or an attached `uncertain` spec —
  and throw a clear `ArgumentError` naming the missing row otherwise. Bare
  `uncertain(tree)` (promote every free parameter with a guessed prior) is
  removed for the same reason; use targeted `uncertain(tree; param = prior,
  ...)` calls, or `update(tree, param_priors(tree; priors = ..., default =
  ...))` once every row has an explicit prior. Fitting and prior selection
  now live in DistributionsInference.jl, which is the place to look for
  prior-choice guidance.

GitHub releases (auto-generated from merged PRs) cover every release.
