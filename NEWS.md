## Unreleased

- **breaking:** removed `default_prior`, `build_priors` and `param_priors`.
  These guessed a prior family from a parameter's name or its leaf's variate support.
  A parameter's name does not reliably say what family of prior it needs (an `InverseGaussian`'s `mu` is positive, a `GEV`/`SkewNormal`'s `shape` is signed), so the guess mis-classified real families.
  Choosing priors is DistributionsInference.jl's job, not this package's: it describes structure, supports and which parameters are free.
  `uncertain(tree)` (bare) still means "estimate every free parameter".
  It now marks each one with the `no_prior()` marker instead of guessing a prior, including a `Resolve`'s branch-probability simplex (`branch_prob_prior` now also accepts `no_prior()` alongside a `Dirichlet`).
  `no_prior()` is a spec value everywhere a prior or a `pool(...)` spec is accepted (`uncertain`, `update`, `params_table`'s `prior` column).
  Attach a real prior afterwards with a targeted `uncertain(tree; param = prior, ...)` call, or by editing `params_table(tree)`'s `prior` column and calling `update(tree, table)`.
  `rand` on an `Uncertain` leaf still carrying a `no_prior()` marker refuses, naming the parameter, rather than silently drawing from nothing.
  A `Resolve` node's `rand` carries the same guard on its `branch_prob_prior`: it refuses, naming the node, when `branch_prob_prior` is marked `no_prior()`, rather than silently drawing from its current fixed `branch_probs` (#366).

GitHub releases (auto-generated from merged PRs) cover every release.
