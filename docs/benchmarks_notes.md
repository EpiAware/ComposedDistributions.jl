<!-- PACKAGE-OWNED — hand-maintained notes on skipped or broken
benchmarks. scaffold writes this once and never overwrites it. The
managed build splices this file verbatim into the generated
`docs/src/benchmarks.md`, right below the overall trend plot and above
the collapsed per-suite detail, under a "Skipped & broken benchmarks"
heading. Note here why a suite is excluded, a scenario is known-broken,
or a benchmark is skipped; delete this comment once you have real
content. Leave the placeholder line below if there is nothing to
report — an empty file (or one that is only this comment) renders
nothing. -->

No benchmark is skipped or known-broken. The performance-history *timeline*
is, however, currently manual rather than automatic: `benchmark-history.yaml`
is parked to `workflow_dispatch`-only until
[ConvolvedDistributions](https://github.com/EpiAware/ConvolvedDistributions.jl)
and [EpiAwareADTools](https://github.com/EpiAware/EpiAwareADTools.jl)
register (see the workflow's own comment and
[#41](https://github.com/EpiAware/ComposedDistributions.jl/issues/41)) — the
kit's benchpkg-side scratch-registry bootstrap does not yet resolve that
unregistered, chained dependency, so a push/tag run fails. Until then the
timeline only gains a point when someone runs the workflow by hand, so the
summary above may show too few revisions to compute a ratio.
