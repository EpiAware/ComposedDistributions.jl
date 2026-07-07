# # [Multi-strata trees and parameter uncertainty](@id strata-uncertainty)
#
# ## Introduction
#
# A composed tree is stationary by default: every leaf is a fixed distribution.
# Real delays are often non-stationary — a delay shortens over a wave, or differs
# by region — and some parameters are not known but estimated.
# ComposedDistributions handles both by generalising the leaf: a [`varying`](@ref)
# leaf reads an observed covariate, and an [`uncertain`](@ref) leaf carries
# distribution-valued parameters.
#
# This tutorial builds a natural-history tree that varies by region and calendar
# time, resolves it per stratum, ties a delay across strata, and then makes a
# parameter uncertain so it can be estimated.
# It builds on [Composing distributions](@ref composing-distributions) and the
# [time-, strata-, and covariate-varying reference](@ref varying-distributions).

using ComposedDistributions
using Distributions
using Random

# ## Varying by region and time
#
# We build an onset-to-admission delay that grows with calendar time and an
# admission-to-death delay that differs by region.
# Both are ordinary leaves, so they drop into [`compose`](@ref) unchanged: a
# time-varying leaf reads the default `:time` covariate, and a region-varying leaf
# names its covariate and gives a `reference` for when none is supplied.

onset_admit = varying(t -> Gamma(2.0, 1.0 + 0.02t))

admit_death = varying(
    r -> r === :north ? LogNormal(0.5, 0.4) : LogNormal(0.8, 0.3);
    covariate = :region, reference = LogNormal(0.5, 0.4))

template = compose((onset_admit = onset_admit, admit_death = admit_death))

# The template still carries varying leaves, so it is not yet ready to score.

(template = has_varying(template),)

# ## Resolving per stratum
#
# [`instantiate`](@ref) resolves a whole tree against a [`Context`](@ref).
# We combine an observed time and region with [`with_covariates`](@ref) and
# resolve one concrete tree per stratum.

north = instantiate(template,
    with_covariates(Context(region = :north); time = 5.0))

south = instantiate(template,
    with_covariates(Context(region = :south); time = 5.0))

# Each resolved tree is concrete and ready to score.

(north = has_varying(north), south = has_varying(south))

# The region-varying admission delay differs between the two strata.

(north = event(north, :admit_death), south = event(south, :admit_death))

# ## Sharing a parameter across strata
#
# Some parameters are common to every stratum, such as a reporting delay recorded
# the same way everywhere.
# [`shared`](@ref) tags a leaf so its occurrences are one free parameter, and the
# tag survives [`instantiate`](@ref), so the leaf is identical in every resolved
# stratum.

report = shared(:report, Gamma(1.5, 1.0))

tied_template = compose((onset_admit = onset_admit,
    admit_death = admit_death, onset_report = report))

north_tied = instantiate(tied_template,
    with_covariates(Context(region = :north); time = 5.0))

south_tied = instantiate(tied_template,
    with_covariates(Context(region = :south); time = 5.0))

# The tied reporting delay is the same leaf in both strata, while the admission
# delay still differs.

(north_report = event(north_tied, :onset_report),
    south_report = event(south_tied, :onset_report))

# [`params_table`](@ref) inventories the tied leaf once, under its tag, rather
# than once per stratum.

unique(params_table(north_tied).edge)

# ## Adding parameter uncertainty
#
# A stratum resolves the observed covariates, but some parameters are still
# unknown and estimated.
# An [`uncertain`](@ref) leaf declares that directly: a parameter given a
# distribution is drawn from it rather than fixed, and that distribution is the
# parameter's prior.
# Here the admission-to-death location `mu` is uncertain.

est_template = compose((
    onset_admit = varying(t -> Gamma(2.0, 1.0 + 0.02t)),
    admit_death = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2))))

# Resolving the stratum leaves the uncertain leaf in place: `instantiate` fills
# the observed covariates, and the uncertain parameter is resolved later by a fit.

resolved = instantiate(est_template, Context(time = 5.0))

(has_varying = has_varying(resolved), has_uncertain = has_uncertain(resolved))

# [`params_table`](@ref) carries the uncertain parameter's prior on its `prior`
# column, so [`build_priors`](@ref) picks it up with no separate override.

tbl = params_table(resolved)
(edge = tbl.edge, param = tbl.param, prior = tbl.prior)

# Only `rand` reports the marginal: it draws the uncertain parameter from its
# prior, rebuilds the leaf, then draws the record.

rand(Xoshiro(1), resolved)

# Every other query — `logpdf`, `mean`, and the rest — silently uses the template
# value while a leaf is still uncertain, so guard a scoring or fitting loop with
# [`has_uncertain`](@ref).
# [`update`](@ref) collapses the uncertain leaf to a concrete one, and the guard
# then passes.

fitted = update(resolved, (onset_admit = (shape = 2.0, scale = 1.1),
    admit_death = (mu = 0.6, sigma = 0.4)))

(before = has_uncertain(resolved), after = has_uncertain(fitted))

# The collapsed tree is fully concrete, so `logpdf` scores a record.

logpdf(fitted, rand(Xoshiro(2), fitted))

# ## Uncertain-first estimation
#
# The `uncertain` surface is the direct way to say "this parameter is estimated,
# with this prior": the spec is the prior and the declaration in one, and every
# parameter without a spec stays fixed.
# The estimation layer keys off exactly these specs.
# [`flatten`](@ref) / [`unflatten`](@ref) / `flat_dimension` and `as_logdensity`
# target the spec'd parameters only, so a tree with no uncertain leaves estimates
# nothing (a pure likelihood at the fixed tree), and the flat table is a derived
# view.

flat = ComposedDistributions.flat_dimension(resolved)
(estimated_parameters = flat,)

# [`update`](@ref) is the verb that moves the estimation boundary. A distribution
# in a parameter slot makes just that parameter uncertain (a partial update);
# `update(tree, param_priors(tree))` promotes every free parameter to uncertain
# with support-derived default priors, the explicit estimate-everything path.

promoted = update(resolved, param_priors(resolved))
(before = ComposedDistributions.flat_dimension(resolved),
    after = ComposedDistributions.flat_dimension(promoted))

# Partial pooling across strata — estimating region-specific parameters that
# shrink towards a shared mean, rather than the fully independent (per-stratum) or
# fully tied ([`shared`](@ref)) extremes shown here — is designed in issue #23 and
# is not yet a built verb.
# For now a parameter is either independent per stratum or tied across all strata.
#
# ## Summary
#
# - A [`varying`](@ref) leaf reads an observed covariate; [`instantiate`](@ref)
#   with a [`Context`](@ref) resolves a whole tree per stratum.
# - [`with_covariates`](@ref) threads several covariates (time and region)
#   together; [`has_varying`](@ref) guards a tree that is not yet resolved.
# - [`shared`](@ref) ties a parameter across strata, and the tie survives
#   `instantiate`.
# - An [`uncertain`](@ref) leaf carries a parameter's prior; [`params_table`](@ref)
#   rides it on the `prior` column, `rand` draws the marginal, and
#   [`update`](@ref) collapses it to a concrete leaf, guarded by
#   [`has_uncertain`](@ref).
#
# ## Where next
#
# - The [time-, strata-, and covariate-varying reference](@ref varying-distributions)
#   covers node-level variation and the renewal-kernel use.
# - [Composing distributions](@ref composing-distributions) is the full verb
#   walkthrough.
