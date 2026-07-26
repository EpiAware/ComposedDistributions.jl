# ComposedDistributions × Bijectors extension: `to_constrained` maps an
# unconstrained flat vector to the constrained ESTIMATED parameters plus a
# log-Jacobian, built per row from each row's prior (or, for a centred-pooled
# row, its population's family). The load-bearing checks are the per-row
# transform against Bijectors itself across every prior family the codec
# produces (an ordinary positive-support prior, a stick-breaking simplex
# `Beta`, a non-centred pooled latent/hyperparameter, a centred-pooled member)
# and the change-of-variables identity a sampler relies on:
# `logdensity(prob, x) + logjac` is the unconstrained-space log-target at `z`.

@testitem "Bijectors extension loads" begin
    using Bijectors

    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsBijectorsExt) !== nothing
end

@testitem "to_constrained: per-row transform across every prior family" begin
    using ComposedDistributions: update
    using Distributions
    using Bijectors: Bijectors, bijector, inverse, with_logabsdet_jacobian
    using ComposedDistributions: as_logdensity, to_constrained, flat_dimension,
                                 CentredPoolPrior, _population_template

    # One tree exercising every row shape the codec's flat layout produces:
    # a positive-support uncertain leaf (Gamma shape, LogNormal prior), a
    # stick-breaking Beta row (a Resolve's Dirichlet-estimated branch
    # probabilities), a non-centred pooled group (a Normal(0, 1) latent per
    # member plus its two hyperparameter rows), and a centred-pooled group
    # (a member scored directly against its population).
    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = update(
            resolve(:death => (Gamma(1.5, 1.0), 0.4), :disch => Gamma(2.0, 1.5)),
            (branch_probs = Dirichlet([2.0, 2.0]),)),
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0);
            shape = pool(:centred_g,
                uncertain(Gamma(2.0, 1.0);
                    shape = truncated(Normal(2.0, 1.0); lower = 0))))))

    zero_lik(d, data) = 0.0
    prob = as_logdensity(tree, nothing; loglik = zero_lik)
    n = flat_dimension(tree)
    @test n == length(prob.flat_priors)

    z = collect(range(-0.8, 0.8; length = n))
    x, logjac = to_constrained(prob, z)
    @test length(x) == n

    # Each row's effective prior: the row's own prior, or (for a
    # centred-pooled row) its population's family template — the bijector
    # depends only on the family/support, not the current hyperparameters.
    _effective_prior(p) = p isa CentredPoolPrior ?
                          _population_template(p.pool.population) : p

    per_row = map(eachindex(z)) do i
        binv = inverse(bijector(_effective_prior(prob.flat_priors[i])))
        with_logabsdet_jacobian(binv, z[i])
    end
    @test x ≈ [xi for (xi, _) in per_row]
    @test logjac ≈ sum(last, per_row)

    # Every constrained value lands in its effective prior's support: the
    # Gamma-shape and centred-pooled rows are positive, the stick coordinate
    # and the non-centred pooled latent land in (0, 1) and the reals
    # respectively.
    for i in eachindex(x)
        @test insupport(_effective_prior(prob.flat_priors[i]), x[i])
    end

    # A length mismatch is rejected eagerly, like the rest of the codec.
    @test_throws DimensionMismatch to_constrained(prob, z[1:(end - 1)])
end

@testitem "to_constrained: closed-form identity for a LogNormal-Gamma row" begin
    using Distributions
    using ComposedDistributions: as_logdensity, to_constrained, logdensity

    # A single uncertain Gamma shape with a LogNormal(mu, sigma) prior: the
    # bijector maps the positive shape through log, so `x = exp(z)` and the
    # log-Jacobian is `z`. Because a LogNormal is exp of a Normal by
    # definition, `logpdf(LogNormal(mu, sigma), exp(z)) + z` collapses to
    # `logpdf(Normal(mu, sigma), z)` exactly — a closed-form oracle
    # independent of the transform machinery itself.
    mu, sigma = 0.4, 0.3
    tree = compose((onset_admit = uncertain(Gamma(2.0, 1.0);
        shape = LogNormal(mu, sigma)),))
    zero_lik(d, data) = 0.0
    prob = as_logdensity(tree, nothing; loglik = zero_lik)

    for z0 in (-0.6, 0.0, 1.1)
        x, logjac = to_constrained(prob, [z0])
        @test x[1] ≈ exp(z0)
        @test logdensity(prob, x) + logjac ≈ logpdf(Normal(mu, sigma), z0)
    end
end

@testitem "to_constrained: logdensity(prob, x) + logjac is the unconstrained target" begin
    using ComposedDistributions: update
    using Distributions
    using Bijectors: Bijectors, transformed
    using ComposedDistributions: as_logdensity, to_constrained, logdensity,
                                 flat_dimension, unflatten, CentredPoolPrior,
                                 _collapse_population, _pool_hyper

    # The same mixed tree as the per-row test, including a centred-pooled
    # row whose scored population depends on the CURRENT draw's
    # hyperparameters (not its static template) — the identity a sampler
    # relies on must hold there too.
    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = update(
            resolve(:death => (Gamma(1.5, 1.0), 0.4), :disch => Gamma(2.0, 1.5)),
            (branch_probs = Dirichlet([2.0, 2.0]),)),
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0);
            shape = pool(:centred_g,
                uncertain(Gamma(2.0, 1.0);
                    shape = truncated(Normal(2.0, 1.0); lower = 0))))))

    zero_lik(d, data) = 0.0
    prob = as_logdensity(tree, nothing; loglik = zero_lik)
    n = flat_dimension(tree)
    z = collect(range(-0.7, 0.7; length = n))
    x, logjac = to_constrained(prob, z)
    nt = unflatten(tree, x)

    # The unconstrained-space log-target, rebuilt independently row by row
    # through Bijectors' own `transformed` distribution (a different code
    # path from `to_constrained`'s `with_logabsdet_jacobian` call) — for a
    # centred-pooled row, against the population collapsed at `nt`'s current
    # hyperparameters, matching what `logdensity` itself scores.
    target = sum(eachindex(z)) do i
        prior = prob.flat_priors[i]
        eff = prior isa CentredPoolPrior ?
              _collapse_population(prior.pool.population,
            _pool_hyper(nt, prior.pool)) : prior
        logpdf(transformed(eff), z[i])
    end
    @test logdensity(prob, x) + logjac ≈ target
end

@testitem "gradient: ForwardDiff through to_constrained ∘ logdensity" begin
    using Distributions
    using ForwardDiff
    using ComposedDistributions: as_logdensity, to_constrained, logdensity,
                                 flat_dimension

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
    data = [(onset_admit = 2.0, north = 1.5, east = 1.8)]
    grouped(d, recs) = sum(
        logpdf(event(d, k), r[k]) for r in recs for k in keys(r))
    prob = as_logdensity(tree, data; loglik = grouped)

    n = flat_dimension(tree)
    z0 = fill(0.1, n)
    target(z) = ((x, logjac) = to_constrained(prob, z); logdensity(prob, x) + logjac)

    g = ForwardDiff.gradient(target, z0)
    @test length(g) == n
    @test all(isfinite, g)

    h = 1e-6
    for i in eachindex(z0)
        e = [j == i ? h : 0.0 for j in eachindex(z0)]
        fd = (target(z0 .+ e) - target(z0 .- e)) / (2h)
        @test g[i] ≈ fd atol = 1e-4
    end
end
