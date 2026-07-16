# Partial-pooling tests: a `pool` spec inside `uncertain` drawing each member's
# parameter from a shared population distribution. Covers the non-centred
# (location-scale) and centred (general population) paths, the pooling spectrum
# (tie / independent / pool), the CD-aligned flat layout, the codec round-trip
# and collapse, the logdensity decomposition, AD gradients, and a
# prior-predictive shrinkage check. See issue #78.

@testitem "pool: constructor and validation" begin
    using Distributions

    using ComposedDistributions: pool_group, pool_noncentred

    # Default: an estimated-LogNormal population, reparameterised non-centred.
    p = pool(:district)
    @test p isa Pool
    @test pool_group(p) === :district
    @test pool_noncentred(p)
    @test p.population isa Uncertain

    # An explicit location-scale population stays non-centred.
    q = pool(:region,
        uncertain(Normal(0.0, 1.0);
            mu = Normal(0.5, 0.3), sigma = truncated(Normal(0.0, 0.2); lower = 0)))
    @test pool_noncentred(q)

    # A general population takes the centred path.
    r = pool(:g, uncertain(Gamma(2.0, 1.0);
        shape = truncated(Normal(2, 1); lower = 0)))
    @test !pool_noncentred(r)

    # A fixed (non-uncertain) population is allowed (no hyperparameters).
    @test pool(:g, LogNormal(0.5, 0.3)).population == LogNormal(0.5, 0.3)

    # Equality by group, population and parameterisation.
    @test pool(:g) == pool(:g)
    @test pool(:g) != pool(:h)
    @test pool(:g, Normal(0.0, 1.0)) != pool(:g, LogNormal(0.0, 1.0))

    # Non-centred cannot be forced on a general population.
    @test_throws ArgumentError pool(:g, Gamma(2.0, 1.0); noncentred = true)
    # But it can be forced off on a location-scale one (centred LogNormal).
    @test !pool_noncentred(pool(:g, LogNormal(0.0, 1.0); noncentred = false))
end

@testitem "pool: rides an uncertain leaf and is seen as uncertain" begin
    using Distributions

    leaf = uncertain(Gamma(2.0, 1.0); shape = pool(:district))
    @test leaf isa Uncertain
    @test leaf.specs.shape isa Pool
    @test has_uncertain(leaf)

    # An unknown parameter is rejected like any spec.
    @test_throws ArgumentError uncertain(Gamma(2.0, 1.0); rate = pool(:g))

    # `update` can also attach a pool spec (merge mode).
    attached = update(compose((a = Gamma(2.0, 1.0),)),
        (a = (shape = pool(:g),),))
    @test has_uncertain(attached)
    @test event(attached, :a).specs.shape isa Pool
end

@testitem "pool: the pooling spectrum (tie / independent / pool)" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    # Three strata whose `shape` is estimated three ways; `scale` fixed at 1.0.
    # Complete pooling: one shared free `shape` across every stratum (tie).
    complete = compose((
        north = shared(:sh, uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0))),
        east = shared(:sh, uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0))),
        south = shared(:sh, uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0)))))
    @test flat_dimension(complete) == 1

    # No pooling: three independent free shapes.
    independent = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0)),
        east = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0)),
        south = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0))))
    @test flat_dimension(independent) == 3

    # Partial pooling: the default population's two hyperparameters (mu, sigma)
    # plus one latent per stratum, so 2 + K over K = 3 strata.
    partial = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
    @test flat_dimension(partial) == 2 + 3
end

@testitem "pool: params_table rows and CD-aligned flat layout" begin
    using Distributions
    using ComposedDistributions: flat_dimension, _flat_layout

    model = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
    tbl = params_table(model)

    # The population's hyperparameters are inventoried once under the group edge.
    hyper = findall(==(:district), tbl.edge)
    @test tbl.param[hyper] == [:mu, :sigma]
    @test tbl.prior[hyper[1]] == Normal(0.0, 1.0)
    @test tbl.prior[hyper[2]] == truncated(Normal(0.0, 1.0); lower = 0.0)

    # One non-centred latent per member, labelled `<edge>.<param>.z`.
    z_rows = findall(==(:z), tbl.param)
    @test tbl.edge[z_rows] ==
          [Symbol("north.shape"), Symbol("east.shape"), Symbol("south.shape")]
    @test all(tbl.prior[i] == Normal(0.0, 1.0) for i in z_rows)

    # The estimated flat layout is exactly [mu, sigma, z_1, z_2, z_3] — the same
    # vector a CensoredDistributions user hand-writes in a Turing @model
    # (mu ~ ...; sigma ~ ...; z ~ filldist(Normal(0, 1), K)).
    layout = _flat_layout(tbl)
    @test layout == [
        ((:district,), :mu),
        ((:district,), :sigma),
        ((:north, :shape), :z),
        ((:east, :shape), :z),
        ((:south, :shape), :z)]
    @test flat_dimension(model) == 5
end

@testitem "pool: non-centred codec round-trip and reconstruction" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten

    model = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))

    x = [0.1, 0.5, 0.3, -0.2, 0.8]      # mu, sigma, z_north, z_east, z_south
    nt = unflatten(model, x)
    @test flatten(model, nt) == x

    # Collapsing at the draw reconstructs each stratum's shape as
    # exp(mu + sigma*z_k), scale held at the template.
    collapsed = update(model, nt)
    @test !has_uncertain(collapsed)
    mu, sigma = 0.1, 0.5
    @test params(event(collapsed, :north))[1] ≈ exp(mu + sigma * 0.3)
    @test params(event(collapsed, :east))[1] ≈ exp(mu + sigma * -0.2)
    @test params(event(collapsed, :south))[1] ≈ exp(mu + sigma * 0.8)
    @test params(event(collapsed, :north))[2] ≈ 1.0    # scale fixed

    # A Normal population uses the identity link mu + sigma*z.
    real_model = compose((
        a = uncertain(Normal(0.0, 1.0);
            mu = pool(:g,
                uncertain(Normal(0.0, 1.0);
                    mu = Normal(0.0, 1.0),
                    sigma = truncated(Normal(0.0, 1.0); lower = 0)))),
        b = uncertain(Normal(0.0, 1.0);
            mu = pool(:g,
                uncertain(Normal(0.0, 1.0);
                    mu = Normal(0.0, 1.0),
                    sigma = truncated(Normal(0.0, 1.0); lower = 0))))))
    rc = update(real_model, unflatten(real_model, [0.2, 0.4, 1.0, -1.0]))
    @test params(event(rc, :a))[1] ≈ 0.2 + 0.4 * 1.0
    @test params(event(rc, :b))[1] ≈ 0.2 + 0.4 * -1.0
end

@testitem "pool: non-centred logdensity is hyperprior + latents + likelihood" begin
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, unflatten

    model = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))

    # Per-stratum grouped likelihood (mirrors CD's batched_event_logpdf).
    strata = (north = [1.0, 2.0], east = [1.5], south = [0.8, 1.2, 2.0])
    grouped(d, data) = sum(logpdf(event(d, k), r) for k in keys(data)
    for r in data[k])
    prob = as_logdensity(model, strata; loglik = grouped)

    x = [0.1, 0.5, 0.3, -0.2, 0.8]
    ld = logdensity(prob, x)

    # The hand-computed joint: the two hyperpriors, the three standard-normal
    # latents, and the reconstructed-tree likelihood.
    hyper = logpdf(Normal(0.0, 1.0), 0.1) +
            logpdf(truncated(Normal(0.0, 1.0); lower = 0.0), 0.5)
    latents = logpdf(Normal(0.0, 1.0), 0.3) + logpdf(Normal(0.0, 1.0), -0.2) +
              logpdf(Normal(0.0, 1.0), 0.8)
    lik = grouped(update(model, unflatten(model, x)), strata)
    @test ld ≈ hyper + latents + lik
end

@testitem "pool: centred general population" begin
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, flatten, unflatten,
                                 flat_dimension

    # A Gamma population (not location-scale) takes the centred path: each
    # member's shape IS its latent, scored directly against the population.
    pop = uncertain(Gamma(2.0, 1.0);
        shape = truncated(Normal(2.0, 1.0); lower = 0),
        scale = truncated(Normal(1.0, 1.0); lower = 0))
    model = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:g, pop)),
        b = uncertain(Gamma(2.0, 1.0); shape = pool(:g, pop))))

    tbl = params_table(model)
    # Two hyperparameters under the group edge, then each member's own shape.
    @test tbl.param[findall(==(:g), tbl.edge)] == [:shape, :scale]
    @test flat_dimension(model) == 4   # 2 hypers + 2 member latents

    x = [2.5, 1.2, 3.0, 1.5]           # pop shape, pop scale, theta_a, theta_b
    nt = unflatten(model, x)
    @test flatten(model, nt) == x

    # Centred: the member's shape is its latent directly.
    collapsed = update(model, nt)
    @test params(event(collapsed, :a))[1] ≈ 3.0
    @test params(event(collapsed, :b))[1] ≈ 1.5

    strata = (a = [1.0], b = [2.0])
    grouped(d, data) = sum(logpdf(event(d, k), r) for k in keys(data)
    for r in data[k])
    prob = as_logdensity(model, strata; loglik = grouped)
    ld = logdensity(prob, x)

    # Joint: the two hyperpriors, each member's shape scored against the
    # population Gamma(2.5, 1.2), and the likelihood.
    hyper = logpdf(truncated(Normal(2.0, 1.0); lower = 0), 2.5) +
            logpdf(truncated(Normal(1.0, 1.0); lower = 0), 1.2)
    population = Gamma(2.5, 1.2)
    linking = logpdf(population, 3.0) + logpdf(population, 1.5)
    lik = grouped(collapsed, strata)
    @test ld ≈ hyper + linking + lik
end

@testitem "pool: ForwardDiff gradient matches finite differences" begin
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, flat_dimension
    using ForwardDiff

    grouped(d, data) = sum(logpdf(event(d, k), r) for k in keys(data)
    for r in data[k])

    function check_gradient(model, x0, strata)
        prob = as_logdensity(model, strata; loglik = grouped)
        g = ForwardDiff.gradient(x -> logdensity(prob, x), x0)
        @test all(isfinite, g)
        h = 1e-6
        for i in eachindex(x0)
            e = [j == i ? h : 0.0 for j in eachindex(x0)]
            fd = (logdensity(prob, x0 .+ e) - logdensity(prob, x0 .- e)) / (2h)
            @test g[i] ≈ fd atol = 1e-4
        end
    end

    # Non-centred (LogNormal population).
    nc = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
    @test flat_dimension(nc) == 5
    check_gradient(nc, [0.1, 0.5, 0.3, -0.2, 0.8],
        (north = [1.0, 2.0], east = [1.5], south = [0.8, 1.2, 2.0]))

    # Centred (Gamma population): the gradient flows through the population
    # log-density at the estimated hyperparameters.
    pop = uncertain(Gamma(2.0, 1.0);
        shape = truncated(Normal(2.0, 1.0); lower = 0),
        scale = truncated(Normal(1.0, 1.0); lower = 0))
    ce = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:g, pop)),
        b = uncertain(Gamma(2.0, 1.0); shape = pool(:g, pop))))
    check_gradient(ce, [2.5, 1.2, 3.0, 1.5], (a = [1.0], b = [2.0]))
end

@testitem "pool: rejects an inconsistent group and a missing population" begin
    using Distributions
    using ComposedDistributions: as_logdensity

    # Two members of one group with different populations is rejected at the
    # log-density gate (one group is one population).
    bad = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:g, LogNormal(0.0, 1.0))),
        b = uncertain(Gamma(2.0, 1.0); shape = pool(:g, Normal(0.0, 1.0)))))
    @test_throws ArgumentError as_logdensity(bad, [1.0])

    # A hand-built update missing the population entry errors clearly.
    model = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:g)),
        b = uncertain(Gamma(2.0, 1.0); shape = pool(:g))))
    @test_throws ArgumentError update(model,
        (a = (shape = (z = 0.1,), scale = 1.0),
            b = (shape = (z = 0.2,), scale = 1.0)))
end

@testitem "pool/shared: rejects a name shared across roles" begin
    using Distributions
    using ComposedDistributions: as_logdensity

    # A pool group and a shared tag with the same name silently clobber each
    # other in the readback merge (#177); the log-density gate rejects it.
    pool_vs_shared = compose((
        a = shared(:g, Gamma(2.0, 1.0)),
        b = uncertain(Gamma(3.0, 1.0); shape = pool(:g))))
    @test_throws ArgumentError as_logdensity(pool_vs_shared, [1.0])

    # A pool group colliding with a sibling root-level edge name collides at
    # the same root-lifted level (#178 risk list).
    pool_vs_edge = compose((
        g = Gamma(2.0, 1.0),
        b = uncertain(Gamma(3.0, 1.0); shape = pool(:g))))
    @test_throws ArgumentError as_logdensity(pool_vs_edge, [1.0])

    # A shared tag colliding with a sibling root-level edge name, same guard.
    shared_vs_edge = compose((
        g = Gamma(2.0, 1.0),
        b = shared(:g, LogNormal(0.5, 0.4))))
    @test_throws ArgumentError as_logdensity(shared_vs_edge, [1.0])
end

@testitem "pool/shared: legitimate tying is not a false positive" begin
    using Distributions
    using ComposedDistributions: as_logdensity

    # The same shared tag tying a parameter across two branches is the
    # intended feature, not a collision, so it must still compose and gate
    # cleanly. The tag name (`:inc`) is distinct from both root edge names
    # (`:a`, `:b`) and any pool group, so no guard fires.
    inc = shared(:inc, Gamma(2.0, 1.0))
    tied = compose((
        a = inc,
        b = compose((src = LogNormal(0.5, 0.4), inc = inc))))
    prob = as_logdensity(tied, [1.0])
    @test prob isa ComposedDistributions.ComposedLogDensity

    # Two distinct pool groups and a distinct shared tag, none colliding with
    # each other or with the root edge names, also gate cleanly.
    clean = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        b = uncertain(Gamma(3.0, 1.0); shape = pool(:region)),
        c = shared(:tag1, LogNormal(0.5, 0.4))))
    prob2 = as_logdensity(clean, [1.0])
    @test prob2 isa ComposedDistributions.ComposedLogDensity

    # A pool group name equal to a NESTED (non-root) edge name is not a
    # collision: the guard only checks the tree's own ROOT edge names
    # (`:a`, `:branch` here), not `:g` two levels down inside `:branch`.
    nested_reuse = compose((
        a = Gamma(2.0, 1.0),
        branch = compose((g = Gamma(2.0, 1.0),
            b = uncertain(Gamma(3.0, 1.0); shape = pool(:g))))))
    prob3 = as_logdensity(nested_reuse, [1.0])
    @test prob3 isa ComposedDistributions.ComposedLogDensity
end

@testitem "pool: rand draws a single-parameter marginal" begin
    using Distributions, Random

    # A tight population concentrates the marginal near exp(0) = 1.
    p = pool(:g,
        uncertain(LogNormal(0.0, 1.0);
            mu = Normal(0.0, 0.01),
            sigma = truncated(Normal(0.0, 0.01); lower = 0)))
    draws = [rand(Xoshiro(i), p) for i in 1:500]
    @test all(>(0), draws)
    @test abs(sum(draws) / length(draws) - 1.0) < 0.1

    # `rand` on a pooled leaf draws its marginal (rebuilds the concrete leaf).
    leaf = uncertain(Gamma(2.0, 1.0); shape = p)
    @test rand(Xoshiro(1), leaf) > 0
end

@testitem "pool: prior-predictive draws shrink toward the population" begin
    using Distributions, Random, Statistics
    using ComposedDistributions: unflatten, flatten

    # A tight population scale, so pooled strata cluster; the unpooled strata
    # each carry the full spread.
    tight = uncertain(LogNormal(0.0, 1.0); mu = Normal(0.0, 1.0),
        sigma = truncated(Normal(0.0, 0.15); lower = 0))
    pooled = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = pool(:g, tight)),
        b = uncertain(Gamma(2.0, 1.0); shape = pool(:g, tight)),
        c = uncertain(Gamma(2.0, 1.0); shape = pool(:g, tight))))
    unpooled = compose((
        a = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0)),
        b = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0)),
        c = uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 1.0))))

    # Draw the joint prior-predictive by sampling the flat priors (the pooled
    # population is shared across strata within each draw) and reconstructing.
    function within_draw_spread(tree, rng, n)
        priors = ComposedDistributions._spec_priors(tree)
        fp = flatten(tree, priors)
        spreads = Float64[]
        for _ in 1:n
            x = [rand(rng, fp[i]) for i in eachindex(fp)]
            d = update(tree, unflatten(tree, x))
            shapes = [log(params(event(d, k))[1]) for k in (:a, :b, :c)]
            push!(spreads, std(shapes))
        end
        return mean(spreads)
    end

    rng = Xoshiro(42)
    pooled_spread = within_draw_spread(pooled, rng, 4000)
    unpooled_spread = within_draw_spread(unpooled, rng, 4000)
    # Pooled strata sit near the shared population, so their within-draw
    # cross-stratum spread is far smaller than the unpooled strata's.
    @test pooled_spread < 0.4 * unpooled_spread
end
