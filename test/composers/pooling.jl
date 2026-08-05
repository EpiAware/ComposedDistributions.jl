# Partial-pooling tests: a `pool` spec inside `uncertain` drawing each member's
# parameter from a shared population distribution. Covers the non-centred
# (location-theta) and centred (general population) paths, the pooling spectrum
# (tie / independent / pool), the CD-aligned flat layout, the codec round-trip
# and collapse, and a prior-predictive shrinkage check. See issue #78.
#
# The as_logdensity/logdensity-dependent scoring/gradient tests moved to
# DistributionsInference.jl (EpiAware/DistributionsInference.jl#70) with the
# rest of the inference layer (#185, #317).

@testitem "pool: constructor and validation" begin
    using Distributions

    using ComposedDistributions: pool_group, pool_noncentred

    # Default: an estimated-LogNormal population, reparameterised non-centred.
    p = pool(:district)
    @test p isa Pool
    @test pool_group(p) === :district
    @test pool_noncentred(p)
    @test p.population isa Uncertain

    # An explicit location-theta population stays non-centred.
    q = pool(:region,
        uncertain(Normal(0.0, 1.0);
            mu = Normal(0.5, 0.3), sigma = truncated(Normal(0.0, 0.2); lower = 0)))
    @test pool_noncentred(q)

    # A general population takes the centred path.
    r = pool(:g, uncertain(Gamma(2.0, 1.0);
        alpha = truncated(Normal(2, 1); lower = 0)))
    @test !pool_noncentred(r)

    # A fixed (non-uncertain) population is allowed (no hyperparameters).
    @test pool(:g, LogNormal(0.5, 0.3)).population == LogNormal(0.5, 0.3)

    # Equality by group, population and parameterisation.
    @test pool(:g) == pool(:g)
    @test pool(:g) != pool(:h)
    @test pool(:g, Normal(0.0, 1.0)) != pool(:g, LogNormal(0.0, 1.0))

    # Non-centred cannot be forced on a general population.
    @test_throws ArgumentError pool(:g, Gamma(2.0, 1.0); noncentred = true)
    # But it can be forced off on a location-theta one (centred LogNormal).
    @test !pool_noncentred(pool(:g, LogNormal(0.0, 1.0); noncentred = false))
end

@testitem "pool: rides an uncertain leaf and is seen as uncertain" begin
    using ComposedDistributions: update
    using Distributions

    leaf = uncertain(Gamma(2.0, 1.0); alpha = pool(:district))
    @test leaf isa Uncertain
    @test leaf.specs.alpha isa Pool
    @test has_uncertain(leaf)

    # An unknown parameter is rejected like any spec.
    @test_throws ArgumentError uncertain(Gamma(2.0, 1.0); rate = pool(:g))

    # `update` can also attach a pool spec (merge mode).
    attached = update(compose((a = Gamma(2.0, 1.0),)),
        (a = (alpha = pool(:g),),))
    @test has_uncertain(attached)
    @test event(attached, :a).specs.alpha isa Pool
end

@testitem "pool: the pooling spectrum (tie / independent / pool)" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    # Three strata whose `alpha` is estimated three ways; `theta` fixed at 1.0.
    # Complete pooling: one shared free `alpha` across every stratum (tie).
    complete = compose((
        north = shared(:sh, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0))),
        east = shared(:sh, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0))),
        south = shared(:sh, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0)))))
    @test flat_dimension(complete) == 1

    # No pooling: three independent free shapes.
    independent = compose((
        north = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0)),
        east = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0)),
        south = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0))))
    @test flat_dimension(independent) == 3

    # Partial pooling: the default population's two hyperparameters (mu, sigma)
    # plus one latent per stratum, so 2 + K over K = 3 strata.
    partial = compose((
        north = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); alpha = pool(:district))))
    @test flat_dimension(partial) == 2 + 3
end

@testitem "pool: params_table rows are the population hypers plus one latent per member" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    model = compose((
        north = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); alpha = pool(:district))))
    tbl = params_table(model)

    # The population's hyperparameters are inventoried once under the group edge.
    hyper = findall(==(:district), tbl.edge)
    @test tbl.param[hyper] == [:mu, :sigma]
    @test tbl.prior[hyper[1]] == Normal(0.0, 1.0)
    @test tbl.prior[hyper[2]] == truncated(Normal(0.0, 1.0); lower = 0.0)

    # One non-centred latent per member, labelled `<edge>.<param>.z`.
    z_rows = findall(==(:z), tbl.param)
    @test tbl.edge[z_rows] ==
          [Symbol("north.alpha"), Symbol("east.alpha"), Symbol("south.alpha")]
    @test all(tbl.prior[i] == Normal(0.0, 1.0) for i in z_rows)

    # 2 hyperparameters plus one latent per member (3 strata) = 5.
    @test flat_dimension(model) == 5
end

@testitem "pool: non-centred codec round-trip and reconstruction" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flatten, unflatten

    model = compose((
        north = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); alpha = pool(:district))))

    x = [0.1, 0.5, 0.3, -0.2, 0.8]      # mu, sigma, z_north, z_east, z_south
    nt = unflatten(model, x)
    @test flatten(model, nt) == x

    # Collapsing at the draw reconstructs each stratum's alpha as
    # exp(mu + sigma*z_k), theta held at the template.
    collapsed = update(model, nt)
    @test !has_uncertain(collapsed)
    mu, sigma = 0.1, 0.5
    @test params(event(collapsed, :north))[1] ≈ exp(mu + sigma * 0.3)
    @test params(event(collapsed, :east))[1] ≈ exp(mu + sigma * -0.2)
    @test params(event(collapsed, :south))[1] ≈ exp(mu + sigma * 0.8)
    @test params(event(collapsed, :north))[2] ≈ 1.0    # theta fixed

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

@testitem "pool: centred general population" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension

    # A Gamma population (not location-theta) takes the centred path: each
    # member's alpha IS its latent, scored directly against the population.
    pop = uncertain(Gamma(2.0, 1.0);
        alpha = truncated(Normal(2.0, 1.0); lower = 0),
        theta = truncated(Normal(1.0, 1.0); lower = 0))
    model = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, pop)),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, pop))))

    tbl = params_table(model)
    # Two hyperparameters under the group edge, then each member's own alpha.
    @test tbl.param[findall(==(:g), tbl.edge)] == [:alpha, :theta]
    @test flat_dimension(model) == 4   # 2 hypers + 2 member latents

    x = [2.5, 1.2, 3.0, 1.5]           # pop alpha, pop theta, theta_a, theta_b
    nt = unflatten(model, x)
    @test flatten(model, nt) == x

    # Centred: the member's alpha is its latent directly.
    collapsed = update(model, nt)
    @test params(event(collapsed, :a))[1] ≈ 3.0
    @test params(event(collapsed, :b))[1] ≈ 1.5
end

@testitem "pool: centred rows and their population prior term" begin
    using Distributions
    using ComposedDistributions: centred_pool_rows, pool_centred_logprior,
                                 unflatten

    # A centred pooled parameter's prior is parameter-dependent (the population
    # is reconstructed at the current hyperparameters), so it is not in the
    # fixed per-row prior vector. These two are how DistributionsInference.jl
    # scores it: `centred_pool_rows` once at construction, then
    # `pool_centred_logprior` per draw.
    pop = uncertain(Gamma(2.0, 1.0);
        alpha = truncated(Normal(2.0, 1.0); lower = 0),
        theta = truncated(Normal(1.0, 1.0); lower = 0))
    model = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, pop)),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, pop))))

    rows = centred_pool_rows(model)
    @test length(rows) == 2
    @test [r[1] for r in rows] == [(:a,), (:b,)]
    @test [r[2] for r in rows] == [:alpha, :alpha]
    @test all(r -> r[3] == pool(:g, pop), rows)

    # pop alpha, pop theta, theta_a, theta_b: each member's latent IS its
    # alpha, scored against the population at the drawn hyperparameters.
    nt = unflatten(model, [2.5, 1.2, 3.0, 1.5])
    @test pool_centred_logprior(rows, nt) ≈
          logpdf(Gamma(2.5, 1.2), 3.0) + logpdf(Gamma(2.5, 1.2), 1.5)

    # A fixed population needs no hyperparameters and is scored as-is.
    fixed = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, Beta(2.0, 3.0))),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, Beta(2.0, 3.0)))))
    fixed_rows = centred_pool_rows(fixed)
    fixed_nt = unflatten(fixed, [0.4, 0.7])
    @test pool_centred_logprior(fixed_rows, fixed_nt) ≈
          logpdf(Beta(2.0, 3.0), 0.4) + logpdf(Beta(2.0, 3.0), 0.7)

    # A non-centred (location-theta) population contributes no rows and no
    # term: its latents are ordinary `Normal(0, 1)` per-row priors.
    noncentred = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0))),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0)))))
    @test isempty(centred_pool_rows(noncentred))
    @test pool_centred_logprior(centred_pool_rows(noncentred),
        unflatten(noncentred, [0.1, 0.2])) == 0.0
end

@testitem "pool: rejects a hand-built update missing the population entry" begin
    using ComposedDistributions: update
    using Distributions

    # A hand-built update missing the population entry errors clearly.
    model = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g)),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g))))
    @test_throws ArgumentError update(model,
        (a = (alpha = (z = 0.1,), theta = 1.0),
            b = (alpha = (z = 0.2,), theta = 1.0)))
end

@testitem "pool: rejects an inconsistent group" begin
    using Distributions
    using ComposedDistributions: validate_pool_groups

    # Every member of a group is one population, so two members declaring
    # different populations under the same group name are rejected. The guard
    # runs once at construction time in DistributionsInference.jl's fit
    # protocol, so drive it directly here.
    bad = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0))),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, Normal(0.0, 1.0)))))
    @test_throws ArgumentError validate_pool_groups(bad)

    # A mismatched parameterisation on one shared population is the same
    # error: centred and non-centred members cannot share a group.
    mixed = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0))),
        b = uncertain(Gamma(2.0, 1.0);
            alpha = pool(:g, LogNormal(0.0, 1.0); noncentred = false))))
    @test_throws ArgumentError validate_pool_groups(mixed)

    # A consistent group passes and returns the tree.
    good = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0))),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, LogNormal(0.0, 1.0)))))
    @test validate_pool_groups(good) === good
end

@testitem "pool/shared: rejects a name shared across roles" begin
    using Distributions
    using ComposedDistributions: validate_tree_names

    # A pool group and a shared tag with the same name silently clobber each
    # other in the readback merge (#177); the name-collision guard rejects it.
    pool_vs_shared = compose((
        a = shared(:g, Gamma(2.0, 1.0)),
        b = uncertain(Gamma(3.0, 1.0); alpha = pool(:g))))
    @test_throws ArgumentError validate_tree_names(pool_vs_shared)

    # A pool group colliding with a sibling root-level edge name collides at
    # the same root-lifted level (#178 risk list).
    pool_vs_edge = compose((
        g = Gamma(2.0, 1.0),
        b = uncertain(Gamma(3.0, 1.0); alpha = pool(:g))))
    @test_throws ArgumentError validate_tree_names(pool_vs_edge)

    # A shared tag colliding with a sibling root-level edge name, same guard.
    shared_vs_edge = compose((
        g = Gamma(2.0, 1.0),
        b = shared(:g, LogNormal(0.5, 0.4))))
    @test_throws ArgumentError validate_tree_names(shared_vs_edge)
end

@testitem "pool/shared: legitimate tying is not a false positive" begin
    using Distributions
    using ComposedDistributions: validate_tree_names

    # The same shared tag tying a parameter across two branches is the
    # intended feature, not a collision, so it must still gate cleanly. The
    # tag name (`:inc`) is distinct from both root edge names (`:a`, `:b`)
    # and any pool group, so no guard fires.
    inc = shared(:inc, Gamma(2.0, 1.0))
    tied = compose((
        a = inc,
        b = compose((src = LogNormal(0.5, 0.4), inc = inc))))
    @test validate_tree_names(tied) === nothing

    # Two distinct pool groups and a distinct shared tag, none colliding with
    # each other or with the root edge names, also gate cleanly.
    clean = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        b = uncertain(Gamma(3.0, 1.0); alpha = pool(:region)),
        c = shared(:tag1, LogNormal(0.5, 0.4))))
    @test validate_tree_names(clean) === nothing

    # A pool group name equal to a NESTED (non-root) edge name is not a
    # collision: the guard only checks the tree's own ROOT edge names
    # (`:a`, `:branch` here), not `:g` two levels down inside `:branch`.
    nested_reuse = compose((
        a = Gamma(2.0, 1.0),
        branch = compose((g = Gamma(2.0, 1.0),
            b = uncertain(Gamma(3.0, 1.0); alpha = pool(:g))))))
    @test validate_tree_names(nested_reuse) === nothing
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
    leaf = uncertain(Gamma(2.0, 1.0); alpha = p)
    @test rand(Xoshiro(1), leaf) > 0
end

@testitem "pool: prior-predictive draws shrink toward the population" begin
    using ComposedDistributions: update
    using Distributions, Random, Statistics
    using ComposedDistributions: unflatten, flatten

    # A tight population theta, so pooled strata cluster; the unpooled strata
    # each carry the full spread.
    tight = uncertain(LogNormal(0.0, 1.0); mu = Normal(0.0, 1.0),
        sigma = truncated(Normal(0.0, 0.15); lower = 0))
    pooled = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, tight)),
        b = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, tight)),
        c = uncertain(Gamma(2.0, 1.0); alpha = pool(:g, tight))))
    unpooled = compose((
        a = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0)),
        b = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0)),
        c = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 1.0))))

    # Draw the joint prior-predictive by sampling the flat priors (the pooled
    # population is shared across strata within each draw) and reconstructing.
    # `default = _ -> nothing` gives every unspec'd row a `nothing` placeholder
    # instead of a default prior; `flatten` only ever reads the spec'd rows, so
    # a fixed row's placeholder is unused.
    function within_draw_spread(tree, rng, n)
        priors = build_priors(params_table(tree); default = _ -> nothing)
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
