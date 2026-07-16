# Tests for the PPL-neutral LogDensityProblems core codec: the flat <-> nested
# `NamedTuple` bijection and the assembled `ComposedLogDensity`. No
# DynamicPPL/Turing/LogDensityProblems dependency here — the weakdep extension
# that wraps `ComposedLogDensity` as a standard `LogDensityProblems` problem is
# deferred (issue #13). Uncertain-first: the flat vector spans EXACTLY the
# spec'd (estimated) parameters, so a fixed leaf contributes nothing and a tree
# with no uncertain leaves estimates nothing. The load-bearing checks are the
# codec round-trip over the estimated subset and that `logdensity` sums the
# specs' and data log-densities correctly.

@testitem "codec: estimated flatten/unflatten round-trip" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4);
            mu = Normal(0.5, 0.3),
            sigma = truncated(Normal(0.4, 0.2); lower = 0))))

    # The estimated dimension is the number of spec'd parameters (shape, mu,
    # sigma); onset_admit.scale is fixed and contributes nothing.
    @test flat_dimension(tree) == 3

    # unflatten fills the estimated parameters from the flat vector and the
    # fixed parameters from the template, then flatten reads the estimated
    # subset back exactly.
    x = [2.5, 0.6, 0.45]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x

    # update collapses each uncertain leaf at the draw, holding scale fixed.
    collapsed = update(tree, nt)
    @test !has_uncertain(collapsed)
    @test event(collapsed, :onset_admit) == Gamma(2.5, 1.0)
    @test event(collapsed, :admit_death) == LogNormal(0.6, 0.45)
end

@testitem "codec: a fully fixed tree estimates nothing" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension,
                                 as_logdensity, logdensity

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # No uncertain leaves => nothing is estimated.
    @test flat_dimension(tree) == 0
    @test isempty(flatten(tree,
        (onset_admit = (shape = 2.0, scale = 1.0),
            admit_death = (mu = 0.5, sigma = 0.4))))
    # unflatten of the empty vector rebuilds the template unchanged.
    @test update(tree, unflatten(tree, Float64[])) == tree

    # logdensity is then just the data likelihood at the fixed tree.
    data = [[0.5, 2.0], [1.0, 3.0]]
    prob = as_logdensity(tree, data)
    @test logdensity(prob, Float64[]) ≈ sum(logpdf(tree, y) for y in data)
end

@testitem "codec: shared spec round-trip" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    d = compose((a = shared(:g, u), b = shared(:g, u)))

    # The tied shape is a single estimated parameter.
    @test flat_dimension(d) == 1
    x = [2.7]
    nt = unflatten(d, x)
    @test haskey(nt, :g)
    @test flatten(d, nt) == x

    collapsed = update(d, nt)
    @test !has_uncertain(collapsed)
    # Both occurrences collapse to the same shared leaf at the drawn value.
    @test event(collapsed, :a) == event(collapsed, :b)
    @test ComposedDistributions.free_leaf(event(collapsed, :a)) == Gamma(2.7, 1.0)
end

@testitem "codec: rejects a tree with unresolved varying leaves" begin
    using Distributions
    using ComposedDistributions: flat_dimension, flatten, unflatten

    tree = compose((onset_admit = varying(t -> Gamma(2.0, 1.0 + 0.1 * t)),
        admit_death = LogNormal(0.5, 0.4)))
    @test has_varying(tree)

    @test_throws ArgumentError flat_dimension(tree)
    @test_throws ArgumentError flatten(tree,
        (; onset_admit = (; shape = 2.0,
                scale = 1.0),
            admit_death = (; mu = 0.5, sigma = 0.4)))
    @test_throws ArgumentError unflatten(tree, Float64[])

    # Resolving against a context lifts the guard; the resolved tree is fully
    # fixed (no uncertain leaves), so it still estimates nothing.
    resolved = instantiate(tree, Context(time = 5.0))
    @test !has_varying(resolved)
    @test flat_dimension(resolved) == 0
end

@testitem "codec: unflatten reports a clear DimensionMismatch" begin
    using Distributions
    using ComposedDistributions: unflatten

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    # One estimated parameter, but a length-2 vector is passed.
    @test_throws DimensionMismatch unflatten(tree, [1.0, 2.0])
end

@testitem "logdensity: sum of specs + data likelihood" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, as_logdensity, logdensity,
                                 _spec_priors

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4);
            mu = Normal(0.5, 0.3),
            sigma = truncated(Normal(0.4, 0.2); lower = 0))))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    prob = as_logdensity(tree, data)

    x = [2.3, 0.55, 0.42]
    ld = logdensity(prob, x)

    # The priors are read off the object's specs; score them at x.
    flat_p = flatten(tree, _spec_priors(tree))
    manual_prior = sum(logpdf(flat_p[i], x[i]) for i in eachindex(x))
    d = update(tree, unflatten(tree, x))
    manual_lik = sum(logpdf(d, y) for y in data)
    @test ld ≈ manual_prior + manual_lik

    # The explicit-priors form overrides the on-object specs at the same
    # estimated rows.
    priors = (onset_admit = (shape = Exponential(2.0),),
        admit_death = (mu = Normal(0.0, 1.0),
            sigma = truncated(Normal(0.0, 1.0); lower = 0)))
    prob2 = as_logdensity(tree, priors, data)
    flat_p2 = flatten(tree, priors)
    manual_prior2 = sum(logpdf(flat_p2[i], x[i]) for i in eachindex(x))
    @test logdensity(prob2, x) ≈ manual_prior2 + manual_lik
end

@testitem "logdensity: custom loglik reducer is used" begin
    using Distributions
    using ComposedDistributions: flatten, as_logdensity, logdensity,
                                 _spec_priors

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.3))))
    data = [[0.5, 2.0], [1.0, 3.0]]
    zero_lik(d, data) = 0.0
    prob = as_logdensity(tree, data; loglik = zero_lik)

    x = [2.1, 0.5]
    flat_p = flatten(tree, _spec_priors(tree))
    manual_prior = sum(logpdf(flat_p[i], x[i]) for i in eachindex(x))
    @test logdensity(prob, x) ≈ manual_prior
end

@testitem "gradient: ForwardDiff through logdensity is finite" begin
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, flat_dimension
    using ForwardDiff

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = uncertain(LogNormal(0.5, 0.4);
            mu = Normal(0.5, 0.3),
            sigma = truncated(Normal(0.4, 0.2); lower = 0))))
    prob = as_logdensity(tree, [[0.5, 2.0], [1.0, 3.0]])

    x0 = [2.0, 0.5, 0.4]
    g = ForwardDiff.gradient(x -> logdensity(prob, x), x0)
    @test length(g) == flat_dimension(tree)
    @test all(isfinite, g)
end

@testitem "update: flat vector shorthand equals unflatten then update" begin
    using Distributions
    using ComposedDistributions: unflatten

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))

    # The two-step path via unflatten.
    x = [3.0]
    nt = unflatten(tree, x)
    result_unflatten = update(tree, nt)

    # The direct vector arm should give the same result.
    result_vector = update(tree, x)
    @test result_vector == result_unflatten
    @test event(result_vector, :onset_admit) == Gamma(3.0, 1.0)
end

@testitem "update: flat vector dimension mismatch throws like unflatten" begin
    using Distributions
    using ComposedDistributions: unflatten

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))

    # Wrong dimension should error.
    @test_throws DimensionMismatch update(tree, [1.0, 2.0])
end

@testitem "update: flat vector accepts duck-typed containers" begin
    using Distributions
    using ComposedDistributions: unflatten

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))

    # Duck-typed containers should work and be equivalent to regular vectors.
    result_vec = update(tree, [3.0])
    result_any = update(tree, Any[3.0])
    @test result_vec == result_any
end
