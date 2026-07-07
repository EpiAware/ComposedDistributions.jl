# Tests for the PPL-neutral LogDensityProblems core codec (#50): the flat <->
# nested `NamedTuple` bijection and the assembled `ComposedLogDensity`. No
# DynamicPPL/Turing/LogDensityProblems dependency here — the weakdep extension
# that wraps `ComposedLogDensity` as a standard `LogDensityProblems` problem is
# deferred (issue #13). The load-bearing checks are the codec round-trip and
# that `logdensity` sums the prior and data log-densities correctly.

@testitem "codec: flatten/unflatten round-trip (basic)" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @test flat_dimension(tree) ==
          length(params_table(tree).edge)

    # The table's `value` column IS the flat layout; unflatten -> update-shaped
    # nested NamedTuple -> flatten reproduces it exactly.
    x = collect(params_table(tree).value)
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    # The nested NamedTuple reconstructs the template via the core `update`.
    @test update(tree, nt) == tree
end

@testitem "codec: shared tag round-trip" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten

    d = compose((a = Gamma(2.0, 1.0), b = Gamma(2.0, 1.0)))
    tied = tie(d, :a, :b; name = :g)
    x = collect(params_table(tied).value)
    nt = unflatten(tied, x)
    @test haskey(nt, :g)
    @test flatten(tied, nt) == x
    @test update(tied, nt) == tied
end

@testitem "codec: an uncertain leaf's row collapses via update, no guard" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    @test has_uncertain(tree)

    x = collect(params_table(tree).value)
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    # A flat draw collapses the uncertain leaf to a concrete distribution.
    collapsed = update(tree, nt)
    @test !has_uncertain(collapsed)
    @test event(collapsed, :onset_admit) == Gamma(2.0, 1.0)
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
    @test_throws ArgumentError unflatten(tree, [2.0, 1.0, 0.5, 0.4])

    # Resolving against a context first lifts the guard.
    resolved = instantiate(tree, Context(time = 5.0))
    @test !has_varying(resolved)
    @test flat_dimension(resolved) == 4
end

@testitem "codec: flatten/unflatten report a clear DimensionMismatch" begin
    using Distributions
    using ComposedDistributions: unflatten

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    @test_throws DimensionMismatch unflatten(tree, [1.0, 2.0])
end

@testitem "logdensity: sum of priors + data likelihood" begin
    using Distributions
    using ComposedDistributions: flatten, unflatten, as_logdensity, logdensity

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(params_table(tree))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]
    prob = as_logdensity(tree, priors, data)

    x = collect(params_table(tree).value)
    ld = logdensity(prob, x)

    flat_p = flatten(tree, priors)
    manual_prior = sum(logpdf(flat_p[i], x[i]) for i in eachindex(x))
    d = update(tree, unflatten(tree, x))
    manual_lik = sum(logpdf(d, y) for y in data)
    @test ld ≈ manual_prior + manual_lik

    # The data-only assembler form defaults priors to build_priors(tree).
    prob2 = as_logdensity(tree, data)
    @test logdensity(prob2, x) ≈ ld
end

@testitem "logdensity: custom loglik reducer is used" begin
    using Distributions
    using ComposedDistributions: flatten, as_logdensity, logdensity

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    priors = build_priors(params_table(tree))
    data = [[0.5, 2.0], [1.0, 3.0]]
    zero_lik(d, data) = 0.0
    prob = as_logdensity(tree, priors, data; loglik = zero_lik)

    x = collect(params_table(tree).value)
    flat_p = flatten(tree, priors)
    manual_prior = sum(logpdf(flat_p[i], x[i]) for i in eachindex(x))
    @test logdensity(prob, x) ≈ manual_prior
end

@testitem "gradient: ForwardDiff through logdensity is finite" begin
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, flat_dimension
    using ForwardDiff

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    prob = as_logdensity(tree, build_priors(params_table(tree)),
        [[0.5, 2.0], [1.0, 3.0]])

    x0 = Float64.(params_table(tree).value)
    g = ForwardDiff.gradient(x -> logdensity(prob, x), x0)
    @test length(g) == flat_dimension(tree)
    @test all(isfinite, g)
end
