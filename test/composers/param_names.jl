# Type-level leaf parameter names (#372): `param_names`/`leaf_param_names`
# derive their answer from the leaf's own TYPE (fieldnames transliterated,
# matched against the arity of `params`), not from a curated table and not
# from the leaf's runtime values. These items pin the derived names for a
# representative slice of Distributions.jl, the value-tie stability property
# that is the whole point of a type-level (not value-level) derivation, the
# structural fallbacks, explicit-override precedence, wrapper read-through,
# and the identity guard that catches a field/params order mismatch a purely
# structural (type-only) rule cannot see.

@testitem "param_names: derived names for a representative slice" begin
    using Distributions

    @test ComposedDistributions.param_names(Normal(0.0, 1.0)) == (:mu, :sigma)
    @test ComposedDistributions.param_names(LogNormal(0.0, 1.0)) ==
          (:mu, :sigma)
    @test ComposedDistributions.param_names(Gamma(2.0, 1.0)) ==
          (:alpha, :theta)
    @test ComposedDistributions.param_names(Weibull(2.0, 1.0)) ==
          (:alpha, :theta)
    @test ComposedDistributions.param_names(Erlang(2, 1.0)) == (:alpha, :theta)
    @test ComposedDistributions.param_names(Exponential(1.0)) == (:theta,)
    @test ComposedDistributions.param_names(Uniform(0.0, 1.0)) == (:a, :b)
    @test ComposedDistributions.param_names(Beta(1.0, 1.0)) == (:alpha, :beta)
    @test ComposedDistributions.param_names(Cauchy(0.0, 1.0)) == (:mu, :sigma)
    @test ComposedDistributions.param_names(Laplace(0.0, 1.0)) ==
          (:mu, :theta)
    @test ComposedDistributions.param_names(TDist(3.0)) == (:nu,)
    @test ComposedDistributions.param_names(Poisson(1.0)) == (:lambda,)
    @test ComposedDistributions.param_names(VonMises(1.0, 1.0)) ==
          (:mu, :kappa)
    @test ComposedDistributions.param_names(NormalCanon(1.0, 1.0)) ==
          (:eta, :lambda)
    @test ComposedDistributions.param_names(JohnsonSU(0.0, 1.0, 0.0, 1.0)) ==
          (:xi, :lambda, :gamma, :delta)
    @test ComposedDistributions.param_names(TriangularDist(0.0, 1.0, 0.5)) ==
          (:a, :b, :c)
end

@testitem "param_names: stable under value ties" begin
    # The whole point of a type-level derivation: names come from the type
    # alone, so two instances of the same family with tied/equal parameter
    # values must still derive the same names as their non-tied counterparts.
    using Distributions

    @test ComposedDistributions.param_names(Gamma(2.0, 2.0)) == (:alpha, :theta)
    @test ComposedDistributions.param_names(DiscreteUniform(1, 1)) == (:a, :b)
    @test ComposedDistributions.param_names(NormalCanon(1.0, 1.0)) ==
          (:eta, :lambda)
    @test ComposedDistributions.param_names(VonMises(1.0, 1.0)) ==
          (:mu, :kappa)
    @test ComposedDistributions.param_names(Beta(1.0, 1.0)) == (:alpha, :beta)
    @test ComposedDistributions.param_names(TriangularDist(1.0, 1.0, 1.0)) ==
          (:a, :b, :c)
    @test ComposedDistributions.param_names(Hypergeometric(2, 2, 2)) ==
          (:ns, :nf, :n)
end

@testitem "param_names: structural fallbacks" begin
    using Distributions

    # `params` arity does not line up 1:1 with the struct's own fields.
    @test ComposedDistributions.param_names(Categorical([0.5, 0.5])) ==
          (:param_1,)
    @test ComposedDistributions.param_names(InverseGamma(2.0, 1.0)) ==
          (:param_1, :param_2)
    @test ComposedDistributions.param_names(OrderStatistic(Normal(), 5, 2)) ==
          (:param_1, :param_2, :param_3, :param_4)
    # Zero parameters derives the empty tuple, not a fallback.
    @test ComposedDistributions.param_names(Kolmogorov()) == ()
end

@testitem "param_names: an explicit override still wins" begin
    using Distributions

    struct P372MomentLeaf <: ContinuousUnivariateDistribution
        mean::Float64
        sd::Float64
    end
    Distributions.params(d::P372MomentLeaf) = (d.mean, d.sd)
    ComposedDistributions.param_names(::P372MomentLeaf) = (:mean, :sd)

    @test ComposedDistributions.param_names(P372MomentLeaf(1.0, 2.0)) ==
          (:mean, :sd)
end

@testitem "param_names: reads through wrapper layers via inner_dist" begin
    using Distributions

    @test ComposedDistributions.param_names(
        truncated(Gamma(2.0, 1.0); upper = 10.0)) == (:alpha, :theta)
    @test ComposedDistributions.param_names(
        censored(Gamma(2.0, 1.0); upper = 10.0)) == (:alpha, :theta)
    u = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2))
    @test ComposedDistributions.param_names(u) == (:alpha, :theta)
end

@testitem "param_names / leaf_param_names: inferred and allocation-free" begin
    using Distributions, Test

    g = Gamma(2.0, 1.0)
    f(d) = ComposedDistributions.param_names(d)
    h(d) = ComposedDistributions.leaf_param_names(d)
    @test (@inferred f(g)) == (:alpha, :theta)
    @test (@inferred h(g)) == (:alpha, :theta)
    f(g) # warm up
    h(g)
    @test (@allocated f(g)) == 0
    @test (@allocated h(g)) == 0
end

@testitem "param_names: field/params order mismatch is caught at \
    uncertain(...)" begin
    using Distributions

    # A contract-conforming leaf (params/fieldtypes structurally line up, both
    # Float64) whose field ORDER differs from its `params` order -- the type
    # rule cannot see this (the structural check only compares types, not
    # position semantics), so it must be caught by the identity guard at
    # `uncertain(...)` construction. Distinct values are required: the guard
    # is itself value-dependent and would pass on a degenerate tie.
    struct P372Swapped <: ContinuousUnivariateDistribution
        sigma::Float64
        mu::Float64
    end
    Distributions.params(d::P372Swapped) = (d.mu, d.sigma)

    # The derivation itself gives the wrong labels (measured, documented
    # behaviour of a type-only rule).
    @test ComposedDistributions.param_names(P372Swapped(1.0, 2.0)) ==
          (:sigma, :mu)

    err = try
        uncertain(P372Swapped(1.0, 2.0); sigma = LogNormal(0.0, 1.0))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("param_names", err.msg)

    # A conforming leaf (correct field order) is untouched by the guard.
    u = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2))
    @test u isa ComposedDistributions.Uncertain
end
