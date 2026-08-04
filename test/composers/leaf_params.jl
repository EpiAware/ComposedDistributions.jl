# #332 — one leaf-parameter table covering the whole univariate Distributions.jl
# library. Table-completeness / agreement tests, the existing-six no-rename
# guard, and the cached-field arity regression (VonMises/NormalCanon and the
# two extra families the review's own scan missed).

@testsnippet LeafParamsFixture begin
    using Distributions
    using ComposedDistributions: LEAF_PARAM_SPECS, LeafParamSpec, param_names,
                                 _param_names_of, _params_arity_of,
                                 leaf_param_names, _leaf_type_param_names,
                                 compose, uncertain, reconstruct, params_table

    # One buildable default instance per family in `LEAF_PARAM_SPECS`, using
    # the same arguments Distributions.jl's own docstrings show as an example
    # construction. Kept here (not derived from the spec table) so the test
    # is an independent check on the table, not a tautology.
    const _LEAF_PARAMS_TEST_INSTANCES = Dict{Type, Any}(
        Arcsine => Arcsine(0.0, 1.0), Bernoulli => Bernoulli(0.3),
        BernoulliLogit => BernoulliLogit(0.3), Beta => Beta(2.0, 3.0),
        BetaBinomial => BetaBinomial(10, 2.0, 3.0),
        BetaPrime => BetaPrime(2.0, 3.0), Binomial => Binomial(10, 0.3),
        Biweight => Biweight(0.0, 1.0), Cauchy => Cauchy(0.0, 1.0),
        Chi => Chi(3.0), Chisq => Chisq(3.0), Cosine => Cosine(0.0, 1.0),
        DiscreteUniform => DiscreteUniform(1, 10),
        Epanechnikov => Epanechnikov(0.0, 1.0), Erlang => Erlang(3, 1.0),
        Exponential => Exponential(2.0), FDist => FDist(3.0, 5.0),
        FisherNoncentralHypergeometric => FisherNoncentralHypergeometric(
            10, 7, 5, 2.0),
        Frechet => Frechet(2.0, 1.0), Gamma => Gamma(2.0, 1.0),
        GeneralizedExtremeValue => GeneralizedExtremeValue(0.0, 1.0, 0.2),
        GeneralizedPareto => GeneralizedPareto(0.0, 1.0, 0.2),
        Geometric => Geometric(0.3), Gumbel => Gumbel(0.0, 1.0),
        Hypergeometric => Hypergeometric(10, 7, 5),
        InverseGamma => InverseGamma(2.0, 1.0),
        InverseGaussian => InverseGaussian(1.0, 1.0),
        JohnsonSU => JohnsonSU(0.0, 1.0, 0.0, 1.0), Kolmogorov => Kolmogorov(),
        Kumaraswamy => Kumaraswamy(2.0, 3.0), Laplace => Laplace(0.0, 1.0),
        Levy => Levy(0.0, 1.0), Lindley => Lindley(1.0),
        LogLogistic => LogLogistic(1.0, 2.0), LogNormal => LogNormal(0.0, 1.0),
        LogUniform => LogUniform(1.0, 10.0), Logistic => Logistic(0.0, 1.0),
        LogitNormal => LogitNormal(0.0, 1.0),
        NegativeBinomial => NegativeBinomial(5, 0.3),
        NoncentralBeta => NoncentralBeta(2.0, 3.0, 1.0),
        NoncentralChisq => NoncentralChisq(3.0, 1.0),
        NoncentralF => NoncentralF(3.0, 5.0, 1.0),
        NoncentralT => NoncentralT(3.0, 1.0), Normal => Normal(0.0, 1.0),
        NormalCanon => NormalCanon(0.0, 1.0),
        NormalInverseGaussian => NormalInverseGaussian(0.0, 1.0, 0.5, 1.0),
        Pareto => Pareto(1.0, 1.0),
        PGeneralizedGaussian => PGeneralizedGaussian(0.0, 1.0, 2.0),
        Poisson => Poisson(3.0), Rayleigh => Rayleigh(1.0),
        Rician => Rician(1.0, 1.0), Semicircle => Semicircle(1.0),
        Skellam => Skellam(2.0, 3.0), SkewNormal => SkewNormal(0.0, 1.0, 0.5),
        SkewedExponentialPower => SkewedExponentialPower(0.0, 1.0, 2.0, 0.5),
        Soliton => Soliton(10, 5, 0.05, 0.1),
        StudentizedRange => StudentizedRange(3.0, 5.0),
        SymTriangularDist => SymTriangularDist(0.0, 1.0), TDist => TDist(3.0),
        TriangularDist => TriangularDist(0.0, 1.0, 0.5),
        Triweight => Triweight(0.0, 1.0), Uniform => Uniform(0.0, 1.0),
        VonMises => VonMises(0.0, 1.0),
        WalleniusNoncentralHypergeometric => WalleniusNoncentralHypergeometric(
            10, 7, 5, 2.0),
        Weibull => Weibull(2.0, 1.0))
end

@testitem "leaf_params: table completeness against a default instance of every family" setup=[
    LeafParamsFixture] begin
    for (T, spec) in LEAF_PARAM_SPECS
        d = get(_LEAF_PARAMS_TEST_INSTANCES, T, nothing)
        @test d !== nothing
        d === nothing && continue
        @test length(spec.names) == length(params(d))
        @test length(spec.supports) == length(spec.names)
    end
    # No family is missing from the fixture map above (the fixture is an
    # independent construction, not derived from the table).
    @test length(LEAF_PARAM_SPECS) == length(_LEAF_PARAMS_TEST_INSTANCES)
end

@testitem "leaf_params: instance-level and type-level tables agree for every family" setup=[
    LeafParamsFixture] begin
    for (T, spec) in LEAF_PARAM_SPECS
        d = _LEAF_PARAMS_TEST_INSTANCES[T]
        @test param_names(d) == spec.names
        @test _param_names_of(typeof(d)) == spec.names
        @test leaf_param_names(d) == _leaf_type_param_names(typeof(d))
        @test _params_arity_of(typeof(d)) == length(spec.names)
    end
end

@testitem "leaf_params: existing six families' row names are unchanged" setup=[
    LeafParamsFixture] begin
    @test param_names(Normal(0.0, 1.0)) == (:mu, :sigma)
    @test param_names(LogNormal(0.0, 1.0)) == (:mu, :sigma)
    @test param_names(Gamma(2.0, 1.0)) == (:shape, :scale)
    @test param_names(Weibull(2.0, 1.0)) == (:shape, :scale)
    @test param_names(Exponential(2.0)) == (:scale,)
    @test param_names(Uniform(0.0, 1.0)) == (:lower, :upper)
end

@testitem "leaf_params: cached-field families no longer read a stale extra field" setup=[
    LeafParamsFixture] begin
    # VonMises: fieldcount 3 (μ, κ, I0κx) against 2 params.
    @test fieldcount(VonMises{Float64}) == 3
    @test _params_arity_of(VonMises{Float64}) == 2
    # NormalCanon: fieldcount 3 (η, λ, μ) against 2 params.
    @test fieldcount(NormalCanon{Float64}) == 3
    @test _params_arity_of(NormalCanon{Float64}) == 2
    # DiscreteUniform / NormalInverseGaussian: found live during this table's
    # construction, the same cached-field shape as VonMises/NormalCanon.
    @test fieldcount(DiscreteUniform) == 3
    @test _params_arity_of(DiscreteUniform) == 2
    @test fieldcount(NormalInverseGaussian{Float64}) == 5
    @test _params_arity_of(NormalInverseGaussian{Float64}) == 4
end

@testitem "leaf_params: VonMises reconstruct regression (was BoundsError)" setup=[
    LeafParamsFixture] begin
    tr = compose((a = uncertain(VonMises(0.0, 1.0); mu = Normal(),
        kappa = Gamma(2.0, 1.0)),))
    rebuilt = reconstruct(tr, [0.5, 2.0])
    d = rebuilt.components[1]
    @test d isa VonMises
    @test d.μ == 0.5
    @test d.κ == 2.0
end

@testitem "leaf_params: a non-scalar native parameter is rejected, not silently pushed" setup=[
    LeafParamsFixture] begin
    tree = compose((a = Categorical([0.2, 0.3, 0.5]),))
    @test_throws ArgumentError params_table(tree)
end
