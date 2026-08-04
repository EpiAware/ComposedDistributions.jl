# #332 — one leaf-parameter table covering the whole univariate Distributions.jl
# library's instance-level `param_names`/`param_supports`. Table-completeness
# tests, the existing-six no-rename guard, and the cached-field arity
# regression (VonMises/NormalCanon and the two extra families the review's
# own scan missed) -- the latter is fixed at the instance level here but
# still open at the type level (codec_gen.jl, `_params_arity_of`) pending the
# staged codec work that replaces that table outright; see
# src/composers/leaf_params.jl's header.

@testsnippet LeafParamsFixture begin
    using Distributions
    using ComposedDistributions: LEAF_PARAM_SPECS, LeafParamSpec, param_names,
                                 param_supports, _params_arity_of, compose,
                                 uncertain, reconstruct, params_table,
                                 build_priors

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

@testitem "leaf_params: instance-level param_names matches the table for every family" setup=[
    LeafParamsFixture] begin
    # `_param_names_of`/`_params_arity_of` (codec_gen.jl's type-level mirror)
    # are deliberately not checked here: they stay their pre-#332
    # hand-written six-family selves on this branch (rescoped so as not to
    # conflict with the staged codec work that replaces them outright), so
    # only the original six agree with this table at the type level -- see
    # the "param_names / _param_names_of" agreement test in
    # codec_consistency.jl for that narrower check.
    for (T, spec) in LEAF_PARAM_SPECS
        d = _LEAF_PARAMS_TEST_INSTANCES[T]
        @test param_names(d) == spec.names
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

@testitem "leaf_params: cached-field families' instance-level arity reads params, not fieldcount" setup=[
    LeafParamsFixture] begin
    # VonMises: fieldcount 3 (μ, κ, I0κx) against 2 params. `param_names`
    # (instance-level, generated from `LEAF_PARAM_SPECS`) is unaffected --
    # it is read straight off the table, never off `fieldcount`.
    @test fieldcount(VonMises{Float64}) == 3
    @test length(param_names(VonMises(0.0, 1.0))) == 2
    # NormalCanon: fieldcount 3 (η, λ, μ) against 2 params.
    @test fieldcount(NormalCanon{Float64}) == 3
    @test length(param_names(NormalCanon(0.0, 1.0))) == 2
    # DiscreteUniform / NormalInverseGaussian: found live during this table's
    # construction, the same cached-field shape as VonMises/NormalCanon.
    @test fieldcount(DiscreteUniform) == 3
    @test length(param_names(DiscreteUniform(1, 10))) == 2
    @test fieldcount(NormalInverseGaussian{Float64}) == 5
    @test length(param_names(
        NormalInverseGaussian(0.0, 1.0, 0.5, 1.0))) == 4

    # `_params_arity_of` (codec_gen.jl's type-level mirror) is a *different*
    # table on this branch (rescoped so as not to conflict with the staged
    # codec work that replaces it outright, see src/composers/leaf_params.jl's
    # header) and still falls back to `fieldcount` for all four of these --
    # a leaf type with no explicit `_param_names_of`/`_params_arity_of`
    # override, VonMises/NormalCanon/DiscreteUniform/NormalInverseGaussian
    # among them, is exactly the case that fallback gets wrong. Left as
    # `@test_broken` so the day the staged codec work lands and fixes this,
    # the flip to an unexpected pass is loud rather than silent.
    @test_broken _params_arity_of(VonMises{Float64}) == 2
    @test_broken _params_arity_of(NormalCanon{Float64}) == 2
    @test_broken _params_arity_of(DiscreteUniform) == 2
    @test_broken _params_arity_of(NormalInverseGaussian{Float64}) == 4
end

@testitem "leaf_params: VonMises reconstruct regression (was BoundsError; still open, blocked on the staged codec work)" setup=[
    LeafParamsFixture] begin
    # This was the motivating bug for #332's `_params_arity_of` fix: with
    # `_params_arity_of(VonMises) == fieldcount(VonMises) == 3` (one too
    # many, the cached `I0κx` field), `reconstruct` reads a third flat value
    # that was never written and throws a `BoundsError`. The fix landed at
    # the type level in codec_gen.jl in the original #332 commit, but this
    # branch was rescoped to leave codec_gen.jl untouched -- that table is
    # being deleted outright by the staged codec work (feat/codec-s2-leaf-swap
    # / feat/codec-s3-*) rather than merged with this branch's copy -- so the
    # bug is still live here. Left as `@test_broken` (via a boolean-returning
    # try/catch, since the failure mode is an exception, not a wrong value)
    # so this flips to an unexpected pass, loudly, once one of those codec
    # branches lands and this can be promoted back to a plain `@test`.
    tr = compose((a = uncertain(VonMises(0.0, 1.0); mu = Normal(),
        kappa = Gamma(2.0, 1.0)),))
    @test_broken try
        rebuilt = reconstruct(tr, [0.5, 2.0])
        d = rebuilt.components[1]
        d isa VonMises && d.μ == 0.5 && d.κ == 2.0
    catch
        false
    end
end

@testitem "leaf_params: a non-scalar native parameter is rejected, not silently pushed" setup=[
    LeafParamsFixture] begin
    tree = compose((a = Categorical([0.2, 0.3, 0.5]),))
    @test_throws ArgumentError params_table(tree)
end

# Named parameters (#332) whose true domain differs from the family's variate
# support and from the generic name heuristics `_is_positive_param`/
# `_is_location_param`. Before `param_supports` was consumed by the walk
# (and before `default_prior` deferred to it ahead of the name heuristics),
# giving these families real names silently flipped `build_priors`'s output
# from right to wrong: `InverseGaussian`'s `mu` is positive despite the
# location-shaped name, `NormalInverseGaussian`'s `beta` is signed despite
# matching `_is_positive_param`, and `GeneralizedExtremeValue`/`SkewNormal`'s
# `shape` is signed despite matching `_is_positive_param` too.
@testitem "leaf_params: params_table reports the parameter's own support, not the variate's" setup=[
    LeafParamsFixture] begin
    tbl = params_table(compose((a = InverseGaussian(1.0, 1.0),
        b = NormalInverseGaussian(0.0, 1.0, 0.5, 1.0),
        c = GeneralizedExtremeValue(0.0, 1.0, 0.2),
        d = SkewNormal(0.0, 1.0, 0.5))))
    rows = Dict((e, p) => s for (e, p, s) in zip(tbl.edge, tbl.param, tbl.support))
    @test rows[(:a, :mu)] == (0.0, Inf)
    @test rows[(:b, :beta)] == (-Inf, Inf)
    @test rows[(:c, :shape)] == (-Inf, Inf)
    @test rows[(:d, :shape)] == (-Inf, Inf)
end

@testitem "leaf_params: build_priors is not misled by a name/domain mismatch" setup=[
    LeafParamsFixture] begin
    tbl = params_table(compose((a = InverseGaussian(1.0, 1.0),
        b = NormalInverseGaussian(0.0, 1.0, 0.5, 1.0),
        c = GeneralizedExtremeValue(0.0, 1.0, 0.2),
        d = SkewNormal(0.0, 1.0, 0.5))))
    priors = build_priors(tbl)
    # InverseGaussian's mu must stay positive: a location-shaped name, but a
    # positive-only domain.
    @test priors.a.mu isa Truncated
    # NormalInverseGaussian's beta is signed (|beta| < alpha): a
    # `_is_positive_param`-matching name, but an unconstrained domain.
    @test priors.b.beta isa Normal
    # GeneralizedExtremeValue's and SkewNormal's shape (xi/alpha) are signed:
    # `_is_positive_param`-matching names, but unconstrained domains.
    @test priors.c.shape isa Normal
    @test priors.d.shape isa Normal
end

@testitem "leaf_params: param_supports covers every registered family" setup=[
    LeafParamsFixture] begin
    for (T, spec) in LEAF_PARAM_SPECS
        d = _LEAF_PARAMS_TEST_INSTANCES[T]
        @test param_supports(d) == spec.supports
    end
end

@testitem "leaf_params: param_supports' generic fallback is empty" setup=[
    LeafParamsFixture] begin
    @test param_supports(1) == ()
    @test param_supports("not a distribution") == ()
end

@testitem "leaf_params: an unregistered leaf falls back to its variate support" setup=[
    LeafParamsFixture] begin
    # `OrderStatistic` has no `LEAF_PARAM_SPECS` entry (its `params` flattens
    # an arbitrary inner distribution's own parameters, #332), so
    # `param_supports` returns `()` and every native parameter row falls back
    # to the leaf's own variate support -- unchanged, pre-#332 behaviour.
    tree = compose((a = Distributions.OrderStatistic(
        Distributions.Uniform(0.0, 1.0), 5, 3),))
    tbl = params_table(tree)
    @test all(==((0.0, 1.0)), tbl.support)
end
