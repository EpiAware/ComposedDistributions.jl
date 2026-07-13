# ComposedDistributions × LogDensityProblems extension: a ComposedLogDensity
# assembled by as_logdensity satisfies the LogDensityProblems interface
# (dimension, logdensity, zeroth-order capability) and takes a gradient through
# LogDensityProblemsAD, so a composed posterior is sampleable by any
# LogDensityProblems consumer (#13).

@testitem "LogDensityProblems extension: interface conformance" begin
    using Distributions
    using LogDensityProblems

    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsLogDensityProblemsExt) !== nothing

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0]]
    prob = ComposedDistributions.as_logdensity(tree, data)

    # One estimated parameter (onset_admit.shape); dimension mirrors the codec.
    @test LogDensityProblems.dimension(prob) ==
          ComposedDistributions.flat_dimension(tree) == 1
    @test LogDensityProblems.capabilities(typeof(prob)) ==
          LogDensityProblems.LogDensityOrder{0}()

    # The interface method is exactly the codec's evaluator.
    x = [2.0]
    @test LogDensityProblems.logdensity(prob, x) ==
          ComposedDistributions.logdensity(prob, x)
    @test isfinite(LogDensityProblems.logdensity(prob, x))

    # A tree with no uncertain leaves estimates nothing: dimension zero.
    fixed = compose((a = Gamma(2.0, 1.0), b = LogNormal(0.5, 0.4)))
    probf = ComposedDistributions.as_logdensity(fixed, data)
    @test LogDensityProblems.dimension(probf) == 0
    @test isfinite(LogDensityProblems.logdensity(probf, Float64[]))
end

@testitem "LogDensityProblems extension: gradient via LogDensityProblemsAD" begin
    using Distributions
    using LogDensityProblems
    using LogDensityProblemsAD
    using ForwardDiff

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0]]
    prob = ComposedDistributions.as_logdensity(tree, data)

    ∇prob = ADgradient(:ForwardDiff, prob)
    x = [2.0]
    value, grad = LogDensityProblems.logdensity_and_gradient(∇prob, x)

    @test value ≈ ComposedDistributions.logdensity(prob, x)
    @test length(grad) == 1
    @test all(isfinite, grad)
    # The wrapped gradient matches a direct ForwardDiff of the codec density.
    direct = ForwardDiff.gradient(z -> ComposedDistributions.logdensity(prob, z), x)
    @test grad ≈ direct
end
