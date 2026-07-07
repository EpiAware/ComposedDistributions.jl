# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Per-backend AD gradient test items. Each backend is its own `@testitem`,
# tagged so the per-backend CI can select it with a tag filter (e.g.
# `julia test/ad/runtests.jl enzyme_reverse`). The harness wiring lives in the
# managed `setup.jl`; the SCENARIOS come from the package's own `ADFixtures`
# registry. Add/trim backends and categories to match the package.

@testitem "ForwardDiff gradients (marginal)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff")
end

@testitem "ReverseDiff gradients (marginal)" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)")
end

@testitem "Enzyme reverse gradients (marginal)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse")
end

@testitem "Mooncake reverse gradients (marginal)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse")
end

# Add latent (or other) scenario groups as the package needs, e.g.:
# @testitem "ForwardDiff gradients (latent)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
#     test_working_backend("ForwardDiff"; category = :latent)
# end

# `_ctor_has_check_args` (src/composers/introspection.jl) is not yet called
# from any scored path in this package — it is dormant reflection for a
# future leaf reconstruction (the DynamicPPL composer-half extension, issue
# #9) — so it never appears inside the `scenarios()` above. This item drives
# it directly through Mooncake reverse to prove
# `ComposedDistributionsMooncakeExt`'s `@zero_adjoint` shield holds ahead of
# that caller landing: without the shield, Mooncake reverse errors tracing
# `hasmethod`'s internals.
@testitem "Mooncake extension: _ctor_has_check_args is AD-safe" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using ADTypes: AutoMooncake
    using ComposedDistributions
    using ComposedDistributions: _ctor_has_check_args
    using DifferentiationInterface: gradient
    using Distributions: Gamma
    using Mooncake

    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsMooncakeExt) !== nothing

    f(θ) = (_ctor_has_check_args(Gamma, (θ[1], θ[2])) ? 1.0 : 0.0) *
           sum(abs2, θ)
    θ0 = [2.0, 1.5]
    g = gradient(f, AutoMooncake(config = nothing), θ0)
    @test g ≈ 2 .* θ0
end

# Constructing a `Resolve` / `Compete` from a heterogeneous outcome tuple must be
# differentiable under Enzyme: the verb / struct constructors build their tuples
# with `map`, not a `Tuple(gen)` comprehension. A generator-collect lowers to
# `collect_to!` building a non-concrete `Array` temporary Enzyme's type analysis
# rejects (`IllegalTypeAnalysisException`), so differentiating through the
# construction proves the `map` form holds (finding C8). The gradient matches
# ForwardDiff.
@testitem "Enzyme differentiates Resolve/Compete construction (#96)" tags=[
    :ad, :enzyme, :enzyme_reverse] begin
    using ADTypes: AutoEnzyme, AutoForwardDiff
    using ComposedDistributions
    using DifferentiationInterface: gradient
    using Distributions: Gamma, logpdf
    using Enzyme, ForwardDiff

    enzyme = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse))
    θ0 = [1.5, 1.0, 2.0, 1.5]

    # A Resolve built (map-constructed) from θ-parameterised delays, scored
    # through its mixture marginal.
    fresolve(θ) = logpdf(
        as_mixture(resolve(:a => (Gamma(θ[1], θ[2]), 0.3),
            :b => (Gamma(θ[3], θ[4]), 0.7))), 2.0)
    @test gradient(fresolve, enzyme, θ0) ≈
          gradient(fresolve, AutoForwardDiff(), θ0)

    # A Compete built from θ-parameterised racing delays, scored at a time.
    fcompete(θ) = logpdf(
        compete(:a => Gamma(θ[1], θ[2]), :b => Gamma(θ[3], θ[4])), 2.0)
    @test gradient(fcompete, enzyme, θ0) ≈
          gradient(fcompete, AutoForwardDiff(), θ0)
end
