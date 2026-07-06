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
