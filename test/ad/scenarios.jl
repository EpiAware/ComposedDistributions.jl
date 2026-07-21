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

# Latent scenario group: the full `as_logdensity`/`logdensity` codec path (an
# uncertain-leaf tree and a centred pool), run across the same four backends.
@testitem "ForwardDiff gradients (latent)" tags=[:ad, :forwarddiff] setup=[ADHelpers] begin
    test_working_backend("ForwardDiff"; category = :latent)
end

@testitem "ReverseDiff gradients (latent)" tags=[:ad, :reversediff] setup=[ADHelpers] begin
    test_working_backend("ReverseDiff (tape)"; category = :latent)
end

@testitem "Enzyme reverse gradients (latent)" tags=[:ad, :enzyme, :enzyme_reverse] setup=[ADHelpers] begin
    test_working_backend("Enzyme reverse"; category = :latent)
end

@testitem "Mooncake reverse gradients (latent)" tags=[:ad, :mooncake, :mooncake_reverse] setup=[ADHelpers] begin
    test_working_backend("Mooncake reverse"; category = :latent)
end

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

# `logdensity`/`unflatten` (`src/composers/logdensity.jl`) must differentiate
# under Mooncake, both reverse and forward: `unflatten` calls `_split_edge`
# unconditionally on every row, and `_split_edge`/the length guards'
# `DimensionMismatch` messages both recurse into Base's UTF-8
# string-indexing continuation machinery, for which Mooncake's whole-program
# rule derivation has no rule (a `sub_ptr` intrinsic), fixing issue #146.
# This tree has no shared/pooled parameters, so it does not touch the separate
# Mooncake-reverse wrong-gradient issue on pooled reconstructions (#99).
@testitem "Mooncake differentiates logdensity/unflatten past the length guard (#146)" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using ADTypes: AutoMooncake, AutoMooncakeForward, AutoForwardDiff
    using ComposedDistributions
    using ComposedDistributions: as_logdensity, logdensity
    using DifferentiationInterface: gradient
    using Distributions: Gamma, LogNormal
    using ForwardDiff, Mooncake

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0]]
    prob = as_logdensity(tree, data)
    f(x) = logdensity(prob, x)
    θ0 = [2.0]

    gref = gradient(f, AutoForwardDiff(), θ0)

    grev = gradient(f, AutoMooncake(config = nothing), θ0)
    @test grev ≈ gref

    gfwd = gradient(f, AutoMooncakeForward(), θ0)
    @test gfwd ≈ gref
end

# A Gamma-family shape landing exactly on `1.0` routes a nonzero cotangent into
# `LogExpFunctions.xlogy`'s `iszero(x)` branch inside `Distributions.gammalogpdf`
# (`xlogy(shape - 1, x / scale)` with `shape - 1 == 0`). Mooncake has no rule for
# `xlogy`/`xlog1py` and derives a wrong (zero) gradient there instead of the
# correct `log(y)` (#99, upstream chalk-lab/Mooncake.jl#1241).
# `ComposedDistributionsMooncakeExt` imports the ChainRulesCore rules for both
# functions. Importing via `@from_chainrules` (both AD directions) rather than
# `@from_rrule` (reverse only) closes the forward-mode gap too, so this exercises
# `shape == 1.0` under Mooncake forward as well as reverse (#214).
@testitem "Mooncake differentiates a Gamma logpdf at shape == 1.0, both directions (#214)" tags=[
    :ad, :mooncake, :mooncake_reverse] begin
    using ADTypes: AutoMooncake, AutoMooncakeForward, AutoForwardDiff
    using DifferentiationInterface: gradient
    using Distributions: Gamma, logpdf
    using ForwardDiff, Mooncake

    obs = [0.5, 1.2, 2.5, 3.8, 5.1]
    # `θ[1]` is the Gamma shape, evaluated at exactly `1.0`, so the reverse pass
    # sends a nonzero cotangent through `xlogy(0.0, x)`.
    f(θ) = sum(x -> logpdf(Gamma(θ[1], 1.0), x), obs)
    θ0 = [1.0]

    gref = gradient(f, AutoForwardDiff(), θ0)

    grev = gradient(f, AutoMooncake(config = nothing), θ0)
    @test grev ≈ gref

    gfwd = gradient(f, AutoMooncakeForward(), θ0)
    @test gfwd ≈ gref
end
