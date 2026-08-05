# Tests for the generic, instance-level `param_names` derivation
# (introspection.jl) that replaced the curated six-family table: a leaf's
# names are read off `fieldnames(typeof(leaf))` by matching `params(leaf)`
# against the field values, order-preserving (both walked in their own
# declared order) and unique. See docs/src/developer/leaf-protocol.md.

@testitem "param_names derives names for representative families" begin
    using ComposedDistributions: param_names
    using Distributions

    # Already-ASCII fields (Normal-family mu/sigma; Uniform's own a/b).
    @test param_names(Normal(0.0, 1.0)) == (:mu, :sigma)
    @test param_names(LogNormal(0.0, 1.0)) == (:mu, :sigma)
    @test param_names(Uniform(0.0, 1.0)) == (:a, :b)

    # Greek field names, transliterated.
    @test param_names(Gamma(2.0, 1.0)) == (:alpha, :theta)
    @test param_names(Weibull(1.0, 1.0)) == (:alpha, :theta)
    @test param_names(Exponential(1.0)) == (:theta,)
    @test param_names(Beta(2.0, 3.0)) == (:alpha, :beta)
    @test param_names(Cauchy(0.0, 1.0)) == (:mu, :sigma)

    # More fields than params (an internal cache field, e.g. VonMises'
    # `I0κx`, NormalCanon's redundant `mu`): still a unique, order-preserving
    # match since the extra field cannot extend a full assignment.
    @test param_names(VonMises(0.5, 2.0)) == (:mu, :kappa)
    @test param_names(NormalCanon(2.0, 3.0)) == (:eta, :lambda)
end

@testitem "param_names Gamma(2.0, 2.0) tie still resolves to (alpha, theta)" begin
    using ComposedDistributions: param_names
    using Distributions

    # Both fields (α, θ) are numerically 2.0, so a naive per-value match is
    # ambiguous; the order-preserving constraint (field index must increase
    # along both `params` and `fieldnames` order) still admits only one full
    # assignment, so this does not fall back to positional. Pinned behaviour.
    @test param_names(Gamma(2.0, 2.0)) == (:alpha, :theta)
    @test param_names(Weibull(1.0, 1.0)) == (:alpha, :theta)
end

@testitem "param_names falls back to positional when values are not fields" begin
    using ComposedDistributions: param_names, leaf_param_names
    using Distributions

    # InverseGamma's shape is not a struct field (it wraps an inner Gamma
    # whose own alpha is not numerically InverseGamma's shape value; the
    # field holding the reciprocal scale is not a value match either), so no
    # full assignment exists and this falls back to `()`, letting
    # `leaf_param_names` pad positionally.
    ig = InverseGamma(3.0, 2.0)
    @test param_names(ig) == ()
    @test leaf_param_names(ig) == (:param_1, :param_2)
end

@testitem "param_names never errors for a leaf with no params() method" begin
    using ComposedDistributions: param_names
    using Distributions

    struct NoParamsLeaf <: Distributions.ContinuousUnivariateDistribution end
    @test param_names(NoParamsLeaf()) == ()
end

@testitem "an explicit param_names override still wins" begin
    using ComposedDistributions
    using Distributions

    struct OverrideLeaf <: Distributions.ContinuousUnivariateDistribution
        vals::Tuple{Float64, Float64}
    end
    Distributions.params(d::OverrideLeaf) = d.vals
    Distributions.logpdf(d::OverrideLeaf, x::Real) = logpdf(
        Normal(d.vals[1], d.vals[2]), x)
    Base.minimum(::OverrideLeaf) = -Inf
    Base.maximum(::OverrideLeaf) = Inf
    ComposedDistributions.param_names(::OverrideLeaf) = (:mean, :sd)
    ComposedDistributions.leaf_ctor(::OverrideLeaf) = (a, b) -> OverrideLeaf((a, b))

    # `vals` is a single Tuple field, so the generic derivation could never
    # produce a scalar-per-parameter match anyway; the point of this test is
    # that dispatch picks the explicit method, not the generic fallback.
    @test ComposedDistributions.param_names(OverrideLeaf((0.0, 1.0))) ==
          (:mean, :sd)
    @test ComposedDistributions.leaf_param_names(OverrideLeaf((0.0, 1.0))) ==
          (:mean, :sd)
end

@testitem "uncertain accepts the Greek spelling of a derived name" begin
    using ComposedDistributions
    using Distributions

    ascii = uncertain(Normal(0.0, 1.0); mu = Normal(0.0, 0.5))
    greek = uncertain(Normal(0.0, 1.0); μ = Normal(0.0, 0.5))
    @test ascii.specs == greek.specs
    @test keys(greek.specs) == (:mu,)

    # A Real value under the Greek spelling re-pins, same as the ASCII one.
    pinned = uncertain(Normal(0.0, 1.0); μ = 3.0, σ = Normal(0.0, 0.5))
    @test params(pinned.template) == (3.0, 1.0)
    @test keys(pinned.specs) == (:sigma,)
end

@testitem "update accepts the Greek spelling of a derived name" begin
    using ComposedDistributions
    using ComposedDistributions: update
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = Normal(0.0, 1.0)))

    # Strict update: every leaf's full parameter set, ASCII vs Greek spelling
    # for the Normal leaf only (Gamma's alpha/theta held fixed either way).
    ascii = update(tree, (onset_admit = (alpha = 2.0, theta = 1.0),
        admit_death = (mu = 2.0, sigma = 0.5)))
    greek = update(tree, (onset_admit = (α = 2.0, θ = 1.0),
        admit_death = (μ = 2.0, σ = 0.5)))
    @test event(ascii, :admit_death) == event(greek, :admit_death)
    @test event(ascii, :onset_admit) == event(greek, :onset_admit)

    promoted = uncertain(tree; onset_admit = (alpha = LogNormal(log(2.0), 0.2),))
    promoted_greek = uncertain(tree;
        onset_admit = (α = LogNormal(log(2.0), 0.2),))
    @test has_uncertain(promoted)
    @test event(promoted, :onset_admit).specs ==
          event(promoted_greek, :onset_admit).specs
end
