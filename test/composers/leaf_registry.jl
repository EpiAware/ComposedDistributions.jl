@testsnippet LeafRegistryFixture begin
    using Distributions
    using ComposedDistributions: register_leaf_wrapper!, _resolve_leaf_free_type,
                                 _resolve_extra_names, flat_dimension, unflatten,
                                 flatten, reconstruct

    struct FakeThinWrap{D <: Distributions.UnivariateDistribution} <:
           Distributions.UnivariateDistribution{Distributions.Continuous}
        dist::D
        factor::Float64
    end

    Distributions.logpdf(d::FakeThinWrap, x::Real) = Distributions.logpdf(d.dist, x)
    Base.minimum(d::FakeThinWrap) = minimum(d.dist)
    Base.maximum(d::FakeThinWrap) = maximum(d.dist)
    Distributions.insupport(d::FakeThinWrap, x::Real) = insupport(d.dist, x)
    Distributions.params(d::FakeThinWrap) = params(d.dist)
    Distributions.cdf(d::FakeThinWrap, x::Real) = Distributions.cdf(d.dist, x)
    Distributions.logcdf(d::FakeThinWrap, x::Real) = Distributions.logcdf(d.dist, x)

    ComposedDistributions.free_leaf(d::FakeThinWrap) = ComposedDistributions.free_leaf(d.dist)
    function ComposedDistributions.rewrap_leaf(d::FakeThinWrap, inner)
        return FakeThinWrap(ComposedDistributions.rewrap_leaf(d.dist, inner), d.factor)
    end
    function ComposedDistributions.extra_leaf_params(d::FakeThinWrap)
        return (fake_extra = (value = d.factor, support = (0.0, 1.0)),)
    end
    ComposedDistributions.set_extra_leaf_params(d::FakeThinWrap, ::NamedTuple{()}) = d
    function ComposedDistributions.set_extra_leaf_params(d::FakeThinWrap, vals::NamedTuple)
        return FakeThinWrap(d.dist, vals.fake_extra)
    end

    struct FakePassThrough{D <: Distributions.UnivariateDistribution} <:
           Distributions.UnivariateDistribution{Distributions.Continuous}
        dist::D
    end

    Distributions.logpdf(d::FakePassThrough, x::Real) = Distributions.logpdf(d.dist, x)
    Base.minimum(d::FakePassThrough) = minimum(d.dist)
    Base.maximum(d::FakePassThrough) = maximum(d.dist)
    Distributions.insupport(d::FakePassThrough, x::Real) = insupport(d.dist, x)
    Distributions.params(d::FakePassThrough) = params(d.dist)

    ComposedDistributions.free_leaf(d::FakePassThrough) = ComposedDistributions.free_leaf(d.dist)
    function ComposedDistributions.rewrap_leaf(d::FakePassThrough, inner)
        return FakePassThrough(ComposedDistributions.rewrap_leaf(d.dist, inner))
    end

    register_leaf_wrapper!(FakeThinWrap; free_index = 1, extra_names = (:fake_extra,))
    register_leaf_wrapper!(FakePassThrough; free_index = 1)
end

@testitem "leaf registry: resolves a registered wrapper's peel and extras" setup=[LeafRegistryFixture] begin
    @test _resolve_leaf_free_type(FakeThinWrap{Gamma{Float64}}) == Gamma{Float64}
    @test _resolve_extra_names(FakeThinWrap{Gamma{Float64}}) == (:fake_extra,)

    @test _resolve_leaf_free_type(FakePassThrough{Gamma{Float64}}) == Gamma{Float64}
    @test _resolve_extra_names(FakePassThrough{Gamma{Float64}}) == ()
end

@testitem "leaf registry: nested registered wrappers peel through every layer" setup=[LeafRegistryFixture] begin
    NestedT = FakePassThrough{FakeThinWrap{Gamma{Float64}}}
    @test _resolve_leaf_free_type(NestedT) == Gamma{Float64}
    @test _resolve_extra_names(NestedT) == (:fake_extra,)
end

@testitem "leaf registry: mixed registered + core (Truncated) nesting peels correctly" setup=[LeafRegistryFixture] begin
    using Distributions: Truncated
    TruncT = typeof(truncated(Gamma(2.0, 1.0); upper = 5.0))
    MixedT = FakeThinWrap{TruncT}
    @test _resolve_leaf_free_type(MixedT) == Gamma{Float64}
    @test _resolve_extra_names(MixedT) == (:fake_extra,)
end

@testitem "leaf registry: a CORE wrapper directly around a registered extension leaf peels correctly" setup=[LeafRegistryFixture] begin
    using Distributions: Truncated
    fake = FakeThinWrap(Gamma(2.0, 1.0), 0.3)
    TruncMixedT = typeof(truncated(fake; upper = 5.0))
    @test _resolve_leaf_free_type(TruncMixedT) == Gamma{Float64}
    @test _resolve_extra_names(TruncMixedT) == (:fake_extra,)
end

@testitem "leaf registry: an unregistered wrapper falls back to the identity (documented gap)" begin
    struct UnregisteredWrap{D}
        dist::D
    end
    @test ComposedDistributions._resolve_leaf_free_type(UnregisteredWrap{Float64}) ==
          UnregisteredWrap{Float64}
    @test ComposedDistributions._resolve_extra_names(UnregisteredWrap{Float64}) == ()
end

@testitem "leaf registry: flat_dimension/unflatten/flatten/reconstruct round-trip a registered wrapper in a real tree" setup=[LeafRegistryFixture] begin
    using ComposedDistributions: compose, uncertain, update

    leaf = uncertain(FakeThinWrap(Gamma(2.0, 1.0), 0.3);
        shape = LogNormal(log(2.0), 0.2), fake_extra = Beta(2.0, 2.0))
    tree = compose((onset = leaf, admit = LogNormal(0.5, 0.4)))

    @test flat_dimension(tree) == 2

    x = [3.0, 0.4]
    nt = unflatten(tree, x)
    @test nt.onset.shape == 3.0
    @test nt.onset.fake_extra == 0.4
    @test flatten(tree, nt) == x

    rebuilt = reconstruct(tree, x)
    @test rebuilt == update(tree, nt)
    onset_leaf = ComposedDistributions.event(rebuilt, :onset)
    @test ComposedDistributions.free_leaf(onset_leaf) == Gamma(3.0, 1.0)
    @test ComposedDistributions.extra_leaf_params(onset_leaf).fake_extra.value == 0.4
end

@testitem "leaf registry: re-registering the same pattern replaces the earlier entry" begin
    using ComposedDistributions: register_leaf_wrapper!, _resolve_leaf_free_type

    struct ReplaceableWrap{D}
        dist::D
    end

    register_leaf_wrapper!(ReplaceableWrap; free_index = 1)
    @test _resolve_leaf_free_type(ReplaceableWrap{Float64}) == Float64
    
    register_leaf_wrapper!(ReplaceableWrap; free_index = 1, extra_names = (:replaced,))
    @test ComposedDistributions._resolve_extra_names(ReplaceableWrap{Float64}) ==
          (:replaced,)
end
