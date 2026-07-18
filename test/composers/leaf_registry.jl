# The load-order-independent leaf-wrapper registry (#189, #178 PR 4):
# `register_leaf_wrapper!` plus the registry-first resolvers
# (`_resolve_leaf_free_type`/`_resolve_extra_names`) that
# `_leaf_type_param_names`/`_leaf_unflatten_expr` now read from instead of
# dispatching on `_leaf_free_type`/`_extra_names_of` directly at generation
# time. These tests exercise the mechanism with a SYNTHETIC wrapper family
# defined entirely within this test file, so they validate the registry in
# isolation from ModifiedDistributions' actual registration (covered by the
# cross-package test in `test/composers/modified_ext.jl` and by
# ModifiedDistributions' own test suite once it calls
# `register_leaf_wrapper!` from its extension's `__init__`).

@testsnippet LeafRegistryFixture begin
    using Distributions
    using ComposedDistributions: register_leaf_wrapper!, _resolve_leaf_free_type,
                                 _resolve_extra_names, flat_dimension, unflatten,
                                 flatten, reconstruct

    # A synthetic "thin-like" wrapper: peels to its inner delay, owns one
    # extra parameter of its own (mirroring ModifiedDistributions' `thin`),
    # and does NOT recurse into any further extras when it owns one (matching
    # `Transformed`/`ThinOp`'s instance-level short-circuit).
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

    # A synthetic pure pass-through wrapper (no extras of its own, always
    # peels through), mirroring Affine/Weighted/Modified's shape.
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

    # Registered BEFORE any tree containing either type is ever built, the
    # same ordering guarantee a real extension's `__init__` gives (see
    # codec_gen.jl's registry comment): by construction there is no window in
    # which the generated codec could be compiled against these types without
    # the registry already populated, since neither type exists in this file
    # before this point. Plain data only (a type-parameter index, a fixed
    # extra-names tuple) -- no callables, so nothing here is ever
    # `Base.invokelatest`-sensitive (see codec_gen.jl's registry comment for
    # why that matters).
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
    # FakePassThrough(FakeThinWrap(Gamma)): the outer pass-through has no
    # extras of its own, so extras peel through to the inner FakeThinWrap's
    # OWN extra (matching the `nothing` = "keep peeling" contract); the free
    # type peels through BOTH layers to the innermost Gamma.
    NestedT = FakePassThrough{FakeThinWrap{Gamma{Float64}}}
    @test _resolve_leaf_free_type(NestedT) == Gamma{Float64}
    @test _resolve_extra_names(NestedT) == (:fake_extra,)
end

@testitem "leaf registry: mixed registered + core (Truncated) nesting peels correctly" setup=[LeafRegistryFixture] begin
    using Distributions: Truncated

    # FakeThinWrap(Truncated(Gamma)): a registered extension wrapper around a
    # CORE (in-module) wrapper. `_leaf_free_type`'s existing Truncated method
    # (ordinary dispatch, no registry involved for that layer) peels the
    # Truncated; the resolver's own recursion then keeps going since the
    # result is not yet a fixed point. `typeof` on a real instance (not a
    # hand-written type-parameter tuple) keeps this robust to Truncated's own
    # internal parametrisation.
    TruncT = typeof(truncated(Gamma(2.0, 1.0); upper = 5.0))
    MixedT = FakeThinWrap{TruncT}
    @test _resolve_leaf_free_type(MixedT) == Gamma{Float64}
    @test _resolve_extra_names(MixedT) == (:fake_extra,)
end

@testitem "leaf registry: an unregistered wrapper falls back to the identity (documented gap)" begin
    # A leaf type that is neither core nor registered: the resolver falls
    # back to the generic `_leaf_free_type` catch-all (`L` unpeeled), matching
    # the existing documented safety-net behaviour (an un-peeled fallback,
    # not silent corruption -- see the `#188` test in this directory).
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

    # Two estimated parameters: onset.shape, onset.fake_extra -- exactly the
    # #188 test's shape, but through the NEW registry mechanism instead of
    # the old (never-implemented) direct-dispatch hooks.
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

    # Re-registering with a DIFFERENT free_index is nonsensical for THIS
    # type's single type parameter, so prove replacement with a distinct
    # observable instead: an extra_names change (deliberately wrong, to prove
    # the second call wins rather than accumulating a duplicate first match).
    register_leaf_wrapper!(ReplaceableWrap; free_index = 1, extra_names = (:replaced,))
    @test ComposedDistributions._resolve_extra_names(ReplaceableWrap{Float64}) ==
          (:replaced,)
end
