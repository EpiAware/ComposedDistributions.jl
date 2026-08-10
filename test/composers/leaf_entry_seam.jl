# Runtime leaf seam (`leaf_param_values`/`_leaf_entry`, introspection.jl):
# STAGE S1 of the codec-generation plan that replaces `codec_gen.jl`'s
# type-level leaf table with instance-level generic calls, matching what
# `_update`'s leaf case already does. This stage adds the seam only; the
# generated codec is untouched, so these tests exercise the new functions
# directly and cross-check `_leaf_entry` against the real `unflatten` output
# wherever a tree already covers the same leaf shape.

@testsnippet LeafEntrySeamFixture begin
    using Distributions
    import ComposedDistributions: leaf_param_values, leaf_param_names,
                                  _leaf_entry, free_leaf, extra_leaf_params

    # A leaf-wrapper type carrying one modifier-owned extra parameter,
    # mirroring ModifiedDistributions' `ThinOp` (the motivating case for
    # `extra_leaf_params`) without depending on that package.
    struct FakeThinLeaf{D <: Distributions.UnivariateDistribution} <:
           Distributions.UnivariateDistribution{Distributions.Continuous}
        dist::D
        factor::Float64
    end
    Distributions.params(d::FakeThinLeaf) = params(d.dist)
    Distributions.logpdf(d::FakeThinLeaf, x::Real) = logpdf(d.dist, x)
    Base.minimum(d::FakeThinLeaf) = minimum(d.dist)
    Base.maximum(d::FakeThinLeaf) = maximum(d.dist)
    free_leaf(d::FakeThinLeaf) = free_leaf(d.dist)
    function ComposedDistributions.extra_leaf_params(d::FakeThinLeaf)
        return (thin = (value = d.factor, support = (0.0, 1.0)),)
    end
end

@testitem "leaf_param_values: plain leaf matches params positionally" setup=[
    LeafEntrySeamFixture] begin
    leaf = Gamma(2.0, 1.0)
    @test leaf_param_values(leaf) == params(leaf) == (2.0, 1.0)
    @test length(leaf_param_values(leaf)) == length(leaf_param_names(leaf))
end

@testitem "leaf_param_values: appends extra params after native ones" setup=[
    LeafEntrySeamFixture] begin
    leaf = FakeThinLeaf(Gamma(2.0, 1.0), 0.4)
    @test leaf_param_names(leaf) == (:alpha, :theta, :thin)
    @test leaf_param_values(leaf) == (2.0, 1.0, 0.4)
end

@testitem "leaf_param_values: a truncated/censored leaf peels to the free \
delay" setup=[LeafEntrySeamFixture] begin
    trunc_leaf = truncated(Gamma(2.0, 1.0); upper = 10.0)
    @test leaf_param_values(trunc_leaf) == (2.0, 1.0)

    cens_leaf = censored(Gamma(2.0, 1.0); upper = 10.0)
    @test leaf_param_values(cens_leaf) == (2.0, 1.0)
end

@testitem "_leaf_entry: no spec keys reproduces the leaf's own values" setup=[
    LeafEntrySeamFixture] begin
    leaf = Gamma(2.0, 1.0)
    entry = _leaf_entry(leaf, Val(()), ())
    @test entry == (alpha = 2.0, theta = 1.0)
end

@testitem "_leaf_entry: substitutes only the named slots" setup=[
    LeafEntrySeamFixture] begin
    leaf = Gamma(2.0, 1.0)
    entry = _leaf_entry(leaf, Val((:alpha,)), (5.0,))
    @test entry == (alpha = 5.0, theta = 1.0)

    entry2 = _leaf_entry(leaf, Val((:theta,)), (9.0,))
    @test entry2 == (alpha = 2.0, theta = 9.0)
end

@testitem "_leaf_entry: extra-param leaf substitutes the extra slot" setup=[
    LeafEntrySeamFixture] begin
    leaf = FakeThinLeaf(Gamma(2.0, 1.0), 0.4)
    entry = _leaf_entry(leaf, Val((:thin,)), (0.9,))
    @test entry == (alpha = 2.0, theta = 1.0, thin = 0.9)

    entry2 = _leaf_entry(leaf, Val((:alpha, :thin)), (7.0, 0.1))
    @test entry2 == (alpha = 7.0, theta = 1.0, thin = 0.1)
end

@testitem "_leaf_entry: a truncated/censored wrapped leaf substitutes the \
    peeled free delay's slot" setup=[LeafEntrySeamFixture] begin
    trunc_leaf = truncated(Gamma(2.0, 1.0); upper = 10.0)
    entry = _leaf_entry(trunc_leaf, Val((:alpha,)), (5.0,))
    @test entry == (alpha = 5.0, theta = 1.0)

    cens_leaf = censored(Gamma(2.0, 1.0); upper = 10.0)
    entry2 = _leaf_entry(cens_leaf, Val((:theta,)), (4.0,))
    @test entry2 == (alpha = 2.0, theta = 4.0)
end

@testitem "_leaf_entry: slots are consumed in leaf_param_names order, not \
    speckeys' own (kwargs) order -- matches the generated codec" setup=[
    LeafEntrySeamFixture] begin
    using ComposedDistributions: unflatten

    # `leaf_param_names(Gamma(...))` is `(:alpha, :theta)`, but the kwargs
    # here are given `theta` first: `keys(specs)` (the spec-key order the
    # `Uncertain` node stores) is therefore `(:theta, :alpha)`, the reverse of
    # canonical order.
    est = uncertain(Gamma(2.0, 1.0); theta = LogNormal(0.0, 0.3),
        alpha = LogNormal(0.0, 0.3))
    tree = compose((g = est,))
    x = [10.0, 20.0]
    gen = unflatten(tree, x)

    speckeys = keys(est.specs)
    @test speckeys == (:theta, :alpha)
    entry = _leaf_entry(est.template, Val(speckeys), (x[1], x[2]))
    @test entry == gen.g == (alpha = 10.0, theta = 20.0)
end

@testitem "_leaf_entry: agrees with unflatten on a wrapped leaf inside a \
    tree" setup=[LeafEntrySeamFixture] begin
    using ComposedDistributions: unflatten

    est = uncertain(censored(Gamma(2.0, 3.0); upper = 10.0);
        alpha = LogNormal(log(2.0), 0.2))
    tree = compose((onset = est, death = LogNormal(0.5, 0.4)))
    x = [2.5]
    gen = unflatten(tree, x)

    speckeys = keys(est.specs)
    entry = _leaf_entry(est.template, Val(speckeys), (x[1],))
    @test entry == gen.onset == (alpha = 2.5, theta = 3.0)
end
