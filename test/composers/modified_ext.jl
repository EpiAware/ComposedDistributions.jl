# ComposedDistributions × ModifiedDistributions extension: a modified leaf peels
# correctly inside a composed tree (free_leaf reaches the inner delay,
# rewrap_leaf rebuilds the modifier, _shared_tag sees through it), and a
# thin(...) reporting probability surfaces as a free `:thin` parameter through
# the core's _thin_factor / _set_thin_factor hooks.

@testitem "Modified extension: free_leaf / rewrap_leaf / _shared_tag" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, weight, thin, modify, get_dist

    # The extension loads once ModifiedDistributions is present.
    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsModifiedDistributionsExt) !== nothing

    inner = Gamma(2.0, 1.0)
    aff = affine(inner; scale = 2.0, shift = 1.0)
    wt = weight(inner, 0.5)
    th = thin(inner, 0.3)
    mod = modify(inner, 0.5)

    # free_leaf reaches the inner free delay through each modifier.
    @test ComposedDistributions.free_leaf(aff) == inner
    @test ComposedDistributions.free_leaf(wt) == inner
    @test ComposedDistributions.free_leaf(th) == inner
    @test ComposedDistributions.free_leaf(mod) == inner

    # rewrap_leaf rebuilds the modifier around a new inner delay.
    new_inner = Gamma(3.0, 1.5)
    raff = ComposedDistributions.rewrap_leaf(aff, new_inner)
    @test ComposedDistributions.free_leaf(raff) == new_inner
    @test get_dist(raff) == new_inner
    rmod = ComposedDistributions.rewrap_leaf(mod, new_inner)
    @test rmod isa ModifiedDistributions.Modified
    @test ComposedDistributions.free_leaf(rmod) == new_inner
    @test rmod.effect == mod.effect
    @test rmod.link == mod.link

    # A shared tag is visible through a modifier wrapper. (`Shared`'s declared
    # value-support is generic `ValueSupport`, so it cannot itself be wrapped
    # in a `Modified` — that constructor requires a `Continuous`-typed inner
    # distribution — but `_shared_tag`/`free_leaf` peel through `Modified`
    # just as they do the other modifiers, tested above via `mod`.)
    tagged = ComposedDistributions.shared(:inc, inner)
    @test ComposedDistributions._shared_tag(tagged) == :inc
    @test get_dist(tagged) == inner
end

@testitem "Modified extension: params_table peels a modified leaf" begin
    using Distributions
    using ModifiedDistributions: affine, modify

    # A composed tree with an affine-modified leaf reports only the inner free
    # delay's parameters (the affine scale/shift are fixed structure).
    tree = compose((onset_admit = affine(Gamma(2.0, 1.0); scale = 2.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    @test tbl.param == [:shape, :scale, :mu, :sigma]

    # Likewise a hazard-modified leaf (the effect/link are fixed structure).
    mtree = compose((onset_admit = modify(Gamma(2.0, 1.0), 0.5),
        admit_death = LogNormal(0.5, 0.4)))
    mtbl = params_table(mtree)
    @test mtbl.param == [:shape, :scale, :mu, :sigma]
end

@testitem "Modified extension: thin factor surfaces through params_table" begin
    using Distributions
    using ModifiedDistributions: thin
    import ComposedDistributions: _thin_factor

    # thin(leaf, p) is a Transformed carrying a ThinOp; the core's thin hooks
    # now surface its reporting probability as a free `:thin` row, mirroring
    # CensoredDistributions' forward_transform.jl precedent.
    tree = compose((inc = Gamma(2.0, 1.0), cases = thin(LogNormal(1.5, 0.4), 0.3)))
    tbl = params_table(tree)
    @test tbl.param == [:shape, :scale, :mu, :sigma, :thin]

    thin_row = only(filter(i -> tbl.param[i] == :thin, eachindex(tbl.param)))
    @test tbl.edge[thin_row] === :cases
    @test tbl.value[thin_row] == 0.3
    @test tbl.support[thin_row] == (0.0, 1.0)

    leaf = event(tree, :cases)
    @test _thin_factor(leaf) == 0.3
end

@testitem "Modified extension: thin factor round-trips through update" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: thin, affine, get_dist
    import ComposedDistributions: _set_thin_factor

    tree = compose((cases = thin(LogNormal(1.5, 0.4), 0.3),))

    # update re-routes a new thin weight into the ThinOp, keeping the inner
    # delay params, and the new weight surfaces back through params_table.
    updated = update(tree, (cases = (mu = 1.0, sigma = 0.5, thin = 0.6),))
    leaf = event(updated, :cases)
    @test leaf isa ModifiedDistributions.Transformed
    @test leaf.op isa ModifiedDistributions.ThinOp
    @test leaf.op.factor == 0.6
    @test get_dist(leaf) == LogNormal(1.0, 0.5)

    rt = params_table(updated)
    @test rt.value[findfirst(==(:thin), rt.param)] == 0.6

    # A thinned leaf wrapped in a fixed-structure modifier (affine) still
    # peels through to the same ThinOp, so _set_thin_factor reaches it.
    wrapped = affine(thin(Gamma(2.0, 1.0), 0.4); scale = 2.0)
    reset = _set_thin_factor(wrapped, 0.9)
    @test reset isa ModifiedDistributions.Affine
    @test reset.dist.op.factor == 0.9
end

@testitem "Modified extension: uncertain specs seen through a modifier" begin
    using Distributions, Random
    using ModifiedDistributions: affine, modify

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    au = affine(u; scale = 2.0, shift = 0.0)

    # The spec protocol sees through the modifier, so the uncertainty is
    # visible to the routing predicate and the prior column.
    @test ComposedDistributions._uncertain_specs(au) == u.specs
    @test has_uncertain(au)

    # Likewise through a hazard-modified leaf.
    mu = modify(u, 0.5)
    @test ComposedDistributions._uncertain_specs(mu) == u.specs
    @test has_uncertain(mu)

    # The marginal `rand` draws through the modifier (a fresh parameter each
    # call), and `update` collapses the wrapped uncertainty, keeping the
    # modifier's fixed structure.
    tree = compose((onset_admit = au,))
    @test all(isfinite, values(rand(Xoshiro(1), tree)))
    collapsed = update(tree, (onset_admit = (shape = 3.0, scale = 1.0),))
    leaf = event(collapsed, :onset_admit)
    @test !has_uncertain(leaf)
    @test ComposedDistributions.free_leaf(leaf) == Gamma(3.0, 1.0)
end

@testitem "Modified extension: a Modified leaf reports no thin factor" begin
    using Distributions
    using ModifiedDistributions: modify
    import ComposedDistributions: _thin_factor, _set_thin_factor

    # `Modified` can only wrap a `Continuous`-typed inner distribution (its own
    # constructor's restriction), and `Transformed`/`Weighted` declare a
    # generic `ValueSupport` rather than propagating their wrapped delay's
    # concrete support, so a `Modified` can never itself sit around a thinned
    # delay via the public constructors today. The peel-through methods added
    # here are future-proofing (symmetry with Affine/Weighted) for when that
    # changes; meanwhile a bare Modified leaf reports no thin factor, same as
    # any other non-thinned leaf.
    mod = modify(Gamma(2.0, 1.0), 0.5)
    @test _thin_factor(mod) === nothing

    reset = _set_thin_factor(mod, 0.4)
    @test reset isa typeof(mod)
    @test _thin_factor(reset) === nothing
    @test reset.dist == mod.dist
    @test reset.effect == mod.effect
end
