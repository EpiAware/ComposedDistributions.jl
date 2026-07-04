# ComposedDistributions × ModifiedDistributions extension: a modified leaf peels
# correctly inside a composed tree (free_leaf reaches the inner delay,
# rewrap_leaf rebuilds the modifier, _shared_tag sees through it).

@testitem "Modified extension: free_leaf / rewrap_leaf / _shared_tag" begin
    using Distributions
    using ModifiedDistributions: affine, weight, thin, get_dist

    # The extension loads once ModifiedDistributions is present.
    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsModifiedDistributionsExt) !== nothing

    inner = Gamma(2.0, 1.0)
    aff = affine(inner; scale = 2.0, shift = 1.0)
    wt = weight(inner, 0.5)
    th = thin(inner, 0.3)

    # free_leaf reaches the inner free delay through each modifier.
    @test ComposedDistributions.free_leaf(aff) == inner
    @test ComposedDistributions.free_leaf(wt) == inner
    @test ComposedDistributions.free_leaf(th) == inner

    # rewrap_leaf rebuilds the modifier around a new inner delay.
    new_inner = Gamma(3.0, 1.5)
    raff = ComposedDistributions.rewrap_leaf(aff, new_inner)
    @test ComposedDistributions.free_leaf(raff) == new_inner
    @test get_dist(raff) == new_inner

    # A shared tag is visible through a modifier wrapper.
    tagged = ComposedDistributions.shared(:inc, inner)
    @test ComposedDistributions._shared_tag(tagged) == :inc
    @test get_dist(tagged) == inner
end

@testitem "Modified extension: params_table peels a modified leaf" begin
    using Distributions
    using ModifiedDistributions: affine

    # A composed tree with an affine-modified leaf reports only the inner free
    # delay's parameters (the affine scale/shift are fixed structure).
    tree = compose((onset_admit = affine(Gamma(2.0, 1.0); scale = 2.0),
        admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    @test tbl.param == [:shape, :scale, :mu, :sigma]
end
