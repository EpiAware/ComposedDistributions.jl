# `@uncertain` macro tests: it rewrites a distribution-valued constructor
# argument into the positional `uncertain` family form, composes through
# `compose`/the verbs, leaves all-literal constructors alone, and wraps
# correctly under a ModifiedDistributions modifier.

@testitem "@uncertain: positional rewrite equals uncertain(D, ...)" begin
    using Distributions

    u = @uncertain LogNormal(Normal(0.0, 1.0), 0.5)
    @test u == uncertain(LogNormal, Normal(0.0, 1.0), 0.5)
    @test u isa Uncertain
end

@testitem "@uncertain: one distribution arg marks that parameter" begin
    using Distributions

    u = @uncertain Gamma(Normal(0.7, 0.2), 1.0)
    @test u == uncertain(Gamma, Normal(0.7, 0.2), 1.0)

    # shape uncertain (a prior), scale fixed at 1.0.
    @test keys(u.specs) == (:shape,)
    @test params(u.template)[2] == 1.0

    tree = compose((onset = u, admit = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    shape_idx = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :shape, eachindex(tbl.edge))
    scale_idx = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :scale, eachindex(tbl.edge))
    @test tbl.prior[shape_idx] == Normal(0.7, 0.2)
    @test tbl.prior[scale_idx] === nothing

    priors = build_priors(tbl)
    @test priors.onset.shape == Normal(0.7, 0.2)
end

@testitem "@uncertain: rewrites leaves inside a composed tree" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    tree = @uncertain compose((
        onset = Gamma(LogNormal(log(2.0), 0.2), 1.0),
        admit = LogNormal(0.5, 0.4)))

    explicit = compose((
        onset = uncertain(Gamma, LogNormal(log(2.0), 0.2), 1.0),
        admit = LogNormal(0.5, 0.4)))
    @test tree == explicit

    # onset uncertain (one estimated param), admit fully fixed.
    @test flat_dimension(tree) == 1
    tbl = params_table(tree)
    onset_shape = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :shape, eachindex(tbl.edge))
    @test tbl.prior[onset_shape] == LogNormal(log(2.0), 0.2)
    admit_rows = findall(i -> tbl.edge[i] == :admit, eachindex(tbl.edge))
    @test all(tbl.prior[i] === nothing for i in admit_rows)
end

@testitem "@uncertain: all-literal constructor is left unchanged" begin
    using Distributions

    @test (@uncertain LogNormal(0.5, 0.4)) == LogNormal(0.5, 0.4)
    @test (@uncertain Gamma(2.0, 1.0)) == Gamma(2.0, 1.0)
    @test !(@uncertain LogNormal(0.5, 0.4) isa Uncertain)
end

@testitem "@uncertain: wraps under a ModifiedDistributions modifier" begin
    using Distributions
    using ModifiedDistributions
    using ModifiedDistributions: affine, get_dist
    using ComposedDistributions: free_leaf

    aff = @uncertain affine(Gamma(Normal(0.7, 0.2), 1.0); scale = 2.0)

    # The affine directly wraps the rewritten uncertain leaf.
    inner = get_dist(aff)
    @test inner isa Uncertain
    @test keys(inner.specs) == (:shape,)
    # free_leaf peels through both the modifier and the uncertainty to the
    # concrete template (its documented job), so the modifier survives an
    # uncertain leaf.
    @test free_leaf(aff) == Gamma(1.0, 1.0)

    # params_table reports the inner shape as uncertain (its spec on the row).
    tree = compose((onset = aff, admit = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    shape_idx = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :shape, eachindex(tbl.edge))
    @test tbl.prior[shape_idx] == Normal(0.7, 0.2)
end
