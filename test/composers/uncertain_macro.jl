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

    # alpha uncertain (a prior), theta fixed at 1.0.
    @test keys(u.specs) == (:alpha,)
    @test params(u.template)[2] == 1.0

    tree = compose((onset = u, admit = LogNormal(0.5, 0.4)))
    tbl = composed_to_table(tree)
    alpha_idx = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :alpha, eachindex(tbl.edge))
    theta_idx = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :theta, eachindex(tbl.edge))
    @test tbl.prior[alpha_idx] == Normal(0.7, 0.2)
    @test tbl.prior[theta_idx] === nothing

    # Bare uncertain(tree) keeps the attached spec and marks theta no_prior().
    promoted = uncertain(tree)
    ptbl = composed_to_table(promoted)
    @test ptbl.prior[alpha_idx] == Normal(0.7, 0.2)
    @test ptbl.prior[theta_idx] == no_prior()
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
    tbl = composed_to_table(tree)
    onset_alpha = findfirst(
        i -> tbl.edge[i] == :onset &&
             tbl.param[i] == :alpha, eachindex(tbl.edge))
    @test tbl.prior[onset_alpha] == LogNormal(log(2.0), 0.2)
    admit_rows = findall(i -> tbl.edge[i] == :admit, eachindex(tbl.edge))
    @test all(tbl.prior[i] === nothing for i in admit_rows)
end

@testitem "@uncertain: all-literal constructor is left unchanged" begin
    using Distributions

    @test (@uncertain LogNormal(0.5, 0.4)) == LogNormal(0.5, 0.4)
    @test (@uncertain Gamma(2.0, 1.0)) == Gamma(2.0, 1.0)
    @test !(@uncertain LogNormal(0.5, 0.4) isa Uncertain)
end
