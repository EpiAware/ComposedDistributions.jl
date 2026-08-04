# `composed_to_table` / `params_table` projection parity and structural
# recovery (#227 slice 1, commit A: walk + sinks + projection).

@testitem "composed_to_table / params_table: golden projection parity" begin
    using ComposedDistributions: update
    using Distributions, ConvolvedDistributions
    using ConvolvedDistributions: convolved

    # A locally defined toy leaf-wrapper (mirrors `ToyWrap` in
    # `codec_gen.jl`'s `register_leaf_wrapper!` docstring): extends
    # `free_leaf`/`rewrap_leaf` like a real leaf-wrapper package would, but
    # deliberately does NOT extend `leaf_layers`/`node_attributes`, so it
    # appears as one opaque node row rather than a peeled one (the "structure
    # is now recoverable" testitem below asserts this explicitly).
    struct ToyWrap{D <: UnivariateDistribution} <:
           ContinuousUnivariateDistribution
        dist::D
    end
    ComposedDistributions.free_leaf(d::ToyWrap) = ComposedDistributions.free_leaf(d.dist)
    function ComposedDistributions.rewrap_leaf(d::ToyWrap, inner)
        return ToyWrap(ComposedDistributions.rewrap_leaf(d.dist, inner))
    end

    fixed_resolve = resolve(:death => (Gamma(2.0, 3.5), 0.3),
        :discharge => (Gamma(1.0, 8.0), 0.7))
    uncertain_resolve = update(
        resolve(:death => (Gamma(2.0, 3.5), 0.3),
            :discharge => (Gamma(1.0, 8.0), 0.7)),
        (branch_probs = Dirichlet(ones(2)),))
    noevent_resolve = resolve(
        :event => (Gamma(1.5, 1.0), 0.4), :none => (NoEvent(), 0.6))
    inc = shared(:inc, Gamma(2.0, 1.0))
    shared_tree = choose(:index => inc,
        :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc)))
    noncentred_pool = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:region)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:region))))
    centred_pool = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:g, Beta(2.0, 3.0))),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:g, Beta(2.0, 3.0)))))
    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))

    fixtures = [
        "bare Sequential" => sequential(
            :a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        "Parallel" => parallel(
            :a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        "Resolve, fixed probs" => fixed_resolve,
        "Resolve, Dirichlet (stick rows)" => uncertain_resolve,
        "Resolve with a NoEvent branch" => noevent_resolve,
        "Compete" => compete(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        "Choose" => choose(
            :index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4)),
        "nested Sequential-of-Parallel" => sequential(
            :step1 => parallel(:x => Gamma(2.0, 1.0), :y => LogNormal(0.5, 0.4)),
            :step2 => Gamma(1.0, 1.0)),
        "shared(:g, ...) tied across two branches" => shared_tree,
        "non-centred pool, two members" => noncentred_pool,
        "centred pool, two members" => centred_pool,
        "uncertain leaf" => compose((onset = uncertain(
            Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),)),
        "truncated leaf" => compose((onset = truncated(
            Gamma(2.0, 1.0); upper = 10.0),)),
        "censored leaf" => compose((onset = censored(
            Gamma(2.0, 1.0); lower = 0.5),)),
        "varying leaf" => compose((onset = varying(
            t -> Gamma(2.0, 1.0 + 0.1 * t); covariate = :time),)),
        "Convolved composite leaf" => compose((total = conv,)),
        "opaque third-party leaf wrapper" => compose((a = ToyWrap(
            Gamma(2.0, 1.0)),))
    ]

    for (label, tree) in fixtures
        full = composed_to_table(tree)
        p = params_table(tree)
        idx = findall(==(:param), full.role)

        # Elementwise, in order — never sets, never sorted.
        @test p.edge == full.edge[idx]
        @test p.param == full.param[idx]
        @test p.value == full.value[idx]
        @test p.support == full.support[idx]
        @test all(
            i -> p.prior[i] === full.prior[idx][i] ||
                 isequal(p.prior[i], full.prior[idx][i]),
            eachindex(p.prior))

        # eltype parity: the projection is a straight column drop, not a
        # rebuild, so the column element types must match exactly.
        @test typeof(p.edge) == typeof(full.edge[idx])
        @test typeof(p.param) == typeof(full.param[idx])
        @test typeof(p.value) == typeof(full.value[idx])
        @test typeof(p.support) == typeof(full.support[idx])
        @test typeof(p.prior) == typeof(full.prior[idx])
    end
end

@testitem "composed_to_table: row invariants" begin
    using ComposedDistributions: update
    using Distributions, ConvolvedDistributions
    using ConvolvedDistributions: convolved

    inc = shared(:inc, Gamma(2.0, 1.0))
    conv = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.0))

    fixtures = [
        sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        parallel(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        resolve(:death => (Gamma(2.0, 3.5), 0.3),
            :discharge => (Gamma(1.0, 8.0), 0.7)),
        resolve(:event => (Gamma(1.5, 1.0), 0.4), :none => (NoEvent(), 0.6)),
        compete(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4)),
        choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4)),
        choose(:index => inc,
            :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc))),
        compose((onset = truncated(Gamma(2.0, 1.0); upper = 10.0),)),
        compose((onset = censored(Gamma(2.0, 1.0); lower = 0.5),)),
        compose((total = conv,))
    ]

    for tree in fixtures
        full = composed_to_table(tree)
        n = length(full.edge)
        # Every column is the same length.
        @test length(full.param) == n
        @test length(full.node) == n
        @test length(full.role) == n
        @test length(full.value) == n
        @test length(full.support) == n
        @test length(full.prior) == n

        @test issubset(Set(full.role), Set((:node, :attribute, :param)))
        @test eltype(full.node) == Symbol
        @test eltype(full.param) == Symbol

        for i in 1:n
            if full.role[i] == :node
                @test full.param[i] == Symbol("")
                @test full.value[i] === nothing
                @test full.support[i] === nothing
                @test full.prior[i] === nothing
            end
        end

        # The root row is the first row, at the empty edge.
        @test full.role[1] == :node
        @test full.edge[1] == Symbol("")
    end
end

@testitem "composed_to_table: structure is now recoverable" begin
    using Distributions

    # A Sequential and a Parallel with the same branch names and identical
    # children are byte-identical in `params_table` but differ in `node`.
    seq = sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    par = parallel(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    @test params_table(seq).edge == params_table(par).edge
    @test params_table(seq).value == params_table(par).value
    @test composed_to_table(seq).node[1] == :Sequential
    @test composed_to_table(par).node[1] == :Parallel
    @test composed_to_table(seq).node[1] != composed_to_table(par).node[1]

    # A Gamma vs a Weibull leaf differ in `node`.
    gtree = compose((a = Gamma(2.0, 1.0),))
    wtree = compose((a = Weibull(2.0, 1.0),))
    leaf_node(tree) = begin
        tbl = composed_to_table(tree)
        i = findfirst(k -> tbl.role[k] == :node && tbl.edge[k] == :a,
            eachindex(tbl.edge))
        tbl.node[i]
    end
    @test leaf_node(gtree) == :Gamma
    @test leaf_node(wtree) == :Weibull

    # A truncated leaf carries `(:Truncated, :lower)`/`(:Truncated, :upper)`
    # attribute rows that a bare leaf does not.
    ttree = compose((a = truncated(Gamma(2.0, 1.0); upper = 10.0),))
    tfull = composed_to_table(ttree)
    attrs = [(tfull.node[i], tfull.param[i])
             for i in eachindex(tfull.node)
             if tfull.role[i] == :attribute]
    @test (:Truncated, :lower) in attrs
    @test (:Truncated, :upper) in attrs
    plain = composed_to_table(gtree)
    @test !any(==(:attribute), plain.role)

    # A Choose carries a `(:Choose, :selector)` attribute row.
    ctree = choose(:index => Gamma(2.0, 1.0), :sourced => LogNormal(0.5, 0.4))
    cfull = composed_to_table(ctree)
    cattrs = [(cfull.node[i], cfull.param[i])
              for i in eachindex(cfull.node)
              if cfull.role[i] == :attribute]
    @test (:Choose, :selector) in cattrs

    # Both members of a `shared(:g, ...)` pair get node rows at their real
    # paths, while the param rows stay once under `:g`.
    inc = shared(:inc, Gamma(2.0, 1.0))
    stree = choose(:index => inc,
        :sourced => compose((src = LogNormal(0.5, 0.4), inc = inc)))
    sfull = composed_to_table(stree)
    node_edges = [sfull.edge[i]
                  for i in eachindex(sfull.edge)
                  if sfull.role[i] == :node && sfull.node[i] == :Shared]
    @test Set(node_edges) == Set([:index, Symbol("sourced.inc")])
    inc_param_edges = [sfull.edge[i]
                       for i in eachindex(sfull.edge)
                       if sfull.role[i] == :param && sfull.node[i] == :Gamma]
    @test inc_param_edges == [:inc, :inc]

    # A two-group non-centred pool gives each member's `z` row a
    # `(:Pool, pname)` attribute row naming its group.
    ptree = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:region_a)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:region_a)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:region_b)),
        west = uncertain(Gamma(2.0, 1.0); shape = pool(:region_b))))
    pfull = composed_to_table(ptree)
    pool_attrs = Dict{Symbol, Any}()
    for i in eachindex(pfull.edge)
        if pfull.role[i] == :attribute && pfull.node[i] == :Pool
            pool_attrs[pfull.edge[i]] = pfull.value[i]
        end
    end
    @test pool_attrs[:north].group == :region_a
    @test pool_attrs[:south].group == :region_a
    @test pool_attrs[:east].group == :region_b
    @test pool_attrs[:west].group == :region_b

    # A third-party leaf wrapper extending only `free_leaf`/`rewrap_leaf`
    # (not `leaf_layers`/`node_attributes`) appears as one opaque node row —
    # its inner delay's node/attribute rows are not separately visible, but
    # its parameters still are.
    struct OpaqueWrap{D <: UnivariateDistribution} <:
           ContinuousUnivariateDistribution
        dist::D
    end
    ComposedDistributions.free_leaf(d::OpaqueWrap) = ComposedDistributions.free_leaf(d.dist)
    otree = compose((a = OpaqueWrap(Gamma(2.0, 1.0)),))
    ofull = composed_to_table(otree)
    node_rows = [(ofull.edge[i], ofull.node[i])
                 for i in eachindex(ofull.edge)
                 if ofull.role[i] == :node]
    @test (:a, :OpaqueWrap) in node_rows
    @test !((:a, :Gamma) in node_rows)
    oparams = [ofull.param[i] for i in eachindex(ofull.edge)
               if ofull.role[i] == :param]
    @test Set(oparams) == Set((:shape, :scale))
end
