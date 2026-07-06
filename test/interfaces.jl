# Interface-conformance suite: the contract a type must satisfy to take part in
# composition, checked uniformly over every built-in node shape and a
# user-defined node. Ported from the CensoredDistributions interface suite
# (EpiAware/CensoredDistributions.jl#795), keeping the generic composer-node,
# one_of-outcome, leaf-wrapper and abstract-membership contracts, dropping the
# censoring families this package does not carry.

@testitem "node interface conformance over every composer shape" begin
    using ComposedDistributions, Distributions, Random
    import ComposedDistributions: child_nleaves, child_logpdf, child_rand!

    # Assert a node satisfies the public node-extension contract, the same way
    # the composers walk an event vector. `child_nleaves(node)` is a positive
    # `Int`; `child_rand!` fills exactly the node's `offset + 1 : offset + n`
    # slot and leaves the padding either side untouched; `child_logpdf` is a
    # finite scalar over that slot and does not depend on the surrounding
    # padding (scoring the same draw at offset 0 gives the same value).
    function check_node_interface(node; offset = 1, pad = 1, rng = Xoshiro(1))
        n = child_nleaves(node)
        @test n isa Int
        @test n >= 1

        len = offset + n + pad
        out = fill(NaN, len)
        ret = child_rand!(out, offset, rng, node)
        @test ret === nothing
        slot = (offset + 1):(offset + n)
        @test all(isfinite, @view out[slot])
        @test all(isnan, @view out[1:offset])
        @test all(isnan, @view out[(offset + n + 1):len])

        lp = child_logpdf(node, out, offset, n)
        @test lp isa Real
        @test isfinite(lp)
        tight = out[slot]
        @test child_logpdf(node, tight, 0, n) ≈ lp
    end

    leaf = Gamma(2.0, 1.0)
    seq = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    par = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    res = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    com = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    cho = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    nested = compose((path = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        other = Gamma(3.0, 1.0)))

    cases = (("leaf", leaf), ("Sequential", seq), ("Parallel", par),
        ("Resolve", res), ("Compete", com), ("Choose", cho),
        ("nested", nested))
    for (name, node) in cases
        @testset "$name" begin
            check_node_interface(node)
        end
    end
end

@testitem "a user-defined composer node satisfies the node contract" begin
    using ComposedDistributions, Distributions, Random
    import ComposedDistributions: child_nleaves, child_logpdf, child_rand!

    # A minimal user node combining two branches side by side (the worked
    # example from the interface-contracts docs). The public contract is reached
    # by the qualified name, the same way the leaf hooks are.
    struct Both{A, B}
        first::A
        second::B
    end
    child_nleaves(b::Both) = child_nleaves(b.first) + child_nleaves(b.second)
    function child_logpdf(b::Both, x, offset, ::Int)
        n1 = child_nleaves(b.first)
        n2 = child_nleaves(b.second)
        return child_logpdf(b.first, x, offset, n1) +
               child_logpdf(b.second, x, offset + n1, n2)
    end
    function child_rand!(out, offset, rng::AbstractRNG, b::Both)
        n1 = child_nleaves(b.first)
        child_rand!(out, offset, rng, b.first)
        child_rand!(out, offset + n1, rng, b.second)
        return nothing
    end

    node = Both(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
    @test child_nleaves(node) == 2

    # The node fills only its own slice and scores it position-independently.
    out = fill(NaN, 4)
    @test child_rand!(out, 1, Xoshiro(1), node) === nothing
    @test all(isfinite, @view out[2:3])
    @test isnan(out[1]) && isnan(out[4])
    lp = child_logpdf(node, out, 1, 2)
    @test isfinite(lp)
    @test child_logpdf(node, out[2:3], 0, 2) ≈ lp

    # The underscored aliases still resolve the same public methods, for callers
    # that reached the contract before it was made public.
    @test ComposedDistributions._child_nleaves === child_nleaves
    @test ComposedDistributions._child_logpdf === child_logpdf
    @test ComposedDistributions._child_rand! === child_rand!
end

@testitem "abstract membership: composers sit under the right supertype" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: AbstractOneOf

    # The one_of-outcome family shares `AbstractOneOf`.
    res = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    com = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    @test res isa AbstractOneOf
    @test com isa AbstractOneOf

    # The named-child composers are sibling multivariate distributions, not
    # members of the one_of family.
    seq = sequential(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    par = parallel(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    cho = choose(:a => Gamma(2.0, 1.0), :b => LogNormal(0.5, 0.4))
    for c in (seq, par, cho)
        @test c isa Distribution{Multivariate, Continuous}
        @test !(c isa AbstractOneOf)
    end

    # A plain leaf and a `Shared` tie are standalone univariate, under no
    # composer supertype.
    @test Gamma(2.0, 1.0) isa UnivariateDistribution
    @test !(Gamma(2.0, 1.0) isa AbstractOneOf)
    sh = shared(:inc, Gamma(2.0, 1.0))
    @test sh isa UnivariateDistribution
    @test !(sh isa AbstractOneOf)
end

@testitem "introspection contract: names, tree and params_table agree" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: component_names

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # `component_names` returns a Tuple of the child names.
    @test component_names(tree) isa Tuple
    @test component_names(tree) == (:onset_admit, :admit_death)

    # `params_table` is a Tables.jl column source: one row per free parameter,
    # reachable by column.
    tbl = params_table(tree)
    @test tbl.edge isa AbstractVector
    @test tbl.param isa AbstractVector
    @test length(tbl.edge) == length(tbl.param)

    # The flat name tuple, nested tree, and name-path lookup describe the same
    # structure.
    @test event_names(tree) isa Tuple
    @test event_tree(tree) isa NamedTuple
    @test event(tree, :onset_admit) == Gamma(2.0, 1.0)
end

@testitem "leaf-wrapper contract: free_leaf / rewrap_leaf round-trip" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: free_leaf, rewrap_leaf

    # A plain leaf is its own free leaf; rewrap replaces it.
    @test free_leaf(Gamma(2.0, 1.0)) == Gamma(2.0, 1.0)
    @test rewrap_leaf(Gamma(2.0, 1.0), Gamma(3.0, 1.5)) == Gamma(3.0, 1.5)

    # A `Truncated` peels to its untruncated base, rewrap rebuilds the bounds,
    # so the reconstructed node scores the new inner leaf under the same
    # truncation.
    tr = truncated(Gamma(2.0, 1.0); upper = 10.0)
    @test free_leaf(tr) == Gamma(2.0, 1.0)
    rw = rewrap_leaf(tr, Gamma(3.0, 1.5))
    @test free_leaf(rw) == Gamma(3.0, 1.5)
    @test logpdf(rw, 2.0) ≈
          logpdf(truncated(Gamma(3.0, 1.5); upper = 10.0), 2.0)

    # A `Shared` tie peels through to its inner leaf, rewrap rebuilds the tie.
    sh = shared(:inc, Gamma(2.0, 1.0))
    @test free_leaf(sh) == Gamma(2.0, 1.0)
    rw2 = rewrap_leaf(sh, Gamma(3.0, 1.5))
    @test free_leaf(rw2) == Gamma(3.0, 1.5)
end
