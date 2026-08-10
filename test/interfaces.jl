# Interface-conformance suite: the contract a type must satisfy to take part in
# composition, checked uniformly over every built-in node shape and a
# user-defined node. Drives the reusable `ComposedDistributions.TestUtils`
# harness (shipped in `src`, ported from the CensoredDistributions interface
# suite, EpiAware/CensoredDistributions.jl#795) over the package's own fixtures,
# and adds the extra assertions the harness does not carry (the user-defined
# `Both` node, the underscored-alias identities, the introspection and
# leaf-wrapper contracts). Censoring-free: this package composes any
# `UnivariateDistribution`.

@testitem "node interface conformance over every composer shape" begin
    using ComposedDistributions, Distributions
    using ComposedDistributions.TestUtils: test_node_interface

    # The reusable node-extension checklist walks each node's flat event vector
    # the same way the composers do: `child_nleaves` is a positive `Int`,
    # `child_rand!` fills exactly the node's slot and leaves the padding either
    # side untouched, and `child_logpdf` is a finite scalar over that slot,
    # independent of the surrounding padding.
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
        test_node_interface(node; name = name)
    end
end

@testitem "public interface conformance over every composer shape" begin
    using ComposedDistributions
    using ComposedDistributions.TestUtils: test_interface, example_fixtures
    import ForwardDiff

    # The reusable public checklist over the package's own fixture registry: a
    # bare leaf, Sequential, Parallel, Resolve, Compete, choose, a nested mix and
    # the deep-nesting matrix (a Sequential of Parallel, a Choose of
    # Sequentials). ForwardDiff is injected so the AD-safety contract (a finite
    # logpdf gradient) runs on every fixture carrying an `ad` probe.
    for fix in example_fixtures()
        test_interface(fix; ad_gradient = ForwardDiff.gradient)
    end
end

@testitem "fixture registry covers every public composer type" begin
    using ComposedDistributions
    using ComposedDistributions.TestUtils: test_registry_coverage

    # A new public composer type added without a `test_interface` fixture fails
    # here (the registry-completeness meta-test).
    test_registry_coverage()
end

@testitem "composers reject invalid construction" begin
    using ComposedDistributions
    using ComposedDistributions.TestUtils: test_rejects_invalid

    test_rejects_invalid()
end

@testitem "composed-interface conformance and the keyword entry point" begin
    using ComposedDistributions, Distributions
    using ComposedDistributions.TestUtils: test_composed_interface, test_interface

    # `test_composed_interface` wraps the node-extension checklist and the public
    # checklist, and asserts the node subtypes `AbstractComposedDistribution`.
    node = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    test_composed_interface(node; draw = [1.5, 0.8], path = (:onset_admit,),
        overall = :vector, has_endpoint = false)
    # The keyword entry, as a downstream author would call it on a bare
    # Distributions.jl leaf (a valid univariate member).
    test_interface(Gamma(2.0, 1.0); draw = 3.0, univariate = true,
        has_endpoint = false)
end

@testitem "a user-defined composer node satisfies the node contract" begin
    using ComposedDistributions, Distributions, Random
    using ComposedDistributions.TestUtils: test_node_interface
    import ComposedDistributions: child_nleaves, child_logpdf, child_rand!,
                                  node_children, node_rebuild, component_names,
                                  AbstractComposedDistribution

    # A minimal user node combining two named branches side by side (the
    # worked example from the interface-contracts docs, #374). Subtyping
    # `AbstractComposedDistribution` and implementing `node_children` /
    # `node_rebuild` / `component_names` is the whole contract: `names` and
    # the children's types as the first two type parameters (matching
    # `Sequential`/`Parallel`/`Choose`/`Compete`'s own shape) is what lets the
    # flat-vector codec read the tree's layout with no method call of its own
    # (see `docs/src/developer/interface-contracts.md`), and everything else
    # -- `child_nleaves` / `child_logpdf` / `child_rand!`, `params`, the table
    # walk, `update` -- comes for free as a generic "concatenating node"
    # default. A node with different combination semantics overrides those
    # three `child_*` methods directly, as the old hand-rolled version of this
    # example used to.
    struct Both{names, C <: Tuple} <:
           AbstractComposedDistribution{Multivariate, Continuous}
        children::C
        function Both{names}(children::C) where {names, C <: Tuple}
            length(names) == length(children) ||
                throw(ArgumentError("names/children length mismatch"))
            new{names, C}(children)
        end
    end
    Both(children::C, names::NTuple{
        N, Symbol}) where {N, C <: Tuple} = Both{names}(children)

    node_children(d::Both) = d.children
    node_rebuild(d::Both, children::Tuple) = Both(children, component_names(d))
    component_names(::Both{names}) where {names} = names

    node = Both((Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (:first, :second))
    @test child_nleaves(node) == 2

    # The reusable harness accepts the user node directly, the same way it
    # checks the built-ins -- including the `flat_dimension` invariant
    # (#374), which a node whose codec layout disagreed with its table would
    # fail here.
    test_node_interface(node; name = "Both")

    # The node fills only its own slice and scores it position-independently,
    # through the generic `child_rand!`/`child_logpdf` defaults.
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

@testitem "a user-defined composer node composes, tables, flattens and \
fits" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: node_children, node_rebuild, component_names,
                                  AbstractComposedDistribution, flat_dimension,
                                  unflatten, flatten, reconstruct

    struct Both{names, C <: Tuple} <:
           AbstractComposedDistribution{Multivariate, Continuous}
        children::C
        function Both{names}(children::C) where {names, C <: Tuple}
            length(names) == length(children) ||
                throw(ArgumentError("names/children length mismatch"))
            new{names, C}(children)
        end
    end
    Both(children::C, names::NTuple{
        N, Symbol}) where {N, C <: Tuple} = Both{names}(children)
    node_children(d::Both) = d.children
    node_rebuild(d::Both, children::Tuple) = Both(children, component_names(d))
    component_names(::Both{names}) where {names} = names

    both = Both(
        (uncertain(Gamma(2.0, 1.0);
                shape = LogNormal(log(2.0), 0.2)), Gamma(1.5, 1.0)),
        (:leg, :tail))

    # Tables: composed_to_table walks the node generically (no method of its
    # own beyond the three above), reporting one estimated row for the
    # uncertain leaf.
    tbl = composed_to_table(both)
    @test tbl.node[1] == :Both
    @test count(!isnothing, tbl.prior) == 1

    # Flattens: the codec reads Both's (names, children-types) type
    # parameters at compile time and the estimated flat vector round-trips.
    @test flat_dimension(both) == 1
    nt = unflatten(both, [3.0])
    @test nt == (leg = (shape = 3.0, scale = 1.0), tail = (shape = 1.5,
        scale = 1.0))
    @test flatten(both, nt) == [3.0]

    # Fits: reconstruct/update collapse the tree at an estimated draw, the
    # primitive a sampler-driven fit routes through.
    fitted = reconstruct(both, [4.0])
    @test fitted isa Both
    @test params(node_children(fitted)[1]) == (4.0, 1.0)
    @test ComposedDistributions.update(both,
        (leg = (shape = 5.0, scale = 1.0), tail = (shape = 1.5, scale = 1.0))) isa
          Both

    # Composes: Both nests as a named child of a built-in composer (and a
    # built-in nests under Both, both directions of the same contract), and
    # the whole tree's own table/codec see straight through it.
    seq = Sequential((both, Gamma(1.0, 1.0)), (:branch, :tail2))
    @test composed_to_table(seq).node[1:2] == [:Sequential, :Both]
    @test flat_dimension(seq) == 1
    sel = choose(:x => both, :y => Gamma(2.0, 1.0))
    @test flat_dimension(sel) == 1
end

@testitem "an incomplete composer node fails loudly, not silently \
(#374)" begin
    using ComposedDistributions, Distributions, Test
    import ComposedDistributions: AbstractComposedDistribution, flat_dimension

    # A type that subtypes AbstractComposedDistribution but implements none
    # of the node contract. Before #374's fix this fell through the codec's
    # closed dispatch chain straight to the LEAF branch and silently reported
    # `flat_dimension == 0` for a tree that plainly has an uncertain leaf.
    # Every entry point must now fail loudly instead, naming the missing
    # method or the exact type-parameter shape the codec needs.
    struct NoMethods{F, S} <: AbstractComposedDistribution{F, S}
        a::Any
        b::Any
    end
    NoMethods(a, b) = NoMethods{Multivariate, Continuous}(a, b)

    incomplete = NoMethods(
        uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        Gamma(1.5, 1.0))

    @test_throws MethodError composed_to_table(incomplete)
    @test_throws MethodError ComposedDistributions.update(incomplete,
        (a = (shape = 3.0, scale = 1.0), b = (shape = 1.5, scale = 1.0)))
    # `flat_dimension` never even reaches `node_children`/`component_names`
    # for a type with no (names, children-types) type-parameter shape at
    # all -- the codec's own generation-time layout check catches it first,
    # with a message naming the fix.
    err = try
        flat_dimension(incomplete)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("type parameters", sprint(showerror, err))

    # `test_node_interface`/`test_estimation_dimension` are EXPECTED to fail
    # on `incomplete` -- that is the point of this fixture -- so each runs
    # under a throwaway parent testset (`Test.push_testset`/`pop_testset`)
    # rather than the real one this `@testitem` provides. Left under the real
    # parent, the inner error would be recorded into the SUITE's own pass/
    # fail tally (an "errored" count on the whole `Pkg.test` run) even though
    # every `@test` written here passes; swapping the ambient testset keeps
    # the expected-failure noise scoped to a throwaway sink this file reads
    # and discards, exactly as `@test_throws` isolates an expected exception.
    # A `Test.DefaultTestSet` pushes every non-passing outcome (a `Fail`, an
    # `Error`, or a failed nested testset) onto its own `results`, and
    # nothing onto it when everything passes, so `isempty(ts.results)` is a
    # reliable "did this fully pass" check.
    using ComposedDistributions.TestUtils: test_node_interface,
                                           test_estimation_dimension
    function _run_scoped(f)
        scratch = Test.DefaultTestSet("scratch")
        Test.push_testset(scratch)
        try
            return f()
        finally
            Test.pop_testset()
        end
    end

    node_ts = _run_scoped(() -> test_node_interface(incomplete))
    @test !isempty(node_ts.results)

    dim_ts = _run_scoped(() -> test_estimation_dimension(incomplete))
    @test !isempty(dim_ts.results)

    # A type that DOES implement node_children/node_rebuild/component_names
    # (so it composes and tables correctly) but not the type-parameter
    # layout the flat-vector codec needs: a clear, actionable error naming
    # the type and the fix, never a silent miscount.
    import ComposedDistributions: node_children, node_rebuild, component_names
    node_children(d::NoMethods) = (d.a, d.b)
    node_rebuild(d::NoMethods, c::Tuple) = NoMethods(c[1], c[2])
    component_names(::NoMethods) = (:a, :b)

    tbl = composed_to_table(incomplete)
    @test count(!isnothing, tbl.prior) == 1
    err2 = try
        flat_dimension(incomplete)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("type parameters", sprint(showerror, err2))
end

@testitem "abstract membership: composers sit under the right supertype" begin
    using ComposedDistributions, Distributions
    using ComposedDistributions.TestUtils: test_abstract_membership
    import ComposedDistributions: AbstractOneOf, AbstractComposedDistribution

    # The meta-test pins the whole hierarchy: every composer subtypes
    # `AbstractComposedDistribution`; `Sequential` / `Parallel` subtype
    # `AbstractMultiChild`; `Resolve` / `Compete` subtype `AbstractOneOf`;
    # `Choose` is a sibling, not a multi-child node.
    test_abstract_membership()

    # A plain leaf and a `Shared` tie are standalone univariate leaves, under no
    # composer supertype.
    @test Gamma(2.0, 1.0) isa UnivariateDistribution
    @test !(Gamma(2.0, 1.0) isa AbstractComposedDistribution)
    sh = shared(:inc, Gamma(2.0, 1.0))
    @test sh isa UnivariateDistribution
    @test !(sh isa AbstractComposedDistribution)
    @test !(sh isa AbstractOneOf)
end

@testitem "introspection contract: names, tree and composed_to_table agree" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: component_names

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # `component_names` returns a Tuple of the child names.
    @test component_names(tree) isa Tuple
    @test component_names(tree) == (:onset_admit, :admit_death)

    # `composed_to_table` is a Tables.jl column source, reachable by column.
    tbl = composed_to_table(tree)
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

    # `Distributions.censored(...)` (a `Censored` wrapper) behaves exactly like
    # `Truncated` above: fixed censoring bounds peel off, rewrap re-applies
    # them around the new inner delay.
    cs = censored(Gamma(2.0, 1.0); upper = 10.0)
    @test free_leaf(cs) == Gamma(2.0, 1.0)
    rwc = rewrap_leaf(cs, Gamma(3.0, 1.5))
    @test free_leaf(rwc) == Gamma(3.0, 1.5)
    @test logpdf(rwc, 2.0) ≈
          logpdf(censored(Gamma(3.0, 1.5); upper = 10.0), 2.0)
end

@testitem "composed_to_table reports a censored leaf's inner params, not its bounds" begin
    using ComposedDistributions: _param_rows
    using ComposedDistributions, Distributions

    # Before Censored had its own leaf-protocol methods, the parameter-only
    # walk fell back to the generic (unpeeled) walk and reported the censoring
    # bounds as if they were free parameters (a spurious `nothing` row for an
    # absent bound, and the fixed bound itself as a "value"), with the wrong
    # support. This is the regression guard for that gap.
    tree = compose((onset = censored(Gamma(2.0, 3.0); upper = 10.0),
        admit = LogNormal(0.5, 0.4)))
    tbl = _param_rows(tree)
    onset_rows = findall(==(:onset), tbl.edge)
    @test length(onset_rows) == 2
    @test tbl.param[onset_rows] == [:shape, :scale]
    @test tbl.value[onset_rows] == [2.0, 3.0]
    # The support is the untruncated/uncensored Gamma's own support, not the
    # censoring bounds.
    @test all(==((0.0, Inf)), tbl.support[onset_rows])
end

@testitem "truncated/censored on a composed tree throws an informative error" begin
    using ComposedDistributions, Distributions

    # `Sequential`/`Parallel` are multivariate; Distributions.jl's own
    # truncated/censored are univariate-only, so these already error via plain
    # dispatch. `Resolve`/`Compete` DO satisfy `UnivariateDistribution`, but
    # their outcome is a structured named event, not a plain scalar, so
    # truncating/censoring the whole node is not well-defined even though it
    # type-checks -- that case must not be allowed to silently construct and
    # fail later (at `rand`, with an unrelated internal error).
    seq = compose((onset = Gamma(2.0, 1.0), death = LogNormal(0.5, 0.4)))
    res = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))

    for node in (seq, res)
        @test_throws ArgumentError truncated(node; upper = 10.0)
        @test_throws ArgumentError censored(node; upper = 10.0)
    end

    # Regression guard: the message is built from the bare function name
    # ("truncated"/"censored"), so a naive "$(verb)ing" template renders the
    # nonsense "truncateding"/"censoreding" -- assert the real word appears
    # and the mangled one does not.
    for (f, verb) in ((truncated, "truncated"), (censored, "censored"))
        err = try
            f(res; upper = 10.0)
            nothing
        catch e
            e
        end
        msg = sprint(showerror, err)
        @test occursin("applying $verb to the whole node", msg)
        @test !occursin("$(verb)ing", msg)
    end
end
