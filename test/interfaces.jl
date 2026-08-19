# Interface-conformance suite: the contract a type must satisfy to take part in
# composition, checked uniformly over every built-in node shape and a
# user-defined node. Drives the reusable `ComposedDistributions.TestUtils`
# harness (shipped in `src`, ported from the CensoredDistributions interface
# suite, EpiAware/CensoredDistributions.jl#795) over the package's own fixtures,
# and adds the extra assertions the harness does not carry (the user-defined
# `Both` node, the underscored-alias identities, the introspection and
# leaf-wrapper contracts). Censoring-free: this package composes any
# `UnivariateDistribution`.

# Scoping for a check that is MEANT to fail. Three items below run a harness
# function against a deliberately broken fixture, so the harness reports
# failures by design; those must be read, not folded into the suite's own
# tally.
#
# `Test.push_testset`/`pop_testset` (the obvious way to swap the ambient
# testset) were removed in Julia 1.13, so this goes through the documented
# `AbstractTestSet` extension API instead, which works on every supported
# version: a testset that absorbs results rather than throwing on them.
#
# `@testset` propagates a custom testset TYPE to nested testsets, so every
# testset inside a scoped block is an `ExpectedFailures` too. That is why the
# count below is an explicit recursion and not `Test.get_test_counts`, which
# reads `DefaultTestSet`'s own tallies and does not apply here.
@testsnippet ExpectedFailureSink begin
    using Test

    mutable struct ExpectedFailures <: Test.AbstractTestSet
        description::String
        results::Vector{Any}
    end
    function ExpectedFailures(desc::AbstractString; kwargs...)
        return ExpectedFailures(String(desc), Any[])
    end
    Test.record(ts::ExpectedFailures, res) = (push!(ts.results, res); res)
    # A nested testset records itself into its parent on finish, exactly as
    # `DefaultTestSet` does. Without this a failure nested two deep is
    # silently dropped and the assertions below pass for the wrong reason.
    function Test.finish(ts::ExpectedFailures)
        Test.get_testset_depth() > 0 && Test.record(Test.get_testset(), ts)
        return ts
    end

    # Run `f` under a throwaway parent, returning what `f` returned (its own
    # testset), so the caller reads that outcome instead of inheriting it.
    function scoped_failures(f)
        inner = Ref{Any}(nothing)
        @testset ExpectedFailures "expected failures" begin
            inner[] = f()
        end
        return inner[]
    end

    # How many checks failed or errored anywhere beneath `ts`.
    function failure_count(ts::ExpectedFailures)
        return sum(failure_count, ts.results; init = 0)
    end
    failure_count(r) = (r isa Test.Fail || r isa Test.Error) ? 1 : 0
end

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

@testitem "a leaf costs zero ComposedDistributions methods, given a \
Distributions.jl-conforming leaf" begin
    using ComposedDistributions, Distributions, Random
    using ComposedDistributions.TestUtils: test_node_interface

    # The worked example from the extending-guide docs' "Adding a leaf:
    # nothing" section (`docs/src/developer/extending.md`): `Zilch` is not a
    # `Distributions.jl` built-in, and implements only the methods
    # `Distributions.jl` itself requires of a univariate distribution --
    # `params`, `logpdf`, `rand`, and `minimum`/`maximum` -- no
    # `ComposedDistributions` method of its own. `minimum`/`maximum` are not
    # optional here: `composed_to_table`'s `support` column reads them
    # directly, and a leaf without them composes and scores but throws
    # (`MethodError: no method matching iterate(::Zilch)`, from
    # `Distributions.jl`'s own default `support` machinery) the first time it
    # is tabled.
    struct Zilch <: ContinuousUnivariateDistribution
        m::Float64
        s::Float64
    end
    Distributions.params(d::Zilch) = (d.m, d.s)
    Distributions.logpdf(d::Zilch, x::Real) = logpdf(Normal(d.m, d.s), x)
    Base.rand(rng::AbstractRNG, d::Zilch) = rand(rng, Normal(d.m, d.s))
    Base.minimum(d::Zilch) = -Inf
    Base.maximum(d::Zilch) = Inf

    # Names are derived from Zilch's own fields, with no `param_names` method.
    @test ComposedDistributions.param_names(Zilch(1.0, 2.0)) == (:m, :s)

    # It composes and tables: one `:node` row for the leaf plus one `:param`
    # row per field, at the right edge, value and (unbounded) support.
    tree = compose((onset = Zilch(1.0, 2.0), admit = LogNormal(0.5, 0.4)))
    tbl = composed_to_table(tree)
    onset_params = [i
                    for i in eachindex(tbl.edge)
                    if tbl.role[i] == :param && tbl.edge[i] == :onset]
    @test tbl.param[onset_params] == [:m, :s]
    @test tbl.value[onset_params] == [1.0, 2.0]
    @test all(==((-Inf, Inf)), tbl.support[onset_params])
    onset_node = findfirst(
        i -> tbl.role[i] == :node && tbl.edge[i] == :onset, eachindex(tbl.edge))
    @test tbl.node[onset_node] === Zilch

    # It scores and samples through the flat codec, unchanged.
    @test isfinite(logpdf(tree, [0.5, 1.0]))
    @test rand(Xoshiro(1), tree) isa NamedTuple

    # It takes an attached prior through `uncertain`, and the codec sees the
    # one estimable parameter with no further method.
    uncertain_tree = compose((onset = uncertain(
        Zilch(1.0, 2.0); m = Normal(0.0, 1.0)),))
    @test ComposedDistributions.flat_dimension(uncertain_tree) == 1
    nt = ComposedDistributions.unflatten(uncertain_tree, [3.0])
    @test nt == (onset = (m = 3.0, s = 2.0),)
    @test ComposedDistributions.flatten(uncertain_tree, nt) == [3.0]

    # The reusable node-contract harness accepts a bare `Zilch` leaf and a
    # tree containing one, the same way it checks a built-in leaf.
    test_node_interface(Zilch(1.0, 2.0); name = "Zilch")
    test_node_interface(tree; name = "tree-with-Zilch")
end

@testitem "a user-defined composer node satisfies the node contract" begin
    using ComposedDistributions, Distributions, Random
    using ComposedDistributions.TestUtils: test_node_interface
    import ComposedDistributions: child_nleaves, child_logpdf, child_rand!,
                                  node_children, node_rebuild, component_names,
                                  AbstractComposedDistribution

    # A minimal user node combining two named branches side by side (the
    # worked example from the extending-guide docs, #374). Subtyping
    # `AbstractComposedDistribution` and implementing `node_children` /
    # `node_rebuild` / `component_names` is the whole contract: `names` and
    # the children's types as the first two type parameters (matching
    # `Sequential`/`Parallel`/`Choose`/`Compete`'s own shape) is what lets the
    # flat-vector codec read the tree's layout with no method call of its own
    # (see `docs/src/developer/extending.md`), and everything else
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

    # `Both` nests as a named child of a built-in: construction accepts it
    # (the `_is_composable` structural check), the flat-vector `logpdf` scores
    # through it (positional, no naming needed), and `has_varying`/
    # `has_uncertain` walk through its `node_children`.
    tree = sequential(:pair => node, :tail => Gamma(1.0, 1.0))
    @test length(tree) == 3
    flat = [1.0, 2.0, 0.5]
    @test logpdf(tree, flat) ≈
          child_logpdf(node, flat, 0, 2) + logpdf(Gamma(1.0, 1.0), flat[3])
    @test has_varying(tree) == false
    @test has_uncertain(tree) == false
    @test parallel(:a => node, :b => Gamma(1.0, 1.0)) isa Parallel
    @test choose(:x => node, :y => Gamma(1.0, 1.0)) isa Choose
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
                alpha = LogNormal(log(2.0), 0.2)), Gamma(1.5, 1.0)),
        (:leg, :tail))

    # Tables: composed_to_table walks the node generically (no method of its
    # own beyond the three above), reporting one estimated row for the
    # uncertain leaf.
    tbl = composed_to_table(both)
    @test tbl.node[1] === Both
    @test count(!isnothing, tbl.prior) == 1

    # Flattens: the codec reads Both's (names, children-types) type
    # parameters at compile time and the estimated flat vector round-trips.
    @test flat_dimension(both) == 1
    nt = unflatten(both, [3.0])
    @test nt == (leg = (alpha = 3.0, theta = 1.0), tail = (alpha = 1.5,
        theta = 1.0))
    @test flatten(both, nt) == [3.0]

    # Fits: reconstruct/update collapse the tree at an estimated draw, the
    # primitive a sampler-driven fit routes through.
    fitted = reconstruct(both, [4.0])
    @test fitted isa Both
    @test params(node_children(fitted)[1]) == (4.0, 1.0)
    @test ComposedDistributions.update(both,
        (leg = (alpha = 5.0, theta = 1.0), tail = (alpha = 1.5, theta = 1.0))) isa
          Both

    # Composes: Both nests as a named child of a built-in composer (and a
    # built-in nests under Both, both directions of the same contract), and
    # the whole tree's own table/codec see straight through it.
    seq = Sequential((both, Gamma(1.0, 1.0)), (:branch, :tail2))
    @test composed_to_table(seq).node[1:2] == [Sequential, Both]
    @test flat_dimension(seq) == 1
    sel = choose(:x => both, :y => Gamma(2.0, 1.0))
    @test flat_dimension(sel) == 1
end

@testitem "an incomplete composer node fails loudly, not silently \
(#374)" setup=[ExpectedFailureSink] begin
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
        uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)),
        Gamma(1.5, 1.0))

    @test_throws MethodError composed_to_table(incomplete)
    @test_throws MethodError ComposedDistributions.update(incomplete,
        (a = (alpha = 3.0, theta = 1.0), b = (alpha = 1.5, theta = 1.0)))
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
    # under a throwaway parent testset rather than the real one this
    # `@testitem` provides. Left under the real parent, the inner error would
    # be recorded into the SUITE's own pass/fail tally (an "errored" count on
    # the whole `Pkg.test` run) even though every `@test` written here passes;
    # swapping the ambient testset keeps the expected-failure noise scoped to
    # a throwaway sink this file reads and discards, exactly as
    # `@test_throws` isolates an expected exception. See the
    # `ExpectedFailureSink` snippet for `scoped_failures`/`failure_count`.
    using ComposedDistributions.TestUtils: test_node_interface,
                                           test_estimation_dimension

    node_ts = scoped_failures(() -> test_node_interface(incomplete))
    @test failure_count(node_ts) > 0

    dim_ts = scoped_failures(() -> test_estimation_dimension(incomplete))
    @test failure_count(dim_ts) > 0

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

@testitem "leaf_mean/leaf_var report the wrapped moment, not the inner one" begin
    using ComposedDistributions, Distributions

    # Regression guard: `leaf_mean`/`leaf_var` used to peel all the way to the
    # inner free delay via `free_leaf`, silently reporting the untruncated /
    # uncensored moment for a `Truncated`/`Censored` leaf. `truncated(Normal;
    # lower)`/`censored(Normal; lower)` have a Distributions.jl closed-form
    # moment, so this is exact, not an approximation.
    tr = truncated(Normal(0.0, 1.0); lower = 0.0)
    @test ComposedDistributions.leaf_mean(tr) ≈ mean(tr)
    @test ComposedDistributions.leaf_mean(tr) != mean(Normal(0.0, 1.0))
    @test ComposedDistributions.leaf_var(tr) ≈ var(tr)

    cs = censored(Normal(0.0, 1.0); lower = 0.0)
    @test ComposedDistributions.leaf_mean(cs) ≈ mean(cs)
    @test ComposedDistributions.leaf_var(cs) ≈ var(cs)

    # It propagates: the overall mean of a Sequential containing a truncated
    # leaf is the sum of the true per-leaf moments.
    seq = ComposedDistributions.Sequential((tr, Gamma(1.0, 1.0)), (:a, :b))
    @test mean(seq) ≈ mean(tr) + mean(Gamma(1.0, 1.0))

    # A family Distributions.jl has no closed-form truncated/censored moment
    # for (e.g. `Truncated{Gamma}`) falls back to the old (inner-delay)
    # approximation rather than throwing.
    trg = truncated(Gamma(2.0, 1.0); upper = 10.0)
    @test ComposedDistributions.leaf_mean(trg) == mean(Gamma(2.0, 1.0))

    # A `Shared` tie is unaffected (it never had its own `mean`/`var`, and
    # still does not need one): the moment is read through to the inner leaf.
    sh = shared(:tag, Gamma(2.0, 1.0))
    @test ComposedDistributions.leaf_mean(sh) == mean(Gamma(2.0, 1.0))
end

@testitem "a wrapper defining only inner_dist is refused, not silently \
dropped" begin
    using ComposedDistributions, Distributions, Random

    # A wrapper carrying fixed structure (`scale`) that defines the peel hook
    # but not the rebuild. Before this was guarded, `rewrap_leaf` fell through
    # to the bare-leaf base case and returned the new inner delay alone, so the
    # `scale` was discarded with no error and every later `logpdf`/`rand` was
    # wrong by that factor -- on the fit path too, since `reconstruct` rebuilds
    # through `update`.
    struct HalfWrap{D} <: ContinuousUnivariateDistribution
        dist::D
        scale::Float64
    end
    ComposedDistributions.inner_dist(w::HalfWrap) = w.dist
    Distributions.params(w::HalfWrap) = params(w.dist)
    function Distributions.logpdf(w::HalfWrap, x::Real)
        return logpdf(w.dist, x / w.scale) - log(w.scale)
    end
    Base.rand(rng::AbstractRNG, w::HalfWrap) = w.scale * rand(rng, w.dist)
    Base.minimum(w::HalfWrap) = minimum(w.dist)
    Base.maximum(w::HalfWrap) = maximum(w.dist)

    w = HalfWrap(Gamma(2.0, 1.0), 3.0)
    err = try
        ComposedDistributions.rewrap_leaf(w, Gamma(5.0, 1.0))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("HalfWrap", msg)
    @test occursin("rewrap_leaf", msg)

    # The same refusal reaches the tree path rather than dropping the layer.
    tree = compose((a = w, b = LogNormal(0.5, 0.4)))
    @test_throws ArgumentError ComposedDistributions.update(
        tree, (a = (alpha = 5.0, theta = 1.0), b = (mu = 0.5, sigma = 0.4)))

    # A wrapper that completes the pair rebuilds and keeps its fixed field.
    ComposedDistributions.rewrap_leaf(d::HalfWrap, inner) = HalfWrap(inner, d.scale)
    rebuilt = ComposedDistributions.rewrap_leaf(w, Gamma(5.0, 1.0))
    @test rebuilt isa HalfWrap
    @test rebuilt.scale == 3.0
    @test params(rebuilt) == (5.0, 1.0)

    # A bare leaf is its own inner distribution, so the base case is unchanged.
    @test ComposedDistributions.rewrap_leaf(Gamma(2.0, 1.0), Gamma(5.0, 1.0)) ==
          Gamma(5.0, 1.0)
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
    @test tbl.param[onset_rows] == [:alpha, :theta]
    @test tbl.value[onset_rows] == [2.0, 3.0]
    # The support is the untruncated/uncensored Gamma's own support, not the
    # censoring bounds.
    @test all(==((0.0, Inf)), tbl.support[onset_rows])
end

@testitem "compose()'s NamedTuple/pairs front ends accept a downstream node" begin
    using ComposedDistributions, Distributions
    import ComposedDistributions: child_nleaves, child_logpdf, child_rand!,
                                  node_children, AbstractComposedDistribution

    # `compose`'s own `_compose_child` lowering has to widen alongside
    # `_is_composable` (`Sequential`/`Parallel`/`Choose`'s inner-constructor
    # guard), since it is a second, separate closed-type check on the same
    # front door.
    struct Pair2{A, B} <: AbstractComposedDistribution{Multivariate, Continuous}
        first::A
        second::B
    end
    child_nleaves(b::Pair2) = child_nleaves(b.first) + child_nleaves(b.second)
    function child_logpdf(b::Pair2, x, offset, ::Int)
        n1 = child_nleaves(b.first)
        return child_logpdf(b.first, x, offset, n1) +
               child_logpdf(b.second, x, offset + n1, child_nleaves(b.second))
    end
    function child_rand!(out, offset, rng, b::Pair2)
        n1 = child_nleaves(b.first)
        child_rand!(out, offset, rng, b.first)
        child_rand!(out, offset + n1, rng, b.second)
        return nothing
    end
    node_children(b::Pair2) = (b.first, b.second)

    node = Pair2(Gamma(2.0, 1.0), Gamma(1.0, 1.0))

    # The NamedTuple front end (`compose((a = node, ...))`) and the pairs
    # spelling both accept it as a child, and a `compose(origin; branches...)`
    # shared-origin call accepts it as the origin.
    t1 = compose((first = node, second = LogNormal(0.5, 0.4)))
    @test t1 isa Parallel
    t2 = compose(:first => node, :second => LogNormal(0.5, 0.4))
    @test t2 == t1
    t3 = compose(node; a = Gamma(1.0, 1.0), b = Gamma(2.0, 1.0))
    @test t3 isa Sequential
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

@testitem "leaf-protocol completeness (#277): built-in wrappers pass" begin
    using ComposedDistributions, Distributions
    using ComposedDistributions.TestUtils: test_leaf_protocol_completeness

    # `truncated`/`censored` and `shared` each implement the full optional
    # leaf-protocol surface for a plain leaf, so the completeness check passes
    # cleanly for each: an attached prior, a shared tie, and the moments all
    # survive the wrapper.
    test_leaf_protocol_completeness(d -> truncated(d; upper = 10.0);
        name = "truncated")
    test_leaf_protocol_completeness(d -> censored(d; upper = 10.0);
        name = "censored")
end

@testitem "leaf-protocol completeness (#277): an incomplete wrapper fails, \
naming the hook" setup=[ExpectedFailureSink] begin
    using ComposedDistributions, Distributions, Random, Test
    using ComposedDistributions.TestUtils: test_leaf_protocol_completeness
    import ComposedDistributions: free_leaf, rewrap_leaf

    # A wrapper implementing only the two mandatory hooks (the #277
    # reproduction from the design review): it peels/rebuilds correctly but
    # silently drops an attached prior and a shared tie, because the optional
    # hooks (`uncertain_specs`, `shared_tag`) fall back to their fixed-leaf
    # defaults instead of forwarding to `.inner`.
    struct IncompleteWrap{D} <: Distributions.ContinuousUnivariateDistribution
        inner::D
    end
    free_leaf(d::IncompleteWrap) = free_leaf(d.inner)
    rewrap_leaf(d::IncompleteWrap, inner) = IncompleteWrap(rewrap_leaf(d.inner, inner))
    Distributions.params(d::IncompleteWrap) = Distributions.params(d.inner)
    Distributions.logpdf(d::IncompleteWrap, x::Real) = logpdf(d.inner, x)
    Base.rand(rng::AbstractRNG, d::IncompleteWrap) = rand(rng, d.inner)
    Distributions.minimum(d::IncompleteWrap) = minimum(d.inner)
    Distributions.maximum(d::IncompleteWrap) = maximum(d.inner)

    # `test_leaf_protocol_completeness` returns its `@testset`, which records
    # its failures rather than throwing (so a caller sees every dropped hook
    # at once, not just the first). Run it under a throwaway scratch parent
    # testset so its two deliberate failures land there and are discarded,
    # not folded into this @testitem's own count; the check reads the
    # returned testset's counts instead. The failure summary still prints
    # below (expected -- that IS the demonstration).
    result = scoped_failures(
        () -> test_leaf_protocol_completeness(
        d -> IncompleteWrap(d); name = "IncompleteWrap"))
    # Both dropped hooks are caught (uncertain_specs and shared_tag).
    @test failure_count(result) >= 2
end

@testitem "sampling-vs-density consistency (#278): built-ins pass, a \
corrupted rand fails" setup=[ExpectedFailureSink] begin
    using ComposedDistributions, Distributions, Random, Test
    using ComposedDistributions.TestUtils: test_sampling_consistency

    # A bare leaf checks clean directly; a `Sequential` (a genuine chain, not
    # `compose`'s NamedTuple-is-Parallel front end) has no scalar `rand` (it
    # is multivariate, a labelled NamedTuple), so it goes through
    # `observed_distribution` first to reach the collapsed univariate view the
    # check needs (documented on `test_sampling_consistency`).
    test_sampling_consistency(Gamma(2.0, 1.0); nsamples = 20_000, name = "leaf")
    seq = sequential(:onset => Gamma(2.0, 1.0), :admit => LogNormal(0.5, 0.4))
    test_sampling_consistency(
        observed_distribution(seq); nsamples = 20_000, name = "Sequential")

    # A deliberately corrupted `rand` (drawn from a scaled distribution while
    # `logpdf`/`cdf`/`mean`/`var` still describe the unscaled one) must fail:
    # the check earning its keep, not merely passing on everything. Run under
    # a throwaway scratch parent testset (see the matching note on the #277
    # test above) so the expected failure summary does not count against this
    # file's pass total.
    struct BrokenGamma <: Distributions.ContinuousUnivariateDistribution
        inner::Gamma{Float64}
    end
    Distributions.logpdf(d::BrokenGamma, x::Real) = logpdf(d.inner, x)
    Distributions.cdf(d::BrokenGamma, x::Real) = cdf(d.inner, x)
    Distributions.mean(d::BrokenGamma) = mean(d.inner)
    Distributions.var(d::BrokenGamma) = var(d.inner)
    Base.rand(rng::AbstractRNG, d::BrokenGamma) = 1.5 * rand(rng, d.inner)

    result = scoped_failures(
        () -> test_sampling_consistency(
        BrokenGamma(Gamma(2.0, 1.0)); nsamples = 20_000, name = "broken"))
    @test failure_count(result) >= 1
end
