# ============================================================================
# TestUtils: a public interface-conformance harness for the composers
# ============================================================================
#
# `ComposedDistributions.TestUtils.test_interface(d)` runs one interface
# checklist over a composed distribution (or a bare leaf), so a downstream author
# writing a new leaf / composer can drop it into their own `@testset` to verify
# conformance. The package itself runs it over a fixture set in
# `test/interfaces.jl` (see `example_fixtures`).
#
# The harness is deliberately dependency-light: it uses `Test` (a stdlib), the
# package's own public surface, and `Tables`. It returns the `@testset` result so
# a caller can assert on it.

"""
    ComposedDistributions.TestUtils

Public interface-conformance harness for the composers.

`TestUtils.test_interface(d)` runs one interface checklist over a composed
distribution (or a bare leaf), so a downstream author writing a new leaf or
composer can drop it into their own `@testset` to verify conformance against the
package's public interface. `test_node_interface(node)` is the companion check
for a new composer node, asserting its `child_nleaves` / `child_logpdf` /
`child_rand!` methods round-trip on a flat event vector. [`test_interface`](@ref),
[`example_fixtures`](@ref), [`test_rejects_invalid`](@ref),
[`test_node_interface`](@ref), [`test_composed_interface`](@ref) and
[`test_abstract_membership`](@ref) are exported from this submodule.
"""
module TestUtils

using Random: Random, AbstractRNG, Xoshiro
using Test: Test, @testset, @test, @test_nowarn, @test_throws
using Distributions: Distributions, mean, var, std, logpdf, cdf, params,
                     UnivariateDistribution, Distribution, Multivariate
import Tables

using ..ComposedDistributions: ComposedDistributions, Sequential, Parallel,
                               Resolve, Compete, AbstractOneOf, Choose,
                               compose, resolve, compete, choose,
                               event, event_names, event_tree, params_table,
                               observed_distribution, component_names,
                               AbstractComposedDistribution, AbstractMultiChild,
                               child_nleaves, child_logpdf, child_rand!

export test_interface, example_fixtures, test_rejects_invalid,
       test_node_interface, test_ad_safety, registry_types,
       test_registry_coverage, test_composed_interface, test_abstract_membership

# --- per-fixture descriptor -------------------------------------------------
#
# A fixture is the distribution plus the metadata the checklist needs that is not
# recoverable from the object alone: a known event-name `path` to round-trip
# through `event`, an in-support `draw` to score, a `kind` selector for a
# `Choose`, and the shape of the overall `mean(d)` moment.

Base.@kwdef struct InterfaceFixture{D}
    name::String
    dist::D
    "An in-support realisation to score (a scalar for a univariate node, a flat
    `Vector` or a labelled `NamedTuple` for a multivariate composer)."
    draw::Any = nothing
    "A known `event` path (tuple of Symbols) that must round-trip, or `nothing`."
    path::Union{Nothing, Tuple} = nothing
    "The `kind` keyword for a `Choose` fixture, or `nothing`."
    kind::Union{Nothing, Symbol} = nothing
    "Whether the node's `rand` is a univariate scalar (a leaf / `Resolve` /
    `Compete`)."
    univariate::Bool = false
    "The shape of the overall `mean(d)`/`var(d)`/`std(d)` moment: `:scalar` for a
    univariate-collapsible node (a leaf, `Sequential`, `Resolve`, `Compete`),
    `:vector` for a genuinely multivariate `Parallel` (a per-endpoint
    NamedTuple), or `:none` to skip the overall-moment check (a `Choose`, or a
    node with no closed-form moment)."
    overall::Symbol = :scalar
    "Whether the node collapses to a univariate endpoint via
    `observed_distribution` (a chain / univariate)."
    has_endpoint::Bool = true
    "An AD-safety probe `(f, θ)`: a closure `f(θ::Vector) -> Real` reconstructing
    the node from a parameter vector and returning a scalar log density, plus an
    in-support point `θ`. When supplied (and an `ad_gradient` backend is passed to
    `test_interface`), the harness asserts the gradient is finite. `nothing`
    skips it."
    ad::Union{Nothing, Tuple{Function, Vector{Float64}}} = nothing
end

# --- the checklist ----------------------------------------------------------

@doc """

Run the public interface-conformance checklist over a composed distribution.

`test_interface(d; name)` runs one `@testset` of interface assertions against `d`
(a composed distribution or a bare leaf), so a downstream author writing a new
leaf / composer can verify conformance by dropping it into their own tests.

The checklist asserts, where applicable to the node's shape:

- a `rand(d)` realisation is a scalar (univariate) or a labelled NamedTuple
  (multivariate composer);
- the overall `mean` / `var` / `std` are shaped as the fixture's `overall`
  declares (a scalar for a univariate-collapsible node, a per-endpoint NamedTuple
  for a `Parallel`);
- `logpdf` is finite on the supplied in-support `draw`;
- a univariate `cdf` is monotone and in `[0, 1]`;
- `params` works and `params_table` is a Tables.jl table
  (`Tables.istable(params_table(d))`);
- `event_names` (flat) and `event_tree` agree in leaf count;
- `event(d, path...)` round-trips the supplied known path;
- `observed_distribution` collapses a chain to a univariate scalar.

Pass the fixture metadata (an [`example_fixtures`](@ref) entry, or the keyword
arguments directly) so the harness knows the in-support `draw`, a known `event`
`path`, a `Choose` `kind`, and the `overall` moment shape. Returns the `@testset`
object.

# Examples
```julia
using ComposedDistributions, Distributions
using ComposedDistributions.TestUtils: test_interface

d = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
test_interface(d; draw = [1.5, 0.8], path = (:onset_admit,),
    overall = :vector, has_endpoint = false)
```
""" function test_interface end

function test_interface(d; name::AbstractString = string(nameof(typeof(d))),
        draw = nothing, path::Union{Nothing, Tuple} = nothing,
        kind::Union{Nothing, Symbol} = nothing, univariate::Bool = false,
        overall::Symbol = :scalar, has_endpoint::Bool = true,
        ad::Union{Nothing, Tuple{Function, Vector{Float64}}} = nothing,
        ad_gradient = nothing)
    fix = InterfaceFixture(; name = name, dist = d, draw = draw, path = path,
        kind = kind, univariate = univariate, overall = overall,
        has_endpoint = has_endpoint, ad = ad)
    return test_interface(fix; ad_gradient = ad_gradient)
end

# `ad_gradient` is an injected gradient backend (e.g. `ForwardDiff.gradient`):
# the harness lives in `src` and is dependency-light (no AD dep of its own), so a
# caller in the test env passes the backend it has loaded. `nothing` skips the
# AD-safety contract check.
function test_interface(fix::InterfaceFixture; ad_gradient = nothing)
    d = fix.dist
    return @testset "interface: $(fix.name)" begin
        _check_choose(d, fix)
        _check_moments_and_rand(d, fix)
        _check_logpdf(d, fix)
        _check_cdf(d, fix)
        _check_params(d)
        _check_event_names(d, fix)
        _check_event_path(d, fix)
        _check_endpoint(d, fix)
        _check_ad(d, fix, ad_gradient)
    end
end

# A Choose needs a selection for length/rand/logpdf, so it is checked on its
# own track (the generic moment / logpdf checks are skipped for it).
_is_choose(::Choose) = true
_is_choose(::Any) = false

function _check_choose(d::Choose, fix)
    @testset "choose" begin
        fix.kind === nothing && return
        @test_nowarn event(d, fix.kind)
        # The selected alternative round-trips through `event` and scores.
        @test fix.draw === nothing ||
              isfinite(logpdf(d, fix.draw; kind = fix.kind))
    end
    return nothing
end
_check_choose(::Any, fix) = nothing

# The overall `mean(d)` is a scalar for a univariate-collapsible node and a
# per-endpoint NamedTuple for a `Parallel`. Any multivariate composed output is a
# labelled `NamedTuple`; a univariate (collapsible) output stays a scalar.
function _check_moments_and_rand(d, fix)
    _is_choose(d) && return nothing
    @testset "moments and rand" begin
        r = rand(d)
        # A multivariate composer realisation is a labelled NamedTuple; a
        # univariate node is a bare scalar.
        if fix.univariate
            @test r isa Real
        else
            @test r isa NamedTuple
        end
        # Overall moment shape.
        if fix.overall === :scalar
            @test mean(d) isa Real
            @test var(d) isa Real
            @test std(d) isa Real
        elseif fix.overall === :vector
            # A genuinely multivariate `Parallel`: the per-endpoint moment is a
            # labelled NamedTuple keyed by the endpoint names.
            m = mean(d)
            v = var(d)
            s = std(d)
            @test m isa NamedTuple
            @test v isa NamedTuple
            @test s isa NamedTuple
            @test keys(m) == keys(v) == keys(s)
        end
    end
    return nothing
end

function _check_logpdf(d, fix)
    _is_choose(d) && return nothing
    fix.draw === nothing && return nothing
    @testset "logpdf finite on an in-support draw" begin
        @test isfinite(_score(d, fix.draw))
    end
    return nothing
end

# Score `draw` under `d`: a scalar for a univariate node, a flat per-value vector
# or a labelled NamedTuple for a composer.
_score(d, draw::Real) = logpdf(d, draw)
_score(d, draw::NamedTuple) = logpdf(d, draw)
_score(d, draw::AbstractVector) = logpdf(d, draw)

# A deterministic in-support flat event vector for a composer, drawn straight
# through the node's own `child_rand!`, so `logpdf(d, draw)` scores the same
# layout the node samples. Used to build the multivariate fixtures' draws.
function flat_draw(d, rng::AbstractRNG = Xoshiro(1))
    out = zeros(child_nleaves(d))
    child_rand!(out, 0, rng, d)
    return out
end

# A univariate node's cdf is monotone and lives in [0, 1].
function _check_cdf(d, fix)
    fix.univariate || return nothing
    @testset "univariate cdf monotone in [0, 1]" begin
        xs = range(0.0, 30.0; length = 12)
        cs = [cdf(d, x) for x in xs]
        @test all(c -> 0.0 - 1e-8 <= c <= 1.0 + 1e-8, cs)
        @test issorted(cs)
    end
    return nothing
end

function _check_params(d)
    @testset "params / params_table" begin
        @test_nowarn params(d)
        if d isa Union{Sequential, Parallel, Resolve, Choose}
            tbl = params_table(d)
            @test Tables.istable(tbl)
        end
    end
    return nothing
end

# The flat event path and the nested `event_tree` must agree in leaf count:
# every `event_tree` leaf (a Resolve outcome / a leaf delay) has its own flat
# slot, plus the flat origin event, so `length(flat) == leaves + 1`. A `Choose`
# (standalone or nested as a composer child) shares one flat slot across its
# alternatives, while its `event_tree` carries every alternative name, so the
# leaf-count equality does not hold; for a Choose-containing node the check is
# that the flat count matches the actual flat event layout and both are
# non-empty.
function _check_event_names(d, fix)
    d isa Union{Sequential, Parallel, Resolve, Choose} || return nothing
    @testset "event_names / event_tree leaf count" begin
        tree = event_tree(d)
        if d isa Choose
            # A Choose has no shared origin / flat path; its record keys are the
            # alternative names, which must be non-empty.
            @test !isempty(event_names(d))
            @test !isempty(keys(tree))
        elseif _contains_choose(d)
            # A nested Choose collapses its alternatives to one shared flat slot,
            # so the flat count tracks the event layout, not the tree leaf count.
            @test length(ComposedDistributions._flat_event_names(d)) ==
                  ComposedDistributions._event_nleaves(d.components) + 1
            @test !isempty(keys(tree))
        else
            @test length(ComposedDistributions._flat_event_names(d)) ==
                  _tree_leaf_count(tree) + 1
        end
    end
    return nothing
end

# Whether a composer tree contains a nested `Choose` anywhere (its alternatives
# share one flat event slot, so the tree-vs-flat leaf-count equality is relaxed).
_contains_choose(::Choose) = true
_contains_choose(c::AbstractMultiChild) = any(_contains_choose, c.components)
_contains_choose(c::AbstractOneOf) = any(_contains_choose, c.delays)
_contains_choose(::Any) = false

# Count the leaves of an `event_tree` (a nested NamedTuple keyed to leaf names).
_tree_leaf_count(x::Symbol) = 1
_tree_leaf_count(nt::NamedTuple) = sum(_tree_leaf_count, values(nt))

function _check_event_path(d, fix)
    fix.path === nothing && return nothing
    @testset "event round-trips a known path" begin
        @test_nowarn event(d, fix.path...)
    end
    return nothing
end

# `observed_distribution` collapses a chain to a univariate scalar. A node with
# several independent endpoints (a `Parallel`, a nested tree rooted in one) has
# no single observed scalar, so the check is skipped for it (both via the
# `has_endpoint` fixture flag and a `hasmethod` guard).
function _check_endpoint(d, fix)
    fix.has_endpoint || return nothing
    hasmethod(observed_distribution, Tuple{typeof(d)}) || return nothing
    @testset "observed_distribution collapses a chain" begin
        obs = observed_distribution(d)
        @test obs isa UnivariateDistribution
    end
    return nothing
end

# --- AD-safety contract -----------------------------------------------------

# `logpdf` must differentiate: the fixture's `ad = (f, θ)` reconstructs the node
# from a parameter vector and returns a scalar log density, and the injected
# `ad_gradient` backend (e.g. `ForwardDiff.gradient`, passed from the test env)
# evaluates `∇f(θ)`, which must be finite. With no backend injected the check is
# skipped, so the harness keeps its `src` AD-dep free; the package's own suite
# injects ForwardDiff.
function _check_ad(d, fix, ad_gradient)
    fix.ad === nothing && return nothing
    ad_gradient === nothing && return nothing
    f, θ = fix.ad
    @testset "logpdf is AD-differentiable (finite gradient)" begin
        g = ad_gradient(f, θ)
        @test g isa AbstractVector
        @test all(isfinite, g)
    end
    return nothing
end

@doc """

Assert a parameterised log density differentiates under an injected AD backend.

`test_ad_safety(f, θ; ad_gradient, name)` evaluates `ad_gradient(f, θ)` (e.g.
`ForwardDiff.gradient`) on a closure `f(θ::Vector) -> Real` reconstructing a
distribution from its parameter vector and returning a scalar log density, and
asserts the gradient is finite. Returns the `@testset` object.
""" function test_ad_safety(f::Function, θ::Vector{Float64}; ad_gradient,
        name::AbstractString = "ad")
    return @testset "AD-safety: $name" begin
        g = ad_gradient(f, θ)
        @test g isa AbstractVector
        @test all(isfinite, g)
    end
end

# --- the package's own fixture set ------------------------------------------

@doc """

The example fixture set over every composer shape, for [`test_interface`](@ref).

Returns a `Vector` of [`test_interface`](@ref)-ready fixtures covering the
composer shapes (`Sequential`, `Parallel`, `Resolve`, `Compete`, `choose`), a
bare leaf, a nested mix, and a deep-nesting matrix (a `Sequential` of `Parallel`,
a `Choose` of `Sequential`s). [`test_registry_coverage`](@ref) asserts these
cover every public composer type. The package runs the conformance checklist over
these in `test/interfaces.jl`; a downstream author can read them as
worked examples of the metadata `test_interface` expects (a `draw`, an `event`
`path`, the `overall` moment shape, an `ad` probe).
""" function example_fixtures end

function example_fixtures()
    G = Distributions.Gamma
    LN = Distributions.LogNormal

    seq = Sequential((G(2.0, 1.0), LN(0.5, 0.4)), (:onset_admit, :admit_death))
    par = Parallel((G(2.0, 1.0), LN(1.0, 0.5)), (:admit, :notif))
    comp = resolve(:death => (G(2.0, 3.5), 0.4), :discharge => (G(1.0, 8.0), 0.6))
    cmp = compete(:death => G(2.0, 1.0), :recover => G(1.5, 2.0))
    sel = choose(:short => G(2.0, 1.0), :long => G(5.0, 1.0))
    nested = compose((
        admit_path = compose((onset_admit = G(1.2, 3.0),
            admit_death = LN(0.5, 0.4))),
        onset_recover = G(0.7, 20.0)))
    # A Sequential whose first step is a Parallel: a chain that fans out.
    seq_of_par = Sequential(
        (Parallel((G(2.0, 1.0), LN(1.5, 2.0)),
                (:a, :b)), G(1.0, 3.0)), (:fanout, :tail))
    # A Choose of Sequentials: each alternative is a two-step chain.
    choose_of_seq = choose(
        :fast => Sequential((G(1.0, 1.0), G(1.0, 1.0)), (:a, :b)),
        :slow => Sequential((G(2.0, 2.0), G(2.0, 2.0)), (:c, :d)))

    # AD probes: reconstruct the node from a parameter vector, score a scalar
    # logpdf, and assert differentiability under an injected backend.
    leaf_ad = (θ -> logpdf(G(θ[1], θ[2]), 3.0), [2.0, 1.0])
    comp_ad = (
        θ -> logpdf(
            resolve(:death => (G(θ[1], θ[2]), 0.4),
                :discharge => (G(1.0, 8.0), 0.6)), 4.0),
        [2.0, 3.5])

    return InterfaceFixture[
        # A plain leaf has the full univariate interface (scalar moment + cdf).
        InterfaceFixture(; name = "bare plain leaf", dist = G(2.0, 1.0),
            draw = 3.0, univariate = true, overall = :scalar,
            has_endpoint = false, ad = leaf_ad),
        # A Sequential collapses to its overall scalar moment (the convolved
        # total) and has a single observed endpoint.
        InterfaceFixture(; name = "Sequential", dist = seq,
            draw = [1.5, 0.8], path = (:onset_admit,), overall = :scalar),
        # A Parallel is genuinely multivariate: the overall moment is a
        # per-endpoint NamedTuple. It has several independent endpoints and so no
        # single observed scalar.
        InterfaceFixture(; name = "Parallel", dist = par,
            draw = [1.2, 2.3], path = (:admit,), overall = :vector,
            has_endpoint = false),
        # A terminal `Resolve`: scalar marginal moments / cdf (via `as_mixture`)
        # and a scalar `rand`.
        InterfaceFixture(; name = "Resolve", dist = comp, draw = 4.0,
            path = (:death,), univariate = true, overall = :scalar,
            has_endpoint = false, ad = comp_ad),
        # Compete (racing hazards): a univariate time-to-first-event marginal.
        InterfaceFixture(; name = "Compete", dist = cmp, draw = 2.0,
            path = (:death,), univariate = true, overall = :scalar,
            has_endpoint = false),
        # A data-selected disjunction: scored against a `kind` selection.
        InterfaceFixture(; name = "choose", dist = sel, draw = 3.0,
            kind = :short, path = (:short,), overall = :none,
            has_endpoint = false),
        # A nested tree branches off a shared origin (a Parallel at its root), so
        # its overall moment is a per-endpoint NamedTuple and it has no single
        # collapsed endpoint.
        InterfaceFixture(; name = "nested mix", dist = nested,
            draw = flat_draw(nested), path = (:admit_path, :admit_death),
            overall = :vector, has_endpoint = false),
        # A Sequential whose step is a Parallel (a chain that fans out): no single
        # observed endpoint and no closed-form overall moment.
        InterfaceFixture(; name = "deep: Sequential of Parallel",
            dist = seq_of_par, draw = flat_draw(seq_of_par),
            path = (:fanout,), overall = :none, has_endpoint = false),
        # A Choose of Sequentials (the selected alternative is a chain).
        InterfaceFixture(; name = "deep: Choose of Sequentials",
            dist = choose_of_seq, draw = [1.0, 2.0], kind = :fast,
            path = (:fast,), overall = :none, has_endpoint = false)
    ]
end

# --- registry completeness --------------------------------------------------

@doc """

The public composer types the fixture registry must cover.

`registry_types()` returns the `Vector` of the package's own public composer
types that [`test_interface`](@ref) is expected to exercise.
[`test_registry_coverage`](@ref) asserts every entry appears in at least one
[`example_fixtures`](@ref) fixture, so a new public composer type added without a
fixture fails.
""" function registry_types()
    return Type[Sequential, Parallel, Resolve, Compete, Choose]
end

# Every concrete type appearing in a fixture's distribution, walked recursively
# through the composer children, so a type nested deep in a tree still counts as
# covered.
function _covered_types(fixtures)
    seen = Set{Type}()
    for fix in fixtures
        _collect_types!(seen, fix.dist)
    end
    return seen
end

function _collect_types!(seen, d)
    push!(seen, typeof(d))
    if d isa AbstractMultiChild
        for c in d.components
            _collect_types!(seen, c)
        end
    elseif d isa AbstractOneOf
        for c in d.delays
            _collect_types!(seen, c)
        end
    elseif d isa Choose
        for c in d.alternatives
            _collect_types!(seen, c)
        end
    end
    return nothing
end

@doc """

Assert the fixture registry covers every public composer type.

`test_registry_coverage(fixtures = example_fixtures())` checks that every type in
[`registry_types`](@ref) appears (possibly nested) in at least one fixture, so a
new public composer type added without a `test_interface` fixture fails here. The
walk descends composer children. Returns the `@testset` object.
""" function test_registry_coverage(fixtures = example_fixtures())
    covered = _covered_types(fixtures)
    is_covered(T) = any(c -> c <: T, covered)
    return @testset "fixture registry covers every public composer type" begin
        for T in registry_types()
            @test is_covered(T)
        end
    end
end

# --- construction rejection -------------------------------------------------

@doc """

Assert each composer rejects invalid construction in its inner constructor.

`test_rejects_invalid()` checks that the standard composers validate their
structural invariants on every construction path, so a malformed node errors at
build time rather than later: `Sequential` needs at least one component,
`Resolve` at least two outcomes with branch probabilities in `[0, 1]`, and
`Choose` at least two alternatives with unique names. Returns the `@testset`
object.
""" function test_rejects_invalid()
    G = Distributions.Gamma
    return @testset "construction rejects invalid input" begin
        # Resolve: branch probabilities out of range, and fewer than two outcomes.
        @test_throws ArgumentError resolve(:a => (G(1.0, 1.0), 0.9),
            :b => (G(1.0, 1.0), 0.9))
        @test_throws ArgumentError resolve(:only => (G(2.0, 1.0), 1.0))
        # Sequential: empty.
        @test_throws ArgumentError Sequential((), ())
        # Choose: needs at least two, and unique names.
        @test_throws ArgumentError choose(:only => G(2.0, 1.0))
        @test_throws ArgumentError choose(:a => G(2.0, 1.0), :a => G(1.0, 1.0))
    end
end

# --- composer-node contract -------------------------------------------------

@doc """

Assert a composer node satisfies the public node-extension contract.

`test_node_interface(node)` checks the three methods a new composer node
implements (`child_nleaves`, `child_logpdf`, `child_rand!`) round-trip on a flat
event vector, the same way the composers walk one. It asserts that

- `child_nleaves(node)` is a positive `Int`;
- `child_rand!` fills exactly the node's `offset + 1 : offset + n` slot, leaving
  any padding either side untouched;
- `child_logpdf(node, x, offset, n)` is a finite scalar on that drawn vector and
  does not depend on the surrounding padding.

Pass `offset` and `pad` to place the node inside a wider vector, and `rng` for a
reproducible draw. Returns the `@testset` object.

# Examples
```julia
using ComposedDistributions, Distributions
using ComposedDistributions.TestUtils: test_node_interface

node = compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
test_node_interface(node)
```
""" function test_node_interface end

function test_node_interface(node; name::AbstractString =
        string(nameof(typeof(node))),
        offset::Int = 1, pad::Int = 1,
        rng::AbstractRNG = Xoshiro(1))
    return @testset "node interface: $name" begin
        # child_nleaves: a positive flat-slot count.
        n = child_nleaves(node)
        @test n isa Int
        @test n >= 1

        # child_rand!: fills exactly the node's slot in a wider vector. A NaN
        # sentinel marks the untouched cells; only the node's slice should be
        # overwritten with finite draws.
        len = offset + n + pad
        out = fill(NaN, len)
        child_rand!(out, offset, rng, node)
        slot = (offset + 1):(offset + n)
        @test all(isfinite, @view out[slot])
        # The padding either side is left untouched.
        @test all(isnan, @view out[1:offset])
        @test all(isnan, @view out[(offset + n + 1):len])

        # child_logpdf: a finite scalar over the drawn slot.
        lp = child_logpdf(node, out, offset, n)
        @test lp isa Real
        @test isfinite(lp)

        # Position-independence: scoring the same draw at offset 0 in a tight
        # vector gives the same value (the node reads only its own slice).
        tight = out[slot]
        @test child_logpdf(node, tight, 0, n) ≈ lp
    end
end

# --- abstract-hierarchy conformance -----------------------------------------

@doc """

Assert a composed distribution satisfies the `AbstractComposedDistribution`
contract.

`test_composed_interface(node; kwargs...)` checks `node` subtypes
`AbstractComposedDistribution`, exposes `component_names`, and passes both the
node-extension checklist ([`test_node_interface`](@ref)) and the public interface
checklist ([`test_interface`](@ref); pass the same fixture keyword arguments).
Returns the `@testset` object.
""" function test_composed_interface end

function test_composed_interface(node; name::AbstractString =
        string(nameof(typeof(node))),
        ad_gradient = nothing, kwargs...)
    return @testset "composed interface: $name" begin
        @test node isa AbstractComposedDistribution
        @test component_names(node) isa Tuple
        test_node_interface(node; name = name)
        test_interface(node; name = name, ad_gradient = ad_gradient, kwargs...)
    end
end

@doc """

Assert the built-in composer types subtype the right family supertype.

`test_abstract_membership()` is the meta-test that the abstract hierarchy stays
consistent: every composer node subtypes `AbstractComposedDistribution`, the two
positional multi-child composers (`Sequential` / `Parallel`) subtype
`AbstractMultiChild`, and the one_of family (`Resolve` / `Compete`) subtypes
`AbstractOneOf`. `Choose` is a sibling, not a multi-child node. A type filed
under the wrong family fails here. Returns the `@testset` object.
""" function test_abstract_membership()
    return @testset "abstract hierarchy membership" begin
        for T in (Sequential, Parallel, Resolve, Compete, Choose)
            @test T <: AbstractComposedDistribution
        end
        # The two positional multi-child composers share `AbstractMultiChild`.
        @test Sequential <: AbstractMultiChild
        @test Parallel <: AbstractMultiChild
        @test !(Choose <: AbstractMultiChild)
        @test !(Resolve <: AbstractMultiChild)
        # The one_of family is its own supertype under the composed abstract.
        @test AbstractOneOf <: AbstractComposedDistribution
        @test Resolve <: AbstractOneOf
        @test Compete <: AbstractOneOf
        # The one_of family stays univariate; the event-tree composers stay
        # multivariate, so existing dispatch is unchanged.
        @test AbstractOneOf <: UnivariateDistribution
        @test Sequential <: Distribution{Multivariate}
        @test Parallel <: Distribution{Multivariate}
        @test Choose <: Distribution{Multivariate}
    end
end

end # module TestUtils
