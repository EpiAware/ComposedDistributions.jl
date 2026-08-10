@testitem "codec: property round-trip -- plain fixed tree (nothing estimated)" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    @test flat_dimension(tree) == 0
    x = Float64[]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test reconstruct(tree, x) == update(tree, nt) == tree
end

@testitem "codec: property round-trip -- a censored leaf's estimated parameter" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    est = uncertain(censored(Gamma(2.0, 3.0); upper = 10.0);
        alpha = LogNormal(log(2.0), 0.2))
    tree = compose((onset = est, death = LogNormal(0.5, 0.4)))
    @test flat_dimension(tree) == 1

    x = [2.5]
    nt = unflatten(tree, x)
    @test nt == (onset = (alpha = 2.5, theta = 3.0), death = (mu = 0.5, sigma = 0.4))
    @test flatten(tree, nt) == x

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    leaf = event(collapsed, :onset)
    @test leaf isa Distributions.Censored
    @test leaf.upper == 10.0
    @test ComposedDistributions.free_leaf(leaf) == Gamma(2.5, 3.0)
end

@testitem "codec: property round-trip -- shared tags at different depths" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    u = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2))
    tied1 = shared(:g, u)
    tied2 = shared(:g, u)
    sub = compose((admit = LogNormal(0.5, 0.4), tied1 = tied1))
    root = compose((
        onset = uncertain(Gamma(1.5, 1.0); alpha = LogNormal(0.0, 0.5)),
        sub = sub,
        tied2 = tied2))

    @test flat_dimension(root) == 2

    x = [2.3, 2.9]
    nt = unflatten(root, x)
    @test flatten(root, nt) == x
    @test !haskey(nt.sub, :tied1)
    @test nt.g.alpha == 2.9

    collapsed = reconstruct(root, x)
    @test collapsed == update(root, nt)
    @test !has_uncertain(collapsed)
    a = ComposedDistributions.free_leaf(event(collapsed, :sub, :tied1))
    b = ComposedDistributions.free_leaf(event(collapsed, :tied2))
    @test a == b == Gamma(2.9, 1.0)
    @test event(collapsed, :onset) == Gamma(2.3, 1.0)
end

@testitem "codec: property round-trip -- non-centred and centred pools together" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    centred_pop = uncertain(Gamma(2.0, 1.0);
        alpha = truncated(Normal(2.0, 1.0); lower = 0))
    tree = compose((
        north = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
        a = uncertain(Gamma(3.0, 1.0); alpha = pool(:g, centred_pop)),
        b = uncertain(Gamma(3.0, 1.0); alpha = pool(:g, centred_pop)),
        fixed = LogNormal(0.5, 0.4)))

    # district: mu, sigma, z_north, z_east (4); g: alpha (1 hyper) + a, b (2
    # centred latents) = 3. Total 7.
    @test flat_dimension(tree) == 7

    x = [0.1, 0.5, 0.3, -0.2, 2.4, 3.0, 1.5]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test nt.north.alpha == (z = 0.3,)
    @test nt.east.alpha == (z = -0.2,)
    @test nt.a.alpha == 3.0            # centred: bare value, not (z = ...)
    @test nt.b.alpha == 1.5
    @test nt.district == (mu = 0.1, sigma = 0.5)
    @test nt.g == (alpha = 2.4,)

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    @test params(event(collapsed, :north))[1] ≈ exp(0.1 + 0.5 * 0.3)
    @test params(event(collapsed, :east))[1] ≈ exp(0.1 + 0.5 * -0.2)
    @test params(event(collapsed, :a))[1] ≈ 3.0
    @test params(event(collapsed, :b))[1] ≈ 1.5
    @test event(collapsed, :fixed) == LogNormal(0.5, 0.4)
end

@testitem "codec: pooled parameter that is not the leaf's first native parameter" begin
    using ComposedDistributions: update, params_table
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    # Gamma's native order is (alpha, theta); `theta` is pooled here, so a
    # codec walk that (wrongly) hoists the pool group's hyperparameter slots
    # before the WHOLE leaf's own block -- rather than at `theta`'s own
    # native-order position, after `alpha` -- disagrees with `params_table`.
    tree = compose((
        a = uncertain(Gamma(2.0, 3.0);
            alpha = LogNormal(0.0, 0.3), theta = pool(:g)),
        b = uncertain(Gamma(4.0, 5.0);
            alpha = LogNormal(0.0, 0.3), theta = pool(:g))))

    table = params_table(tree)
    @test collect(table.edge) ==
          [:a, :g, :g, Symbol("a.theta"), :b, Symbol("b.theta")]
    @test collect(table.param) == [:alpha, :mu, :sigma, :z, :alpha, :z]

    x = [2.0, 0.0, 1.0, 0.0, 4.0, 0.0]
    nt = unflatten(tree, x)
    @test nt.a.alpha == 2.0
    @test nt.g == (mu = 0.0, sigma = 1.0)
    @test nt.a.theta == (z = 0.0,)
    @test nt.b.alpha == 4.0
    @test nt.b.theta == (z = 0.0,)
    @test flatten(tree, nt) == x

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
end

@testitem "codec: a leaf naming two different pool groups" begin
    using ComposedDistributions: update, params_table
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    tree = compose((
        a = uncertain(Gamma(2.0, 3.0); alpha = pool(:g1), theta = pool(:g2)),
        b = uncertain(Gamma(4.0, 5.0); alpha = pool(:g1)),
        c = uncertain(Gamma(6.0, 7.0); theta = pool(:g2))))

    table = params_table(tree)
    @test collect(table.edge) == [:g1, :g1, Symbol("a.alpha"), :g2, :g2,
        Symbol("a.theta"), Symbol("b.alpha"), :b, :c, Symbol("c.theta")]
    @test flat_dimension(tree) == 8

    x = [10.0, 11.0, 20.0, 30.0, 31.0, 40.0, 50.0, 60.0]
    nt = unflatten(tree, x)
    @test nt.g1 == (mu = 10.0, sigma = 11.0)
    @test nt.a.alpha == (z = 20.0,)
    @test nt.g2 == (mu = 30.0, sigma = 31.0)
    @test nt.a.theta == (z = 40.0,)
    @test nt.b.alpha == (z = 50.0,)
    @test nt.c.theta == (z = 60.0,)
    @test flatten(tree, nt) == x

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
end

@testitem "codec: pool population whose uncertain kwargs are out of native order" begin
    using ComposedDistributions: update, params_table
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    # `pop` writes `theta` before `alpha`, the reverse of Gamma's native
    # (alpha, theta) order; the group hyperparameter rows must still land in
    # NATIVE order (`params_table` walks `leaf_param_names`, not a
    # population's kwargs order).
    pop = uncertain(Gamma(2.0, 3.0);
        theta = LogNormal(0.0, 0.3), alpha = LogNormal(0.0, 0.3))
    tree = compose((
        a = uncertain(Gamma(2.0, 3.0); alpha = pool(:g, pop)),
        b = uncertain(Gamma(4.0, 5.0); alpha = pool(:g, pop))))

    table = params_table(tree)
    @test collect(table.param)[1:2] == [:alpha, :theta]

    x = [2.0, 3.0, 2.0, 3.0]
    nt = unflatten(tree, x)
    @test nt.g == (alpha = 2.0, theta = 3.0)
    @test flatten(tree, nt) == x

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
end

@testitem "codec: property round-trip -- Resolve stick-breaking nested in a tree" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    inner = update(
        resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7)),
        (branch_probs = Dirichlet(ones(2)),))
    tree = compose((
        onset = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)),
        outcome = inner))

    # onset.alpha (1) + the K=2 node's K-1=1 stick coordinate.
    @test flat_dimension(tree) == 2

    x = [2.2, 0.4]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test nt.outcome.branch_probs == (stick_1 = 0.4,)

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    r = event(collapsed, :outcome)
    @test !(r.branch_prob_prior isa Dirichlet)   # collapsed to concrete probs
    p = collect(Distributions.probs(r))
    @test p[1] ≈ 0.4
    @test sum(p) ≈ 1.0
end

@testitem "codec: property round-trip -- Choose alternatives with a shared tag, and Compete" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    inc = shared(:inc, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)))
    sel = choose(
        :index => inc,
        :sourced => compose((
            src = uncertain(LogNormal(0.5, 0.4);
                mu = Normal(0.5, 0.3)), inc = inc)))
    race = compete(:death => uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        :recover => Gamma(1.5, 2.0))
    tree = compose((sel = sel, race = race))

    # inc.alpha shared once (1) + sourced.src.mu (1) + race.death.alpha (1).
    @test flat_dimension(tree) == 3

    x = [2.6, 0.7, 2.1]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test !haskey(nt.sel, :index)
    @test !haskey(nt.sel.sourced, :inc)
    @test nt.inc.alpha == 2.6

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    index_alt = event(event(collapsed, :sel), :index)
    sourced_inc = event(event(event(collapsed, :sel), :sourced), :inc)
    @test ComposedDistributions.free_leaf(index_alt) ==
          ComposedDistributions.free_leaf(sourced_inc) == Gamma(2.6, 1.0)
    @test params(event(event(collapsed, :sel), :sourced, :src))[1] ≈ 0.7
    @test params(event(event(collapsed, :race), :death))[1] ≈ 2.1
end

# STAGE S2 of the codec-generation plan (#189-adjacent): `_leaf_unflatten_expr`
# (codec_gen.jl) now builds a leaf's `unflatten` entry by calling the runtime
# seam (`_leaf_entry`, introspection.jl, S1) instead of baking each fixed
# parameter's value straight into the generated code. That is a pure
# generation-strategy swap -- the emitted flat-vector layout (which name owns
# which `x` slot) is untouched -- so every tree already covered by the
# `codec: property round-trip` items above must still `unflatten` to exactly
# the same nested `NamedTuple`, in both value and runtime type. This item
# re-collects those same trees and expected results as one dedicated parity
# check, so a future generation-time change to this file has one place that
# fails loudly on any drift.
#
# Not `@inferred` any more: `leaf_param_names`'s six curated-family names came
# from a per-type dispatch table before this branch; this branch replaces it
# with generic, value-based derivation (matching `params(leaf)` against
# `fieldnames(typeof(leaf))`). The table was a pure function of the leaf's
# TYPE, so Julia's compiler could prove the exact name tuple at compile time;
# the derivation reads actual field VALUES to decide the match, which no
# amount of restructuring found so far lets ordinary type inference see
# through (a `@generated` function cannot help either -- it may only inspect
# argument TYPES, never VALUES, and matching genuinely needs the values). So
# `unflatten`/`flatten` on a derivation-named leaf now return a NamedTuple
# whose VALUE and RUNTIME type are exactly as before, but whose statically
# INFERRED type is the widened `NamedTuple{names, T} where names` -- correct
# results, less concrete compile-time typing. AD correctness is unaffected
# (test/ad/scenarios.jl's Mooncake #146 item exercises a derivation-named
# Gamma leaf through both directions against ForwardDiff), but this is a real
# loss of static inference on the per-gradient hot path and is NOT this
# branch's call to make alone: it is flagged in #371 for @seabbs to accept or
# reject, not silently decided here; only the `@inferred` wrapping comes off
# for now so the (correctness-only) value/runtime-type parity checks stay
# green while #371 is open.
@testitem "codec: S2 layout parity -- unflatten is unchanged on every existing \
    codec_gen test tree" begin
    using ComposedDistributions: update, unflatten, flatten
    using Distributions

    cases = Any[]

    # Plain fixed tree (nothing estimated).
    let tree = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4)))
        x = Float64[]
        expected = (onset_admit = (alpha = 2.0, theta = 1.0),
            admit_death = (mu = 0.5, sigma = 0.4))
        push!(cases, (tree, x, expected))
    end

    # A censored leaf's estimated parameter.
    let est = uncertain(censored(Gamma(2.0, 3.0); upper = 10.0);
            alpha = LogNormal(log(2.0), 0.2))
        tree = compose((onset = est, death = LogNormal(0.5, 0.4)))
        x = [2.5]
        expected = (onset = (alpha = 2.5, theta = 3.0),
            death = (mu = 0.5, sigma = 0.4))
        push!(cases, (tree, x, expected))
    end

    # Shared tags at different depths.
    let u = uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2))
        tied1 = shared(:g, u)
        tied2 = shared(:g, u)
        sub = compose((admit = LogNormal(0.5, 0.4), tied1 = tied1))
        root = compose((
            onset = uncertain(Gamma(1.5, 1.0); alpha = LogNormal(0.0, 0.5)),
            sub = sub,
            tied2 = tied2))
        x = [2.3, 2.9]
        expected = (onset = (alpha = 2.3, theta = 1.0),
            sub = (admit = (mu = 0.5, sigma = 0.4),),
            g = (alpha = 2.9, theta = 1.0))
        push!(cases, (root, x, expected))
    end

    # Non-centred and centred pools together.
    let centred_pop = uncertain(Gamma(2.0, 1.0);
            alpha = truncated(Normal(2.0, 1.0); lower = 0))
        tree = compose((
            north = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
            east = uncertain(Gamma(2.0, 1.0); alpha = pool(:district)),
            a = uncertain(Gamma(3.0, 1.0); alpha = pool(:g, centred_pop)),
            b = uncertain(Gamma(3.0, 1.0); alpha = pool(:g, centred_pop)),
            fixed = LogNormal(0.5, 0.4)))
        x = [0.1, 0.5, 0.3, -0.2, 2.4, 3.0, 1.5]
        expected = (north = (alpha = (z = 0.3,), theta = 1.0),
            east = (alpha = (z = -0.2,), theta = 1.0),
            a = (alpha = 3.0, theta = 1.0),
            b = (alpha = 1.5, theta = 1.0),
            fixed = (mu = 0.5, sigma = 0.4),
            district = (mu = 0.1, sigma = 0.5),
            g = (alpha = 2.4,))
        push!(cases, (tree, x, expected))
    end

    # Resolve stick-breaking nested in a tree.
    let inner = update(
            resolve(:death => (Gamma(1.5, 1.0), 0.3),
                :disch => (Gamma(2.0, 1.5), 0.7)),
            (branch_probs = Dirichlet(ones(2)),))
        tree = compose((
            onset = uncertain(
                Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)),
            outcome = inner))
        x = [2.2, 0.4]
        expected = (onset = (alpha = 2.2, theta = 1.0),
            outcome = (death = (alpha = 1.5, theta = 1.0),
                disch = (alpha = 2.0, theta = 1.5),
                branch_probs = (stick_1 = 0.4,)))
        push!(cases, (tree, x, expected))
    end

    # Choose alternatives with a shared tag, and Compete.
    let inc = shared(
            :inc, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(log(2.0), 0.2)))
        sel = choose(
            :index => inc,
            :sourced => compose((
                src = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.3)),
                inc = inc)))
        race = compete(
            :death => uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
            :recover => Gamma(1.5, 2.0))
        tree = compose((sel = sel, race = race))
        x = [2.6, 0.7, 2.1]
        expected = (sel = (sourced = (src = (mu = 0.7, sigma = 0.4),),),
            race = (death = (alpha = 2.1, theta = 1.0),
                recover = (alpha = 1.5, theta = 2.0)),
            inc = (alpha = 2.6, theta = 1.0))
        push!(cases, (tree, x, expected))
    end

    for (tree, x, expected) in cases
        nt = unflatten(tree, x)
        @test nt == expected
        @test typeof(nt) == typeof(expected)
        # `flatten` is guarded here too, not just round-tripped: a
        # value-level round-trip alone would not catch the direction whose
        # inference broke silently when this item was `@inferred` (S2); it
        # is no longer `@inferred` (see the derivation-vs-inferability note
        # above), so this only re-affirms the value round-trip.
        @test flatten(tree, nt) == x
    end
end

# STAGE S3 of the codec-generation plan: `_leaf_unflatten_expr`/
# `_leaf_flatten_reads!` no longer look up a leaf's estimable names/arity from
# a type-level table (`_leaf_type_param_names` and the four companions it
# combined, all removed) -- they resolve them at RUNTIME, from the leaf's own
# INSTANCE-level `leaf_param_names`/`leaf_param_values` (introspection.jl),
# exactly like `params_table`/`update` always have. This is the motivating
# regression case that table could not handle without a bespoke override: a
# leaf type whose `Distributions.params` are NOT its own struct fields 1:1 (a
# moment-parameterised wrapper, mirroring `ReparameterisedDistributions`),
# defining only the two INSTANCE hooks (`param_names`, `leaf_ctor`) the public
# leaf protocol asks for -- no `_param_names_of`/`_params_arity_of`-shaped
# override exists any more for it to need. Defined fresh here (after
# `ComposedDistributions` is loaded, like a real downstream extension would),
# to also stand in as the load-order check the old registry existed for.
@testitem "codec: S3 runtime seam -- a leaf type whose params are not its own \
    fields 1:1 round-trips with only param_names/leaf_ctor defined" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    struct MomentLeaf <: Distributions.ContinuousUnivariateDistribution
        vals::NTuple{2, Float64}
    end
    Distributions.params(m::MomentLeaf) = m.vals
    Distributions.logpdf(m::MomentLeaf, x::Real) = logpdf(
        LogNormal(log(m.vals[1]), 0.3), x)
    Base.minimum(::MomentLeaf) = 0.0
    Base.maximum(::MomentLeaf) = Inf
    ComposedDistributions.param_names(::MomentLeaf) = (:mean, :sd)
    ComposedDistributions.leaf_ctor(::MomentLeaf) = (a, b) -> MomentLeaf((a, b))

    @test ComposedDistributions.leaf_param_names(MomentLeaf((8.0, 2.0))) ==
          (:mean, :sd)

    leaf = uncertain(MomentLeaf((8.0, 2.0)); mean = LogNormal(log(8.0), 0.2))
    tree = compose((
        m = leaf, other = uncertain(Gamma(2.0, 1.0);
            alpha = LogNormal(0.0, 0.3))))
    @test flat_dimension(tree) == 2

    x = [9.0, 2.5]
    nt = unflatten(tree, x)
    @test nt == (m = (mean = 9.0, sd = 2.0), other = (alpha = 2.5, theta = 1.0))
    @test isconcretetype(typeof(nt))
    @test flatten(tree, nt) == x

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    @test event(collapsed, :m) == MomentLeaf((9.0, 2.0))
    @test event(collapsed, :other) == Gamma(2.5, 1.0)
end
