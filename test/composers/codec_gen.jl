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
        shape = LogNormal(log(2.0), 0.2))
    tree = compose((onset = est, death = LogNormal(0.5, 0.4)))
    @test flat_dimension(tree) == 1

    x = [2.5]
    nt = unflatten(tree, x)
    @test nt == (onset = (shape = 2.5, scale = 3.0), death = (mu = 0.5, sigma = 0.4))
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

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    tied1 = shared(:g, u)
    tied2 = shared(:g, u)
    sub = compose((admit = LogNormal(0.5, 0.4), tied1 = tied1))
    root = compose((
        onset = uncertain(Gamma(1.5, 1.0); shape = LogNormal(0.0, 0.5)),
        sub = sub,
        tied2 = tied2))

    @test flat_dimension(root) == 2

    x = [2.3, 2.9]
    nt = unflatten(root, x)
    @test flatten(root, nt) == x
    @test !haskey(nt.sub, :tied1)
    @test nt.g.shape == 2.9

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
        shape = truncated(Normal(2.0, 1.0); lower = 0))
    tree = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        a = uncertain(Gamma(3.0, 1.0); shape = pool(:g, centred_pop)),
        b = uncertain(Gamma(3.0, 1.0); shape = pool(:g, centred_pop)),
        fixed = LogNormal(0.5, 0.4)))

    # district: mu, sigma, z_north, z_east (4); g: shape (1 hyper) + a, b (2
    # centred latents) = 3. Total 7.
    @test flat_dimension(tree) == 7

    x = [0.1, 0.5, 0.3, -0.2, 2.4, 3.0, 1.5]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test nt.north.shape == (z = 0.3,)
    @test nt.east.shape == (z = -0.2,)
    @test nt.a.shape == 3.0            # centred: bare value, not (z = ...)
    @test nt.b.shape == 1.5
    @test nt.district == (mu = 0.1, sigma = 0.5)
    @test nt.g == (shape = 2.4,)

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    @test params(event(collapsed, :north))[1] ≈ exp(0.1 + 0.5 * 0.3)
    @test params(event(collapsed, :east))[1] ≈ exp(0.1 + 0.5 * -0.2)
    @test params(event(collapsed, :a))[1] ≈ 3.0
    @test params(event(collapsed, :b))[1] ≈ 1.5
    @test event(collapsed, :fixed) == LogNormal(0.5, 0.4)
end

@testitem "codec: property round-trip -- Resolve stick-breaking nested in a tree" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    inner = update(
        resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7)),
        (branch_probs = Dirichlet(ones(2)),))
    tree = compose((
        onset = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
        outcome = inner))

    # onset.shape (1) + the K=2 node's K-1=1 stick coordinate.
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

    inc = shared(:inc, uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)))
    sel = choose(
        :index => inc,
        :sourced => compose((
            src = uncertain(LogNormal(0.5, 0.4);
                mu = Normal(0.5, 0.3)), inc = inc)))
    race = compete(:death => uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 0.3)),
        :recover => Gamma(1.5, 2.0))
    tree = compose((sel = sel, race = race))

    # inc.shape shared once (1) + sourced.src.mu (1) + race.death.shape (1).
    @test flat_dimension(tree) == 3

    x = [2.6, 0.7, 2.1]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test !haskey(nt.sel, :index)
    @test !haskey(nt.sel.sourced, :inc)
    @test nt.inc.shape == 2.6

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
# the same nested `NamedTuple`, both in value and in (fully concrete, still
# `@inferred`-stable) type. This item re-collects those same trees and
# expected results as one dedicated parity check, so a future generation-time
# change to this file has one place that fails loudly on any drift.
@testitem "codec: S2 layout parity -- unflatten is unchanged on every existing \
    codec_gen test tree" begin
    using ComposedDistributions: update, unflatten
    using Distributions

    cases = Any[]

    # Plain fixed tree (nothing estimated).
    let tree = compose((onset_admit = Gamma(2.0, 1.0),
            admit_death = LogNormal(0.5, 0.4)))
        x = Float64[]
        expected = (onset_admit = (shape = 2.0, scale = 1.0),
            admit_death = (mu = 0.5, sigma = 0.4))
        push!(cases, (tree, x, expected))
    end

    # A censored leaf's estimated parameter.
    let est = uncertain(censored(Gamma(2.0, 3.0); upper = 10.0);
            shape = LogNormal(log(2.0), 0.2))
        tree = compose((onset = est, death = LogNormal(0.5, 0.4)))
        x = [2.5]
        expected = (onset = (shape = 2.5, scale = 3.0),
            death = (mu = 0.5, sigma = 0.4))
        push!(cases, (tree, x, expected))
    end

    # Shared tags at different depths.
    let u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
        tied1 = shared(:g, u)
        tied2 = shared(:g, u)
        sub = compose((admit = LogNormal(0.5, 0.4), tied1 = tied1))
        root = compose((
            onset = uncertain(Gamma(1.5, 1.0); shape = LogNormal(0.0, 0.5)),
            sub = sub,
            tied2 = tied2))
        x = [2.3, 2.9]
        expected = (onset = (shape = 2.3, scale = 1.0),
            sub = (admit = (mu = 0.5, sigma = 0.4),),
            g = (shape = 2.9, scale = 1.0))
        push!(cases, (root, x, expected))
    end

    # Non-centred and centred pools together.
    let centred_pop = uncertain(Gamma(2.0, 1.0);
            shape = truncated(Normal(2.0, 1.0); lower = 0))
        tree = compose((
            north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
            east = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
            a = uncertain(Gamma(3.0, 1.0); shape = pool(:g, centred_pop)),
            b = uncertain(Gamma(3.0, 1.0); shape = pool(:g, centred_pop)),
            fixed = LogNormal(0.5, 0.4)))
        x = [0.1, 0.5, 0.3, -0.2, 2.4, 3.0, 1.5]
        expected = (north = (shape = (z = 0.3,), scale = 1.0),
            east = (shape = (z = -0.2,), scale = 1.0),
            a = (shape = 3.0, scale = 1.0),
            b = (shape = 1.5, scale = 1.0),
            fixed = (mu = 0.5, sigma = 0.4),
            district = (mu = 0.1, sigma = 0.5),
            g = (shape = 2.4,))
        push!(cases, (tree, x, expected))
    end

    # Resolve stick-breaking nested in a tree.
    let inner = update(
            resolve(:death => (Gamma(1.5, 1.0), 0.3),
                :disch => (Gamma(2.0, 1.5), 0.7)),
            (branch_probs = Dirichlet(ones(2)),))
        tree = compose((
            onset = uncertain(
                Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
            outcome = inner))
        x = [2.2, 0.4]
        expected = (onset = (shape = 2.2, scale = 1.0),
            outcome = (death = (shape = 1.5, scale = 1.0),
                disch = (shape = 2.0, scale = 1.5),
                branch_probs = (stick_1 = 0.4,)))
        push!(cases, (tree, x, expected))
    end

    # Choose alternatives with a shared tag, and Compete.
    let inc = shared(
            :inc, uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)))
        sel = choose(
            :index => inc,
            :sourced => compose((
                src = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.3)),
                inc = inc)))
        race = compete(
            :death => uncertain(Gamma(2.0, 1.0); shape = LogNormal(0.0, 0.3)),
            :recover => Gamma(1.5, 2.0))
        tree = compose((sel = sel, race = race))
        x = [2.6, 0.7, 2.1]
        expected = (sel = (sourced = (src = (mu = 0.7, sigma = 0.4),),),
            race = (death = (shape = 2.1, scale = 1.0),
                recover = (shape = 1.5, scale = 2.0)),
            inc = (shape = 2.6, scale = 1.0))
        push!(cases, (tree, x, expected))
    end

    for (tree, x, expected) in cases
        nt = @inferred unflatten(tree, x)
        @test nt == expected
        @test typeof(nt) == typeof(expected)
    end
end
