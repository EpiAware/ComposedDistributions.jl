# Property tests for the generated type-domain codec (#178 PR 2):
# `flatten(d, unflatten(d, x)) == x` and
# `reconstruct(d, x) == update(d, unflatten(d, x))` across every composer
# shape the design review's migration gate lists -- nested shared tags at
# different depths, non-centred AND centred pools, Resolve stick-breaking,
# thin extras, and Choose/Compete. These are the correctness gate for
# replacing the old Dict-based `unflatten`/`params_table` walk with the
# generation-time layout walk in `codec_gen.jl`.

@testitem "codec: property round-trip -- plain fixed tree (nothing estimated)" begin
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

@testitem "codec: property round-trip -- shared tags at different depths" begin
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

    # onset.shape (1) + the tied g.shape, counted once despite two occurrences
    # at two different depths (root-level `tied2` and depth-2 `sub.tied1`).
    @test flat_dimension(root) == 2

    x = [2.3, 2.9]
    nt = unflatten(root, x)
    @test flatten(root, nt) == x
    @test !haskey(nt.sub, :tied1)   # tagged leaf omitted positionally
    @test nt.g.shape == 2.9         # root-lifted under the tag, not the path

    collapsed = reconstruct(root, x)
    @test collapsed == update(root, nt)
    @test !has_uncertain(collapsed)
    a = ComposedDistributions.free_leaf(event(collapsed, :sub, :tied1))
    b = ComposedDistributions.free_leaf(event(collapsed, :tied2))
    @test a == b == Gamma(2.9, 1.0)
    @test event(collapsed, :onset) == Gamma(2.3, 1.0)
end

@testitem "codec: property round-trip -- non-centred and centred pools together" begin
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

@testitem "codec: property round-trip -- thin extras through the generated codec" begin
    using Distributions
    using ModifiedDistributions: thin
    using ComposedDistributions: unflatten, flatten, flat_dimension, reconstruct

    leaf = uncertain(thin(Gamma(2.0, 1.0), 0.3);
        shape = LogNormal(log(2.0), 0.2), thin = Beta(2.0, 2.0))
    tree = compose((onset = leaf, admit = LogNormal(0.5, 0.4)))

    @test flat_dimension(tree) == 2   # onset.shape, onset.thin

    x = [2.4, 0.45]
    nt = unflatten(tree, x)
    @test flatten(tree, nt) == x
    @test nt.onset.shape == 2.4
    @test nt.onset.thin == 0.45

    collapsed = reconstruct(tree, x)
    @test collapsed == update(tree, nt)
    onset_dist = event(collapsed, :onset)
    @test ComposedDistributions.free_leaf(onset_dist) == Gamma(2.4, 1.0)
    @test ComposedDistributions.extra_leaf_params(onset_dist).thin.value ≈ 0.45
end
