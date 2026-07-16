# ComposedDistributions × DynamicPPL: `as_turing(dist, data)` builds a
# DynamicPPL model over a composed distribution's estimated parameters, a light
# wrapper on the `as_logdensity` codec. The extension loads when DynamicPPL
# alone is present. These tests prove (a) the model's `~` site names match the
# FlexiChains readback exactly, so a fitted chain reads back through
# `chain_to_params` / `update(dist, chain)` unchanged; (b) the model's total
# log-density equals the codec's `logdensity` by construction; (c) a
# non-centred pooled tree samples and reads back through `update(dist, chain)`
# like any other tree; and (d) a centred-pool tree is rejected with a clear
# pointer to the codec path.

@testitem "as_turing extension loads under DynamicPPL alone" begin
    using ComposedDistributions, DynamicPPL
    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsDynamicPPLExt) !== nothing
end

@testitem "as_turing round-trip: NUTS chain reads back through the codec" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    tree = compose((
        onset_admit = uncertain(Gamma(2.0, 1.0);
            shape = LogNormal(log(2.0), 0.2)),
        admit_death = LogNormal(0.5, 0.4)))
    data = [[0.5, 2.0], [1.0, 3.0], [0.8, 2.5]]

    model = as_turing(tree, data)

    # The model's total log-density equals the codec's `logdensity` at the
    # corresponding constrained point (the "built on the codec" check).
    # Conditioning every site at its value scores each prior plus the
    # `@addlogprob!` likelihood, exactly what `logdensity` sums.
    prob = ComposedDistributions.as_logdensity(tree, data)
    x = [2.3]
    cm = DynamicPPL.condition(model, @varname(d.onset_admit.shape) => x[1])
    @test DynamicPPL.logjoint(cm, DynamicPPL.VarInfo(cm)) ≈
          ComposedDistributions.logdensity(prob, x)

    Random.seed!(1)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    # The single estimated parameter is sampled at the readback's dotted name.
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.onset_admit.shape" in vns

    # The chain reads back through the EXISTING FlexiChains machinery unchanged:
    # `chain_to_params` reduces it to the nested NamedTuple, `update` rebuilds
    # the tree, collapsing the uncertain leaf.
    params = chain_to_params(tree, chain)
    @test keys(params) == (:onset_admit, :admit_death)
    @test haskey(params.onset_admit, :shape)

    fit = update(tree, chain)
    @test event(fit, :onset_admit) isa Gamma
    @test event(fit, :admit_death) isa LogNormal
    @test !has_uncertain(fit)
end

@testitem "as_turing round-trip: uncertain branch_probs stick coordinate" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    # A Resolve whose branch-probability simplex is uncertain (K = 2, so one
    # stick coordinate). The stick is the estimated parameter; the sampled site
    # must carry the readback name `d.resolution.branch_probs.stick_1`.
    tree = compose((resolution = update(
        resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)),
        (branch_probs = Dirichlet([1.0, 1.0]),)),))
    # A single-branch `compose` NamedTuple is a `Parallel`, so each record is a
    # one-element event vector.
    data = [[0.8], [1.5], [2.2], [0.6]]

    Random.seed!(7)
    chain = sample(as_turing(tree, data), NUTS(), 200; chain_type = VNChain,
        progress = false)

    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.resolution.branch_probs.stick_1" in vns

    nt = chain_to_params(tree, chain)
    @test keys(nt.resolution.branch_probs) == (:stick_1,)

    fitted = update(tree, chain)
    @test !has_uncertain(fitted)
    p = collect(values(probs(event(fitted, :resolution))))
    @test sum(p) ≈ 1.0
end

@testitem "as_turing rejects a centred pooled tree" begin
    using ComposedDistributions, Distributions

    data = [[0.5, 2.0], [1.0, 3.0]]

    # A centred pool (a general, non-location-scale population) has a
    # hyperparameter-dependent member prior, so no fixed `~` prior at all; its
    # sampling path does not exist yet, so it stays rejected.
    centred = compose((
        north = uncertain(Gamma(2.0, 1.0);
            shape = pool(:district, Gamma(2.0, 1.0); noncentred = false)),
        south = uncertain(Gamma(2.0, 1.0);
            shape = pool(:district, Gamma(2.0, 1.0); noncentred = false))))
    @test_throws ArgumentError as_turing(centred, data)
end

@testitem "as_turing round-trip: non-centred pooled tree" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    # A non-centred (location-scale) pool samples correctly and the readback
    # now consumes its pooled chain, so it round-trips through
    # `update(tree, chain)` like an ordinary uncertain tree.
    tree = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))
    data = [[0.5, 2.0], [1.0, 3.0]]

    model = as_turing(tree, data)

    Random.seed!(13)
    chain = sample(model, NUTS(), 200; chain_type = VNChain, progress = false)

    # The hyperparameters and each member's latent are sampled at the
    # readback's dotted names.
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.district.mu" in vns
    @test "d.district.sigma" in vns
    @test "d.north.shape.z" in vns
    @test "d.south.shape.z" in vns

    fitted = update(tree, chain)
    @test !has_uncertain(fitted)
    @test event(fitted, :north) isa Gamma
    @test event(fitted, :south) isa Gamma
end
