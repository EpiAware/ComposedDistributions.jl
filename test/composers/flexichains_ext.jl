# ComposedDistributions × DynamicPPL × FlexiChains: read a fitted chain's
# parameters back onto a composed-distribution template. The extension loads
# once both DynamicPPL and FlexiChains are present. This package has no
# Turing-model builder of its own (that stays a downstream extension's job),
# so these tests hand-build a small two-level `to_submodel` model matching the
# `d.<edge>.<param>` naming such a builder would produce, and sample it with
# Turing's `Prior()` (fast, no gradients) purely to get a real chain to read
# back.

@testitem "chain_to_params / param_draws / update(template, chain) round-trip" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain
    using Statistics: mean

    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsFlexiChainsExt) !== nothing

    template = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    @model function onset_admit_model()
        shape ~ truncated(Normal(2.0, 0.3); lower = 0)
        scale ~ truncated(Normal(1.0, 0.3); lower = 0)
        return (shape = shape, scale = scale)
    end

    @model function admit_death_model()
        mu ~ Normal(0.5, 0.2)
        sigma ~ truncated(Normal(0.4, 0.1); lower = 0)
        return (mu = mu, sigma = sigma)
    end

    @model function tree_model()
        onset_admit ~ to_submodel(onset_admit_model())
        admit_death ~ to_submodel(admit_death_model())
        return (onset_admit = onset_admit, admit_death = admit_death)
    end

    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(51)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)

    # Sampled VarNames carry the edge path under the `d` submodel prefix.
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.onset_admit.shape" in vns
    @test "d.admit_death.mu" in vns

    means = chain_to_params(template, chain)
    @test keys(means) == (:onset_admit, :admit_death)
    @test keys(means.onset_admit) == (:shape, :scale)

    # The default `summary = mean` reduction matches a hand-computed mean of
    # the raw draws.
    shape_vn = only(filter(v -> string(v) == "d.onset_admit.shape",
        collect(FlexiChains.parameters(chain))))
    @test means.onset_admit.shape ≈ mean(vec(chain[shape_vn]))

    # `draw = i` reads exactly the i-th iteration (no reduction).
    d3 = chain_to_params(template, chain; draw = 3)
    @test d3.onset_admit.shape == vec(chain[shape_vn])[3]

    # `update(template, chain)` is `update(template, chain_to_params(...))` in
    # one call.
    @test update(template, chain) == update(template, means)
    fitted = update(template, chain)
    @test event(fitted, :onset_admit) isa Gamma
    @test event(fitted, :admit_death) isa LogNormal

    # `param_draws` is the vectorised, every-draw form of `chain_to_params`.
    draws = param_draws(template, chain)
    @test length(draws) == 30
    @test draws[5] == chain_to_params(template, chain; draw = 5)
    @test all(d -> update(template, d) isa typeof(template), draws)

    # `draws` restricts to a subset of iterations (a range, or a predicate).
    @test length(param_draws(template, chain; draws = 1:5)) == 5
    @test length(param_draws(template, chain; draws = iseven)) ==
          count(iseven, 1:30)
end

@testitem "strip_prefix drops the outer submodel prefix" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    @model function onset_admit_model()
        shape ~ truncated(Normal(2.0, 0.3); lower = 0)
        scale ~ truncated(Normal(1.0, 0.3); lower = 0)
        return (shape = shape, scale = scale)
    end

    @model function tree_model()
        onset_admit ~ to_submodel(onset_admit_model())
        return (onset_admit = onset_admit,)
    end

    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(53)
    chain = sample(full_model(), Prior(), 20; chain_type = VNChain,
        progress = false)

    stripped = strip_prefix(chain)
    names_before = Set(string.(collect(FlexiChains.parameters(chain))))
    names_after = Set(string.(collect(FlexiChains.parameters(stripped))))
    @test "d.onset_admit.shape" in names_before
    @test "onset_admit.shape" in names_after
    @test !("d.onset_admit.shape" in names_after)

    # Reading the stripped chain with an empty prefix agrees with reading the
    # original chain at the default `:d` prefix.
    template = compose((onset_admit = Gamma(2.0, 1.0),))
    means_prefixed = chain_to_params(template, chain)
    means_stripped = chain_to_params(
        template, stripped; prefix = Symbol(""))
    @test means_prefixed == means_stripped
end

@testitem "chain_to_params collapses an Uncertain leaf; a Varying leaf keeps varying" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: VNChain

    @model function onset_admit_model()
        shape ~ truncated(Normal(2.0, 0.3); lower = 0)
        scale ~ truncated(Normal(1.0, 0.3); lower = 0)
        return (shape = shape, scale = scale)
    end

    @model function tree_model()
        onset_admit ~ to_submodel(onset_admit_model())
        return (onset_admit = onset_admit,)
    end

    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(52)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)

    # An Uncertain leaf collapses to a concrete leaf at the read values, the
    # same collapse `update` always performs when given concrete parameters.
    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    u_fitted = update(compose((onset_admit = u,)), chain)
    @test !has_uncertain(event(u_fitted, :onset_admit))
    @test event(u_fitted, :onset_admit) isa Gamma

    # A Varying leaf stays a Varying: only its `reference`'s fixed values are
    # updated; `instantiate` is what resolves the covariate.
    v = varying(t -> Gamma(2.0, 1.0))
    v_fitted = update(compose((onset_admit = v,)), chain)
    @test has_varying(v_fitted)
    @test event(v_fitted, :onset_admit) isa Varying
end

@testitem "chain_to_params reads uncertain branch_probs stick coordinates" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    # A Resolve whose branch-probability simplex is uncertain (K = 2, so one
    # stick coordinate), inside a tree. The stick is what the sampler estimates;
    # readback must find it at `d.resolution.branch_probs.stick_1`.
    template = compose((resolution = update(
        resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)),
        (branch_probs = Dirichlet([1.0, 1.0]),)),))

    @model function branch_probs_model()
        stick_1 ~ Beta(1.0, 1.0)
        return (stick_1 = stick_1,)
    end
    @model function resolution_model()
        branch_probs ~ to_submodel(branch_probs_model())
        return (branch_probs = branch_probs,)
    end
    @model function tree_model()
        resolution ~ to_submodel(resolution_model())
        return (resolution = resolution,)
    end
    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(89)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.resolution.branch_probs.stick_1" in vns

    nt = chain_to_params(template, chain)
    @test keys(nt.resolution.branch_probs) == (:stick_1,)

    # Reading back collapses the uncertain node to concrete probabilities that
    # sum to one and are the stick-breaking of the read coordinate.
    fitted = update(template, chain)
    @test !has_uncertain(fitted)
    p = collect(values(probs(event(fitted, :resolution))))
    @test sum(p) ≈ 1.0
    v1 = nt.resolution.branch_probs.stick_1
    @test p[1] ≈ v1
    @test p[2] ≈ 1 - v1
end

@testitem "chain_to_params reads a non-centred pooled tree (fixed population)" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain
    using Statistics: mean

    # A fixed (non-`uncertain`) location-scale population contributes no
    # hyperparameter rows: only each member's `z` latent is estimated.
    template = compose((
        north = uncertain(Gamma(2.0, 1.0);
            shape = pool(:district, Normal(0.7, 0.3))),
        south = uncertain(Gamma(2.0, 1.0);
            shape = pool(:district, Normal(0.7, 0.3)))))

    @model function shape_z_model()
        z ~ Normal(0.0, 1.0)
        return (z = z,)
    end
    @model function north_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function south_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function tree_model()
        north ~ to_submodel(north_model())
        south ~ to_submodel(south_model())
        return (north = north, south = south)
    end
    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(97)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.north.shape.z" in vns
    @test "d.south.shape.z" in vns

    nt = chain_to_params(template, chain)
    @test keys(nt.north.shape) == (:z,)

    fitted = update(template, chain)
    @test !has_uncertain(fitted)
    z_north_vn = only(filter(v -> string(v) == "d.north.shape.z",
        collect(FlexiChains.parameters(chain))))
    z_north = mean(vec(chain[z_north_vn]))
    @test params(event(fitted, :north))[1] ≈ 0.7 + 0.3 * z_north
end

@testitem "chain_to_params reads pooled hyperparameters once per group" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain

    # The default pool population is an estimated LogNormal, so its `mu`,
    # `sigma` hyperparameters are read once under the `district` group and
    # threaded to every member's non-centred reconstruction.
    template = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district))))

    @model function district_model()
        mu ~ Normal(0.0, 1.0)
        sigma ~ truncated(Normal(0.0, 1.0); lower = 0.0)
        return (mu = mu, sigma = sigma)
    end
    @model function shape_z_model()
        z ~ Normal(0.0, 1.0)
        return (z = z,)
    end
    @model function north_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function south_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function tree_model()
        district ~ to_submodel(district_model())
        north ~ to_submodel(north_model())
        south ~ to_submodel(south_model())
        return (district = district, north = north, south = south)
    end
    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(98)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.district.mu" in vns
    @test "d.district.sigma" in vns

    nt = chain_to_params(template, chain)
    @test keys(nt.district) == (:mu, :sigma)

    fitted = update(template, chain)
    @test !has_uncertain(fitted)
    mu, sigma = nt.district.mu, nt.district.sigma
    zn = nt.north.shape.z
    @test params(event(fitted, :north))[1] ≈ exp(mu + sigma * zn)
end

@testitem "chain_to_params reads two distinct pooling groups in one tree" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain
    using Statistics: mean

    # `district` has an estimated (LogNormal) population, `region` a fixed
    # (Normal) one, so `_pool_params`'s walk over the tree's pooling groups
    # must keep each group's hyperparameters (or absence of them) separate.
    template = compose((
        north = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        south = uncertain(Gamma(2.0, 1.0); shape = pool(:district)),
        east = uncertain(Gamma(2.0, 1.0);
            shape = pool(:region, Normal(0.4, 0.2))),
        west = uncertain(Gamma(2.0, 1.0);
            shape = pool(:region, Normal(0.4, 0.2)))))

    @model function district_model()
        mu ~ Normal(0.0, 1.0)
        sigma ~ truncated(Normal(0.0, 1.0); lower = 0.0)
        return (mu = mu, sigma = sigma)
    end
    @model function shape_z_model()
        z ~ Normal(0.0, 1.0)
        return (z = z,)
    end
    @model function leaf_shape_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function tree_model()
        district ~ to_submodel(district_model())
        north ~ to_submodel(leaf_shape_model())
        south ~ to_submodel(leaf_shape_model())
        east ~ to_submodel(leaf_shape_model())
        west ~ to_submodel(leaf_shape_model())
        return (district = district, north = north, south = south,
            east = east, west = west)
    end
    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(101)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.district.mu" in vns
    @test "d.district.sigma" in vns
    @test "d.east.shape.z" in vns
    @test "d.west.shape.z" in vns
    @test !("d.region.mu" in vns)

    nt = chain_to_params(template, chain)
    @test keys(nt.district) == (:mu, :sigma)
    @test !haskey(nt, :region)

    fitted = update(template, chain)
    @test !has_uncertain(fitted)

    # Hand-compute each expected value straight from the raw chain draws,
    # independent of `chain_to_params`'s own reduction.
    vn(name) = only(filter(v -> string(v) == name,
        collect(FlexiChains.parameters(chain))))
    mu = mean(vec(chain[vn("d.district.mu")]))
    sigma = mean(vec(chain[vn("d.district.sigma")]))
    z_north = mean(vec(chain[vn("d.north.shape.z")]))
    z_south = mean(vec(chain[vn("d.south.shape.z")]))
    z_east = mean(vec(chain[vn("d.east.shape.z")]))
    z_west = mean(vec(chain[vn("d.west.shape.z")]))

    @test params(event(fitted, :north))[1] ≈ exp(mu + sigma * z_north)
    @test params(event(fitted, :south))[1] ≈ exp(mu + sigma * z_south)
    @test params(event(fitted, :east))[1] ≈ 0.4 + 0.2 * z_east
    @test params(event(fitted, :west))[1] ≈ 0.4 + 0.2 * z_west
end

@testitem "chain_to_params reads a pooled parameter inside a shared leaf" begin
    using ComposedDistributions, Distributions, DynamicPPL, Turing, Random
    using FlexiChains: FlexiChains, VNChain
    using Statistics: mean

    # A `shared`-tagged leaf is inventoried once under its tag (see
    # `_walk_rows!`, introspection.jl): a pooled parameter inside it lowers its
    # `z` latent row to `<tag>.<param>.z`, not `<branch>.<param>.z`, and
    # `update` reads the one tagged entry back into every occurrence.
    leaf = shared(:tied, uncertain(Gamma(2.0, 1.0); shape = pool(:district)))
    template = compose((north = leaf, south = leaf))

    @model function district_model()
        mu ~ Normal(0.0, 1.0)
        sigma ~ truncated(Normal(0.0, 1.0); lower = 0.0)
        return (mu = mu, sigma = sigma)
    end
    @model function shape_z_model()
        z ~ Normal(0.0, 1.0)
        return (z = z,)
    end
    @model function tied_model()
        shape ~ to_submodel(shape_z_model())
        return (shape = shape,)
    end
    @model function tree_model()
        district ~ to_submodel(district_model())
        tied ~ to_submodel(tied_model())
        return (district = district, tied = tied)
    end
    @model function full_model()
        d ~ to_submodel(tree_model())
        return d
    end

    Random.seed!(103)
    chain = sample(full_model(), Prior(), 30; chain_type = VNChain,
        progress = false)
    vns = Set(string.(collect(FlexiChains.parameters(chain))))
    @test "d.tied.shape.z" in vns
    @test !("d.north.shape.z" in vns)
    @test !("d.south.shape.z" in vns)

    nt = chain_to_params(template, chain)
    @test keys(nt.district) == (:mu, :sigma)
    @test keys(nt.tied.shape) == (:z,)

    fitted = update(template, chain)
    @test !has_uncertain(fitted)

    vn(name) = only(filter(v -> string(v) == name,
        collect(FlexiChains.parameters(chain))))
    mu = mean(vec(chain[vn("d.district.mu")]))
    sigma = mean(vec(chain[vn("d.district.sigma")]))
    z = mean(vec(chain[vn("d.tied.shape.z")]))

    # Tied, so both occurrences collapse to the SAME reconstructed value.
    @test event(fitted, :north) == event(fitted, :south)
    @test params(event(fitted, :north))[1] ≈ exp(mu + sigma * z)
end
