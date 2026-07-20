# Node-level uncertainty tests: a `Resolve`'s branch probabilities made
# uncertain with a simplex-valued (`Dirichlet`) prior, flattened through the
# stick-breaking codec, and the deliberate absence of a node-level free
# parameter on `Compete` / `Choose`. See issue #89.

@testitem "branch_probs: update attaches a Dirichlet spec" begin
    using ComposedDistributions: update
    using Distributions

    r = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    @test !has_uncertain(r)

    ur = update(r, (branch_probs = Dirichlet([1.0, 1.0]),))
    @test has_uncertain(ur)
    # The current probabilities stay as the point; the delays are untouched.
    @test ur.branch_probs == r.branch_probs
    @test event(ur, :death) == Gamma(1.5, 1.0)
    @test event(ur, :disch) == Gamma(2.0, 1.5)

    # A Dirichlet whose length does not match the outcome count is rejected.
    @test_throws "must have one weight per outcome" update(
        r, (branch_probs = Dirichlet([1.0, 1.0, 1.0]),))

    # Attaching also works on a named node inside a tree.
    tree = compose((resolution = r, tail = LogNormal(0.5, 0.4)))
    utree = update(tree, (resolution = (branch_probs = Dirichlet([1.0, 1.0]),),))
    @test has_uncertain(utree)
    @test has_uncertain(event(utree, :resolution))
    @test !has_uncertain(event(utree, :tail))
end

@testitem "branch_probs: params_table emits stick rows with Beta priors" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flat_dimension

    # probs = (0.2, 0.3, 0.5); K = 3 => two stick coordinates.
    r = resolve(:death => (Gamma(1.5, 1.0), 0.2),
        :disch => (Gamma(2.0, 1.5), 0.3),
        :recover => Gamma(1.0, 1.0))
    ur = update(r, (branch_probs = Dirichlet([1.0, 1.0, 1.0]),))

    tbl = params_table(ur)
    stick = findall(==(:branch_probs), tbl.edge)
    @test length(stick) == 2
    @test tbl.param[stick] == [:stick_1, :stick_2]

    # The stick priors are the exact Dirichlet factorisation
    # Beta(alpha_k, sum(alpha_{k+1:K})).
    @test tbl.prior[stick[1]] == Beta(1.0, 2.0)
    @test tbl.prior[stick[2]] == Beta(1.0, 1.0)

    # The stick VALUES are the inverse stick-breaking of the current probs.
    @test tbl.value[stick[1]] ≈ 0.2
    @test tbl.value[stick[2]] ≈ 0.3 / 0.8

    # Only the two stick coordinates are estimated (the delays are fixed).
    @test flat_dimension(ur) == 2

    # A fixed Resolve contributes no estimated stick rows.
    @test flat_dimension(r) == 0
    ptbl = params_table(r)
    @test !any(==(:stick_1), ptbl.param)
end

@testitem "branch_probs: codec round-trip and collapse to a simplex" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flatten, unflatten, flat_dimension

    r = resolve(:death => (Gamma(1.5, 1.0), 0.2),
        :disch => (Gamma(2.0, 1.5), 0.3),
        :recover => Gamma(1.0, 1.0))
    ur = update(r, (branch_probs = Dirichlet([1.0, 1.0, 1.0]),))

    # Round-trip over the estimated (stick) subset.
    x = [0.3, 0.6]
    nt = unflatten(ur, x)
    @test flatten(ur, nt) == x

    # Collapsing at the draw rebuilds a concrete Resolve whose probabilities are
    # the stick-breaking of x and sum to one.
    collapsed = update(ur, nt)
    @test !has_uncertain(collapsed)
    p = collect(values(probs(collapsed)))
    @test sum(p) ≈ 1.0
    @test p[1] ≈ 0.3                 # stick_1
    @test p[2] ≈ 0.6 * (1 - 0.3)     # stick_2 * remaining
    @test p[3] ≈ (1 - 0.3) * (1 - 0.6)
    # The collapsed Resolve's logpdf is the p-weighted mixture of the three
    # delays' own densities, not just some finite number.
    @test logpdf(collapsed, 1.5) ≈ log(
        p[1] * pdf(Gamma(1.5, 1.0), 1.5) +
        p[2] * pdf(Gamma(2.0, 1.5), 1.5) +
        p[3] * pdf(Gamma(1.0, 1.0), 1.5))
end

@testitem "branch_probs: stick Betas reproduce the Dirichlet" begin
    using ComposedDistributions: update
    using Distributions, Random, Statistics
    using ComposedDistributions: unflatten

    alpha = [2.0, 3.0, 1.5, 4.0]
    r = resolve(:a => (Gamma(1.0, 1.0), 0.25),
        :b => (Gamma(1.0, 1.0), 0.25),
        :c => (Gamma(1.0, 1.0), 0.25),
        :d => Gamma(1.0, 1.0))
    ur = update(r, (branch_probs = Dirichlet(alpha),))

    tbl = params_table(ur)
    stick = findall(==(:branch_probs), tbl.edge)
    betas = tbl.prior[stick]

    # Draw sticks from the Beta priors, collapse, and compare the mean simplex
    # to the Dirichlet mean: the factorisation is exact.
    rng = Xoshiro(1)
    K = 4
    acc = zeros(K)
    N = 40_000
    for _ in 1:N
        xs = [rand(rng, b) for b in betas]
        p = collect(values(probs(update(ur, unflatten(ur, xs)))))
        acc .+= p
    end
    @test acc ./ N ≈ alpha ./ sum(alpha) atol = 0.02
end

@testitem "branch_probs: promote attaches a flat Dirichlet" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flat_dimension

    tree = compose((resolution = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7)),))
    promoted = update(tree, param_priors(tree))

    @test has_uncertain(promoted)
    res = event(promoted, :resolution)
    @test has_uncertain(res)
    # The two delays' params (2 each) plus one stick coordinate (K - 1 = 1).
    @test flat_dimension(promoted) == 5

    # The attached branch-prob prior is the flat Dirichlet.
    tbl = params_table(promoted)
    stick = findall(i -> tbl.edge[i] == Symbol("resolution.branch_probs"),
        eachindex(tbl.edge))
    @test length(stick) == 1
    @test tbl.prior[stick[1]] == Beta(1.0, 1.0)
end

@testitem "branch_probs: ForwardDiff gradient matches finite differences" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: as_logdensity, logdensity, flat_dimension
    using ForwardDiff

    # A bare (univariate) Resolve whose branch-probability simplex is uncertain:
    # the gradient flows through the stick-breaking reconstruction into the
    # mixture marginal's AD-safe log-sum-exp.
    r = update(resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)),
        (branch_probs = Dirichlet([2.0, 2.0]),))
    prob = as_logdensity(r, [1.5, 0.8, 2.1, 3.2])

    @test flat_dimension(r) == 1
    x0 = [0.4]
    g = ForwardDiff.gradient(x -> logdensity(prob, x), x0)
    @test all(isfinite, g)

    h = 1e-6
    fd = (logdensity(prob, x0 .+ h) - logdensity(prob, x0 .- h)) / (2h)
    @test g[1] ≈ fd atol = 1e-4
end

@testitem "Compete has no node-level free parameter" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flat_dimension

    # A racing-hazard node's winning probability is derived from the hazards,
    # not a free parameter, so it has no branch-probability row and promote adds
    # only the causes' own delay parameters.
    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    tbl = params_table(c)
    @test !any(==(:branch_probs), tbl.edge)
    @test !any(==(:stick_1), tbl.param)

    promoted = update(c, param_priors(c))
    # Only the two Gamma delays' shape/scale: four parameters, no node dim.
    @test flat_dimension(promoted) == 4
end

@testitem "Choose selector is data, not a parameter" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flat_dimension

    # The active alternative is chosen by the data selector, so a Choose has no
    # node-level free weight parameter; only the alternatives' own params are
    # estimated.
    ch = choose(:index => Gamma(2.0, 1.0), :sourced => Gamma(3.0, 1.5))
    tbl = params_table(ch)
    @test !any(==(:branch_probs), tbl.edge)
    @test !any(==(:weights), tbl.param)

    promoted = update(ch, param_priors(ch))
    @test flat_dimension(promoted) == 4
end

@testitem "branch_probs: promote across a mixed tree with a composite leaf" begin
    using ComposedDistributions: update
    using Distributions
    using ComposedDistributions: flat_dimension

    # A mixed tree: a Resolve (uncertain branch_probs via promote), a Convolved
    # composite leaf (see-through component fitting, #81), and a plain leaf.
    # Promote must make all three estimable together, descending past the
    # composite without disturbing the branch-probability injection.
    total = convolved(Gamma(2.0, 1.0), Gamma(1.0, 1.5))
    tree = compose((
        resolution = resolve(:death => (Gamma(1.5, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)),
        total = total,
        report = LogNormal(0.5, 0.4)))
    promoted = update(tree, param_priors(tree))

    @test has_uncertain(promoted)
    @test has_uncertain(event(promoted, :resolution))
    tbl = params_table(promoted)
    # The Resolve contributes exactly one stick coordinate (K = 2).
    @test count(==(Symbol("resolution.branch_probs")), tbl.edge) == 1
    # The composite's components are still inventoried and now estimated.
    @test Symbol("total.component_1") in tbl.edge
    # Promote estimates every row (leaves, composite components, and the stick),
    # so the flat dimension equals the promoted table's row count.
    @test flat_dimension(promoted) == length(tbl.edge)
end

@testitem "branch_probs: update rejects ill-typed branch_probs values" begin
    using ComposedDistributions: update
    using Distributions

    r = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.7))
    # Merge mode (partial): a non-Dirichlet distribution at the branch_probs
    # slot errors.
    @test_throws "merge mode must be a `Dirichlet`" update(
        r, (branch_probs = Beta(1.0, 1.0),))
    # A strict update covers every leaf; a non-NamedTuple branch_probs errors.
    full = (death = (shape = 1.5, scale = 1.0),
        disch = (shape = 2.0, scale = 1.5))
    @test_throws "strict `branch_probs` update must be a NamedTuple" update(
        r, merge(full, (branch_probs = 0.5,)))
    # A fixed node accepts a concrete per-outcome replacement (strict).
    r2 = update(r, merge(full, (branch_probs = (death = 0.4, disch = 0.6),)))
    @test !has_uncertain(r2)
    @test collect(values(probs(r2))) ≈ [0.4, 0.6]
    # Direct construction with a non-Dirichlet prior is rejected.
    @test_throws "branch-probability prior must be a `Dirichlet`" Resolve(
        (:a, :b), (Gamma(1.0, 1.0), Gamma(1.0, 1.0)), (0.3, 0.7),
        Normal(0.0, 1.0))
end
