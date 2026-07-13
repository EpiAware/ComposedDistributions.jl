# ComposedDistributions × LoweredDistributions extension: the composer methods
# of `lower` build the phase-type / CTMC of a composed delay structure (#149).
# The scalar composers lower to a phase-type whose absorption-time moments are
# checked against the composition (series adds means and variances, a mixture
# is the probability-weighted mixture, competing risks is the minimum). The
# vector composers lower to a joint CTMC whose generator is validated by the
# constructor and whose transition matrix is stochastic.

@testitem "Lowered extension: scalar composers lower to phase-types" begin
    using Distributions
    using LoweredDistributions: lower, PhaseType
    using LinearAlgebra

    # The extension loads once LoweredDistributions is present.
    @test Base.get_extension(ComposedDistributions,
        :ComposedDistributionsLoweredDistributionsExt) !== nothing

    # Phase-type absorption-time moments: m = (-S)^{-1} 1 is the expected time
    # in each phase, so the mean is α ⋅ m and the second moment is 2 α (-S)^{-2} 1.
    function pt_moments(p::PhaseType)
        A = -Matrix(p.S)
        m1 = A \ ones(length(p.α))
        m2 = A \ m1
        mean = dot(p.α, m1)
        var = 2 * dot(p.α, m2) - mean^2
        return mean, var
    end

    # Sequential convolves the steps: means and variances add.
    seq = sequential(Exponential(2.0), Exponential(3.0))
    p = lower(seq)
    @test p isa PhaseType
    @test length(p.α) == 2
    @test isapprox(sum(p.α), 1; atol = 1e-8)
    mean, var = pt_moments(p)
    @test isapprox(mean, 2.0 + 3.0; atol = 1e-8)
    @test isapprox(var, 4.0 + 9.0; atol = 1e-8)

    # A multi-phase step (Erlang) convolved with an exponential.
    seq2 = sequential(Gamma(3.0, 1.5), Exponential(2.0))
    p2 = lower(seq2)
    @test length(p2.α) == 4
    m2, _ = pt_moments(p2)
    @test isapprox(m2, mean(Gamma(3.0, 1.5)) + 2.0; atol = 1e-6)

    # Resolve mixes the outcomes weighted by the branch probabilities.
    res = resolve(:a => (Exponential(2.0), 0.3), :b => (Exponential(5.0), 0.7))
    pr = lower(res)
    @test pr isa PhaseType
    @test isapprox(sum(pr.α), 1; atol = 1e-8)
    mr, _ = pt_moments(pr)
    @test isapprox(mr, 0.3 * 2.0 + 0.7 * 5.0; atol = 1e-8)

    # Compete races the causes: the minimum of two exponentials is exponential
    # with the summed rate, so the mean is 1 / (1/2 + 1/3) = 1.2.
    cmp = compete(:a => Exponential(2.0), :b => Exponential(3.0))
    pc = lower(cmp)
    @test pc isa PhaseType
    mc, vc = pt_moments(pc)
    @test isapprox(mc, 1 / (1 / 2 + 1 / 3); atol = 1e-8)
    @test isapprox(vc, (1 / (1 / 2 + 1 / 3))^2; atol = 1e-8)
end

@testitem "Lowered extension: Shared is a transparent tie" begin
    using Distributions
    using LoweredDistributions: lower, PhaseType, AbstractLowering

    # A shared-parameter leaf lowers exactly as its wrapped distribution.
    d = Gamma(3.0, 1.0)
    tied = shared(:incubation, d)
    @test lower(tied) isa AbstractLowering
    @test typeof(lower(tied)) == typeof(lower(d))

    # A tie inside a chain leaves the series phase-type unchanged.
    plain = lower(sequential(Exponential(2.0), d))
    withtie = lower(sequential(Exponential(2.0), tied))
    @test plain.α ≈ withtie.α
    @test plain.S ≈ withtie.S
end

@testitem "Lowered extension: vector composers lower to joint CTMCs" begin
    using Distributions
    using LoweredDistributions: lower, CTMC, transition_probability

    # Parallel runs branches jointly: the product state space of two
    # two-state branches (one phase plus one absorbing state each) has four
    # states, and the transition matrix is stochastic.
    par = parallel(Exponential(2.0), Exponential(3.0))
    mp = lower(par)
    @test mp isa CTMC
    @test length(mp.states) == 4
    P = transition_probability(mp, 1.5)
    @test all(isapprox.(sum(P; dims = 2), 1; atol = 1e-8))

    # Choose unions the alternatives block-diagonally: two two-state
    # alternatives give a four-state chain, again a valid generator.
    ch = choose(:fast => Exponential(1.0), :slow => Exponential(4.0))
    mc = lower(ch)
    @test mc isa CTMC
    @test length(mc.states) == 4
    Pc = transition_probability(mc, 2.0)
    @test all(isapprox.(sum(Pc; dims = 2), 1; atol = 1e-8))
end

@testitem "Lowered extension: unsupported structures error clearly" begin
    using Distributions
    using LoweredDistributions: lower

    # A no-event Resolve is defective: its initial distribution cannot sum to
    # one, so it has no phase-type lowering.
    res = resolve(:event => (Exponential(2.0), 0.4), :none => (NoEvent(), 0.6))
    @test_throws ArgumentError lower(res)

    # A vector composer nested inside a scalar composition lowers to a
    # multi-absorbing CTMC that cannot fold into a phase-type.
    nested = sequential(Exponential(2.0),
        parallel(Exponential(1.0), Exponential(1.0)))
    @test_throws ArgumentError lower(nested)
end
