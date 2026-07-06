"""
    ADFixtures

Shared AD gradient scenarios and backend metadata for ComposedDistributions.
Used by `test/ad/runtests.jl`. Covers the composed `logpdf` of a `Sequential`
chain, a `Resolve` mixture marginal (differentiating through a covariate branch
probability), and a `Compete` racing-hazard marginal (differentiating through
the survival product), across the ForwardDiff / ReverseDiff / Enzyme / Mooncake
backend matrix.

The composed value `logpdf` is a sum over flat leaf slices, so the gradient
flows through each leaf's own `logpdf`; the reference is computed with
`ForwardDiff` and matched by the reverse backends to ~1e-6.
"""
module ADFixtures

# `__precompile__(false)` skips the precompile cache so the Mooncake / Enzyme
# load chain does not break the package build on CI.
__precompile__(false)

using ComposedDistributions
using Distributions: Distributions, Gamma, LogNormal, Normal, mean, var, logpdf
using ADTypes: ADTypes, AutoForwardDiff, AutoReverseDiff, AutoMooncake,
               AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Mooncake, Enzyme

export scenarios, backends, broken_scenario_names,
       backend_broken_scenarios, backend_skip_scenarios

function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...)
end

"""
    backends()

AD backends tested, as `(; name, backend)` named tuples. The `name` is what
`test/ad/scenarios.jl` selects by tag.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)",
            backend = AutoReverseDiff(compile = false)),
        (name = "Mooncake reverse",
            backend = AutoMooncake(config = nothing)),
        (name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse)))
    ]
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
backend_broken_scenarios() = Dict{String, Set{String}}()

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

"""
    scenarios(; with_reference::Bool = false, category::Symbol = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`. All
scenarios sit in one group, so `category` is accepted for the harness contract
but unused.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    obs = [0.5, 1.2, 2.5, 3.8, 5.1]

    out = DIT.Scenario{:gradient, :out}[]

    function _push!(name, f, θ₀, contexts)
        res1 = with_reference ? _reference(f, θ₀, contexts) : nothing
        prep_args = (; x = θ₀, contexts = contexts)
        push!(out,
            res1 === nothing ?
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...; prep_args = prep_args, name = name) :
            DIT.Scenario{:gradient, :out}(
                f, θ₀, contexts...;
                res1 = res1, prep_args = prep_args, name = name))
    end

    # Sequential chain: the composed value `logpdf` is a sum over the flat leaf
    # slices, so the gradient flows through each step's own `logpdf`. Score the
    # two-value vector `[obs_i, 1.0]` (the second step held constant) so the
    # differentiated parameters land on the first (Gamma) step.
    _push!("Sequential Gamma+LogNormal logpdf",
        (θ,
            obs) -> sum(
            x -> logpdf(
                sequential(:a => Gamma(θ[1], θ[2]),
                    :b => LogNormal(0.5, 0.4)), [x, 1.0]), obs),
        [2.0, 1.0], (Constant(obs),))

    # Resolve mixture marginal: the branch probability is a covariate quantity
    # (`θ[3]`), so the gradient must flow through the AD-safe `_one_of_logmix`
    # reduction (not `float.(branch_probs)`). The two delays' shapes and the
    # branch probability are all differentiated.
    _push!("Resolve mixture marginal logpdf",
        (θ,
            obs) -> sum(
            x -> logpdf(
                resolve(:death => (Gamma(θ[1], 1.0), θ[3]),
                    :disch => (Gamma(θ[2], 1.5), 1 - θ[3])), x), obs),
        [1.5, 2.0, 0.3], (Constant(obs),))

    # Compete racing-hazard marginal: the survival product `∏ S_k` goes through
    # the AD-safe `_logccdf_ad_safe`, so a Gamma survival differentiates w.r.t.
    # its shape/scale.
    _push!("Compete racing-hazard marginal logpdf",
        (θ,
            obs) -> sum(
            x -> logpdf(
                compete(:death => Gamma(θ[1], θ[2]),
                    :recover => Gamma(3.0, 2.0)), x), obs),
        [2.0, 3.0], (Constant(obs),))

    return out
end

end # module ADFixtures
