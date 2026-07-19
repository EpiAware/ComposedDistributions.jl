"""
    ADFixtures

Shared AD gradient scenarios and backend metadata for ComposedDistributions.
Used by `test/ad/runtests.jl`. Two categories:

`:marginal` covers the composed `logpdf` of a `Sequential` chain, a `Resolve`
mixture marginal (differentiating through a covariate branch probability), a
`Compete` racing-hazard marginal (differentiating through the survival
product), a `Resolve` whose branch-probability simplex is uncertain
(differentiating through the stick-breaking reconstruction), a partially pooled
parameter (differentiating through the non-centred `exp(mu + tau*z)`
reconstruction), a `Choose` scored at a selected alternative (differentiating
through the picked branch's own `logpdf`), and a `Censored` leaf (`#215`,
differentiating through Distributions.jl's censored `logpdf`/`logcdf`).

`:latent` covers the full `as_logdensity`/`logdensity` codec path: an
uncertain-leaf tree (differentiating the flat-vector -> nested-NamedTuple
codec, `unflatten`/`update`, into the data likelihood), a centred pool
(differentiating the `_pool_centred_logprior` term against the population), a
`Shared`-tagged uncertain leaf occurring twice (differentiating through the
tag-dedup: one flat parameter, its gradient accumulated from both
occurrences' likelihoods), and a `Truncated`-wrapped uncertain leaf (`#215`,
exercising the `#216` leaf-wrapper registry's codec path — `free_leaf`/
`rewrap_leaf` — under the generated `unflatten`/`update` walk).

All scenarios run across the ForwardDiff / ReverseDiff / Enzyme / Mooncake
backend matrix. The reference is computed with `ForwardDiff` and matched by the
reverse backends to ~1e-6.
"""
module ADFixtures

# `__precompile__(false)` skips the precompile cache so the Mooncake / Enzyme
# load chain does not break the package build on CI.
__precompile__(false)

using ComposedDistributions
using Distributions: Distributions, Gamma, LogNormal, Normal, mean, var, logpdf,
                     truncated, censored
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

# The partial-pooling reconstruction differentiates `logpdf(Gamma(shape), x)`
# w.r.t. a shape that is `exp(mu + tau*z)` with `mu`, `tau` SHARED across the
# strata. This used to be marked broken on Mooncake: at this fixture's eval
# point the second stratum's shape lands exactly on `1.0`, which routes a
# nonzero cotangent into `LogExpFunctions.xlogy`'s `iszero(x)` branch inside
# `Distributions.gammalogpdf`, and Mooncake had no rule for the two-argument
# `xlogy`/`xlog1py` (it derives one from the primal branch, giving `0` instead
# of `log(y)` at `x == 0`; see #99 and upstream
# https://github.com/chalk-lab/Mooncake.jl/issues/1241).
# `ComposedDistributionsMooncakeExt` now imports the ChainRulesCore rules for
# `xlogy`/`xlog1py` (already shipped by `LogExpFunctionsChainRulesCoreExt`) as
# Mooncake primitives, so this scenario is no longer broken on Mooncake.
#
# The `:latent` "Uncertain-leaf logdensity codec" scenario differentiates the
# full `as_logdensity`/`logdensity` path, whose `unflatten` rebuilds the nested
# `NamedTuple` `update` consumes. It used to be marked broken on Enzyme
# reverse: the old Dict-based walk built a type-unstable, heap-boxed
# `NamedTuple`, and Enzyme's cache-store type reasoning hit `Taking the type
# of an opaque pointer is illegal` on that reconstruction (an Enzyme/LLVM
# internal limitation with type-unstable heap-building, the same family as
# the map-vs-generator `IllegalTypeAnalysisException`, finding C8). #178 PR 2
# replaced that walk with a `@generated` compile-time layout walk
# (`codec_gen.jl`) that emits a concretely-typed (`@inferred`-stable)
# `NamedTuple` construction with the slot indices baked in as literals, so
# Enzyme reverse differentiated it like every other backend for a while
# (#162).
#
# #190 (`#178` PR 2's actual landing) then reintroduced a different Enzyme
# reverse failure on this same scenario: an `EnzymeInternalError` (compiler-
# internal, not a Julia-level type/activity error) inside Enzyme's LLVM
# `nodecayed_phis!` optimisation pass, failing within `_update(::Parallel,
# ::NamedTuple, ::NamedTuple, ::Bool)` (introspection.jl) — the `ntuple(...)
# do i ... end` closure that rebuilds a `Parallel`'s children. Every `ad`
# CI run since #190 landed reproduces it deterministically (same 149-pass/
# 1-error split, same failing method) on GitHub Actions' runners; it does
# NOT reproduce locally on a dev machine with the exact same resolved
# package versions (Enzyme 0.13.188, Enzyme_jll 0.0.285+0, GPUCompiler
# 1.23.0, LLVM.jl 9.10.1) — see the #223 comment for the full
# characterisation and a draft upstream report. Ledgered here (rather than
# left to error CI red) so a real regression stays visible without failing
# the suite; remove this entry once the upstream fix lands or the local
# machine can reproduce a fix to verify.
#
"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
function backend_broken_scenarios()
    return Dict{String, Set{String}}(
        "Enzyme reverse" => Set(["Uncertain-leaf logdensity codec"])
    )
end

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

"""
    scenarios(; with_reference::Bool = false, category::Symbol = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`. `category`
selects the group: `:marginal` (default) returns the composed-`logpdf`
scenarios; `:latent` returns the `logdensity` codec scenarios.
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

    # --- latent category: the full as_logdensity/logdensity codec path -------
    if category == :latent
        # Uncertain-leaf codec: differentiate `logdensity(prob, θ)` for a tree
        # with an ordinary uncertain leaf, so the gradient flows through the
        # flat-vector -> nested-NamedTuple codec (`unflatten`/`update`) into the
        # data likelihood. This is the systematic (all-backend) companion to the
        # bespoke Mooncake-only #146 item in `scenarios.jl`.
        codec_tree = compose((
            onset_admit = uncertain(Gamma(2.0, 1.0);
                shape = LogNormal(log(2.0), 0.2)),
            admit_death = LogNormal(0.5, 0.4)))
        codec_prob = ComposedDistributions.as_logdensity(
            codec_tree, [[0.5, 2.0], [1.0, 3.0]])
        _push!("Uncertain-leaf logdensity codec",
            (θ, prob) -> ComposedDistributions.logdensity(prob, θ),
            [2.0], (Constant(codec_prob),))

        # Centred pool: two members pool a `shape` centred against a fixed
        # `Gamma` population, so the gradient flows through
        # `_pool_centred_logprior` (the population-scored latent term) as well as
        # each member's own Gamma likelihood. The centred reconstruction is the
        # identity (the latent IS the parameter), so this exercises the centred
        # scoring path distinct from the non-centred reconstruction.
        pool_tree = compose((
            north = uncertain(Gamma(2.0, 1.0);
                shape = pool(:district, Gamma(2.0, 1.0); noncentred = false)),
            south = uncertain(Gamma(2.0, 1.0);
                shape = pool(:district, Gamma(2.0, 1.0); noncentred = false))))
        pool_prob = ComposedDistributions.as_logdensity(
            pool_tree, [[0.5, 2.0], [1.0, 3.0]])
        _push!("Pool centred logdensity",
            (θ, prob) -> ComposedDistributions.logdensity(prob, θ),
            [2.0, 3.0], (Constant(pool_prob),))

        # Shared-tag codec: the same uncertain template occurs twice under one
        # `shared(:g, ...)` tag, so `params_table`/`unflatten` dedup it to ONE
        # flat parameter and `update` places the drawn value in both
        # occurrences (mirrors `test/composers/logdensity.jl`'s "codec: shared
        # spec round-trip"). The reverse-mode gradient of that one parameter
        # must accumulate from both occurrences' data likelihoods, the
        # AD-critical path for tag dedup that #96/#146 exercised for
        # construction/unflatten but no scenario here has yet driven through
        # the full logdensity codec.
        shared_u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
        shared_tree = compose((a = shared(:g, shared_u), b = shared(:g, shared_u)))
        shared_prob = ComposedDistributions.as_logdensity(
            shared_tree, [[0.5, 2.0], [1.0, 3.0]])
        _push!("Shared-tag logdensity codec",
            (θ, prob) -> ComposedDistributions.logdensity(prob, θ),
            [2.0], (Constant(shared_prob),))

        # Truncated-wrapped uncertain leaf: `truncated(uncertain(...); upper)`
        # pushes the wrap inside the `Uncertain` template (`wrapped_leaves.jl`,
        # #215), so the generated codec's leaf-wrapper registry (#216:
        # `free_leaf`/`rewrap_leaf` dispatch on `Distributions.Truncated`) is
        # what `unflatten`/`update` walk through to reach the wrapped leaf's
        # `mu`. No AD scenario drove a wrapped leaf through
        # `as_logdensity`/`logdensity` before this; #215/#216 landed with
        # value-level tests only. Truncates a `LogNormal` (not `Gamma`):
        # `Distributions.truncated`'s normalising constant calls the wrapped
        # leaf's `logcdf`, and `Gamma`'s routes through `StatsFuns`'
        # `_gammalogcdf`, which has concrete `Float64`/`Float32`/`Float16`
        # methods only (no generic fallback) and so errors under ReverseDiff's
        # tracked reals — an upstream Distributions.jl/StatsFuns gap, not a
        # ComposedDistributions one (flagged separately on #223). `LogNormal`'s
        # `logcdf` goes through `normlogcdf(μ::Real, σ::Real, x::Number)`,
        # which is genuinely generic, so it isolates the registry/codec path
        # this scenario targets from that unrelated gap.
        trunc_tree = compose((
            onset = truncated(
                uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2));
                upper = 8.0),
            admit = Gamma(2.0, 1.0)))
        trunc_prob = ComposedDistributions.as_logdensity(
            trunc_tree, [[0.5, 2.0], [1.0, 3.0]])
        _push!("Truncated uncertain-leaf logdensity codec",
            (θ, prob) -> ComposedDistributions.logdensity(prob, θ),
            [0.6], (Constant(trunc_prob),))

        return out
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

    # Resolve with an uncertain branch-probability simplex: the estimated stick
    # coordinate `θ[3]` reconstructs the branch probabilities through the
    # stick-breaking map, so the gradient of the mixture marginal flows through
    # the reconstruction (the AD-critical path for node-level uncertainty, #89)
    # as well as the two delays' shapes.
    _push!("Resolve stick-breaking branch-prob logpdf",
        (θ,
            obs) -> begin
            p = ComposedDistributions._stick_to_simplex((θ[3],))
            sum(
                x -> logpdf(
                    resolve(:death => (Gamma(θ[1], 1.0), p[1]),
                        :disch => (Gamma(θ[2], 1.5), p[2])), x), obs)
        end,
        [1.5, 2.0, 0.4], (Constant(obs),))

    # Partial pooling: two strata whose shapes are reconstructed non-centred
    # from the shared location-scale population `(mu, sigma)` = `(θ[1], θ[2])`
    # and their own latents `θ[3]`, `θ[4]` through `exp(mu + sigma*z)` (a
    # `LogNormal` population — exactly the non-centred map the codec applies),
    # the AD-critical path for pooling (#78). The gradient flows through the
    # reconstruction — with `mu`, `sigma` shared across both strata, so the
    # reverse pass must accumulate each hyperparameter from both — into each
    # stratum's Gamma `logpdf`.
    _push!("Pool non-centred reconstruction logpdf",
        (θ,
            obs) -> begin
            s1 = exp(θ[1] + θ[2] * θ[3])
            s2 = exp(θ[1] + θ[2] * θ[4])
            sum(
                x -> logpdf(Gamma(s1, 1.0), x) + logpdf(Gamma(s2, 1.0), x),
                obs)
        end,
        [0.2, 0.5, 0.3, -0.4], (Constant(obs),))

    # Choose selected-branch marginal: a `Choose` scored at the named `:index`
    # alternative routes through the type-stable `_pick`/`_select_logpdf` to that
    # branch's own `logpdf`, so the gradient flows through the selected Gamma's
    # shape/scale. The `kind` selector is discrete (held constant), so only the
    # scored branch is a gradient path.
    _push!("Choose selected-branch logpdf",
        (θ,
            obs) -> sum(
            x -> logpdf(
                choose(:index => Gamma(θ[1], θ[2]),
                    :sourced => Gamma(4.0, 1.5)), x; kind = :index), obs),
        [2.0, 1.0], (Constant(obs),))

    # Censored leaf marginal (#215): a `Distributions.censored` wrap on a
    # composed child routes `logpdf` through Distributions.jl's own censored
    # `logpdf`/`logcdf` (boundary-mass) branches, so the gradient must flow
    # through both the interior density and the boundary `logcdf`/`logccdf`
    # terms. No AD scenario exercised `Censored` before this; #215 landed with
    # value-level leaf-protocol-parity tests only.
    _push!("Censored leaf marginal logpdf",
        (θ,
            obs) -> sum(
            x -> logpdf(
                sequential(:a => censored(Gamma(θ[1], θ[2]), 0.0, 8.0),
                    :b => LogNormal(0.5, 0.4)), [x, 1.0]), obs),
        [2.0, 1.0], (Constant(obs),))

    return out
end

end # module ADFixtures
