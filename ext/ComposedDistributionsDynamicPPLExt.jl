module ComposedDistributionsDynamicPPLExt

# ComposedDistributions x DynamicPPL: `as_turing(dist, data)` builds a
# DynamicPPL model over a composed distribution's ESTIMATED parameters, a light
# wrapper on the `as_logdensity` codec (declared with its docstring in
# `src/composers/turing.jl`). Loaded when DynamicPPL alone is available, so the
# core stays Turing-free. The model reuses the codec end to end: each estimated
# parameter is a named `~` site sampled from its own prior (`prob.flat_priors`),
# and the data likelihood is added with `@addlogprob!` from the codec's
# reconstruction, so the model's total log-density equals
# `logdensity(prob, x)` by construction. The `~` site names match the
# FlexiChains readback's dotted `<prefix>.<edge...>.<param>` names exactly, so a
# fitted chain reads back through `chain_to_params` / `update(dist, chain)`
# unchanged.

using ComposedDistributions: ComposedDistributions,
                             AbstractComposedDistribution, ComposedLogDensity,
                             as_logdensity, unflatten, update, params_table
import ComposedDistributions: as_turing
using DynamicPPL: DynamicPPL, @model, NamedDist, VarName

# `AbstractPPL` (re-exported through DynamicPPL) owns the `VarName` optic types.
# There is no public constructor for a runtime dotted optic, so the two optic
# primitives are reached through the parent module: `Property{sym}(child)` for a
# `.sym` access and `Iden()` for the leaf. These are the same primitives
# DynamicPPL's own `@varname` lowers to.
const _AbstractPPL = parentmodule(VarName)
const _Property = _AbstractPPL.Property
const _Iden = _AbstractPPL.Iden

# Build the `VarName` a submodel-sampled parameter carries: the `prefix` symbol
# then the dotted edge path and parameter name, so `string(vn)` is
# `"<prefix>.<segs...>"` (e.g. prefix `:d`, segs `(:onset_admit, :shape)` ->
# `"d.onset_admit.shape"`). This matches `_dotted`/`string(vn)` in the
# FlexiChains readback exactly, so the chain reads back unchanged. The optic is
# built outermost-property-first (`reverse`) so the earliest segment renders
# nearest the prefix.
function _dotted_varname(prefix::Symbol, segs::Tuple)
    optic = foldl((acc, s) -> _Property{s}(acc), reverse(segs); init = _Iden())
    return VarName{prefix}(optic)
end

# Reject a centred-pool parameter (a general, non-location-scale pooled
# population): its population prior is hyperparameter-dependent, so it carries a
# `CentredPoolPrior` marker rather than a fixed `~` prior, and cannot be a named
# prior site. Point to the codec + LogDensityProblemsAD path, which scores the
# centred population term directly.
function _reject_centred_pools(prob::ComposedLogDensity)
    any(p -> p isa ComposedDistributions.CentredPoolPrior, prob.flat_priors) &&
        throw(ArgumentError(
            "as_turing does not support centred-pool parameters (a general, " *
            "non-location-scale pooled population); the population prior is " *
            "hyperparameter-dependent and has no fixed `~` prior. Sample the " *
            "tree with `as_logdensity(dist, data)` + LogDensityProblemsAD " *
            "(the LogDensityProblems extension), or use a location-scale " *
            "(Normal/LogNormal) pooled population for the non-centred form."))
    return nothing
end

# The model: sample each estimated parameter from its prior at its dotted
# VarName (a `NamedDist`, so the site name is the readback name regardless of
# the LHS), then add the data likelihood with `@addlogprob!` from the codec's
# reconstruction. Priors via `~`, likelihood via `@addlogprob!`, so no double
# counting; the total equals `logdensity(prob, θ)`. `θ` has an abstract element
# type so a sampled/AD value (a `Dual`/tracked number) flows through `unflatten`
# / `update` unchanged.
@model function _composed_turing_model(prob::ComposedLogDensity, vns)
    fp = prob.flat_priors
    n = length(fp)
    θ = Vector{Real}(undef, n)
    for i in 1:n
        # A plain scalar LHS: the `NamedDist` supplies the site's VarName (the
        # dotted readback name), while the sampled value is bound to `param` and
        # stored into `θ` by index. An indexed LHS (`θ[i] ~ ...`) instead makes
        # DynamicPPL apply the site's optic to `θ`, which is not what we want.
        param ~ NamedDist(fp[i], vns[i])
        θ[i] = param
    end
    d = update(prob.dist, unflatten(prob.dist, θ))
    DynamicPPL.@addlogprob! prob.loglik(d, prob.data)
    return d
end

function as_turing(dist::AbstractComposedDistribution, data;
        prefix::Symbol = :d, loglik = ComposedDistributions._default_loglik)
    prob = as_logdensity(dist, data; loglik = loglik)
    _reject_centred_pools(prob)
    # The estimated rows' `(path, param)` keys, in the same table order as
    # `prob.flat_priors`, so `vns[i]` names the site scored by `fp[i]`.
    layout = ComposedDistributions._flat_layout(params_table(dist))
    vns = [_dotted_varname(prefix, (path..., param))
           for (path, param) in layout]
    return _composed_turing_model(prob, vns)
end

end # module ComposedDistributionsDynamicPPLExt
