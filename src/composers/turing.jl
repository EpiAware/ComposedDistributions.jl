# A DynamicPPL model over a composed distribution's ESTIMATED parameters, built
# as a light wrapper on the `as_logdensity` codec. Declared here with no method
# (a Turing-free stub); the model lives in
# `ext/ComposedDistributionsDynamicPPLExt.jl`, triggered by `DynamicPPL` alone,
# so this package stays Turing-free until that extension loads.

@doc "

A DynamicPPL model over a composed distribution's estimated parameters.

`as_turing(dist, data)` returns a `DynamicPPL`/`Turing` model whose free
parameters are the ESTIMATED parameters of `dist` (the [`uncertain`](@ref)
specs, the same flat parameters [`as_logdensity`](@ref) exposes), so a composed
posterior is sampleable with `sample(as_turing(dist, data), NUTS(), ...)`. It is
a light wrapper on the [`as_logdensity`](@ref) codec: each estimated parameter
is a named `~` site sampled from its own prior (so Turing constrains it and the
site carries the constrained value), and the data likelihood is added with
`DynamicPPL.@addlogprob!` from the codec's reconstruction
([`update`](@ref)`(dist, `[`unflatten`](@ref)`(dist, θ))` scored by `loglik`).
The model's total log-density equals
[`logdensity`](@ref)`(as_logdensity(dist, data), x)` at the corresponding
constrained `x` by construction.

The `~` sites are named to match the inference readback exactly: an estimated
leaf parameter is `<prefix>.<edge...>.<param>` (e.g. `d.onset_admit.shape`), an
uncertain node's branch probabilities are their stick coordinates
`<prefix>.<edge>.branch_probs.stick_k`, and a shared-tagged leaf is sampled once
under its tag. So a chain from `sample(as_turing(dist, data), ...)` reads back
through [`chain_to_params`](@ref) / [`update`](@ref)`(dist, chain)` unchanged.

Supports estimated rows with a concrete prior: ordinary uncertain leaves and
stick-breaking node branch probabilities. A pooled tree (see [`pool`](@ref)) is
rejected with an `ArgumentError`: a centred pool has no fixed `~` prior (its
population is hyperparameter-dependent), and the inference readback does not yet
consume a pooled chain, so a fitted pooled tree would not round-trip through
[`update`](@ref)`(dist, chain)`. Sample a pooled tree with
[`as_logdensity`](@ref) + `LogDensityProblemsAD` (the `LogDensityProblems`
extension) instead.

This method is available only when `DynamicPPL` is loaded (the model lives in a
package extension).

# Arguments
- `dist`: the composed distribution, carrying its uncertain specs.
- `data`: the observed records scored by `loglik`.

# Keyword Arguments
- `prefix`: the outer submodel variable name the sites are namespaced under
  (default `:d`), matching the readback prefix.
- `loglik`: a reducer `(d, data) -> Real` scoring `data` against the
  reconstructed distribution (default: sum of `logpdf(d, record)`), the same
  default [`as_logdensity`](@ref) uses.

# Examples
```@example
using ComposedDistributions, Distributions, Turing

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
data = [[0.5, 2.0], [1.0, 3.0]]

chain = sample(as_turing(tree, data), NUTS(), 200)
fit = update(tree, chain)
```

# See also
- [`as_logdensity`](@ref): the PPL-neutral log-density this wraps.
- [`chain_to_params`](@ref) / [`update`](@ref): read the fitted chain back.
- [`flatten`](@ref) / [`unflatten`](@ref): the flat <-> nested codec.
"
function as_turing end
