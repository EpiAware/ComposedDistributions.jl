"""
    ComposedDistributionsMooncakeExt

Shields `_ctor_has_check_args` with a Mooncake `@zero_adjoint`. Its
`hasmethod` reflection lowers to a `jl_gf_invoke_lookup` foreigncall that
Mooncake reverse has no rule for; the result is a `Bool` constant with
respect to the differentiated parameters, so a zero-adjoint primitive is
sound and keeps any reconstruction built on top of it (e.g. a future
DynamicPPL leaf rebuild, issue #9) AD-safe under Mooncake reverse.

Also shields `_split_edge` (the flat-vector codec's `Symbol` path-splitter,
`unflatten`/`flatten`'s unconditional hot-path call) and the codec's
`DimensionMismatch`-throwing helpers (`_throw_unflatten_dimmismatch`,
`_throw_logdensity_dimmismatch`, `_throw_as_named_dimmismatch`,
`_throw_logpdf_dimmismatch`) with `@zero_derivative` (both forward and
reverse), fixing issue #146. `_split_edge` calls `Base.split`, which lowers
to `findnext` over the `Symbol`'s string form; the `DimensionMismatch`
helpers interpolate their arguments into an error message via `show`. Both
recurse into Base's UTF-8 string-indexing continuation machinery, for which
Mooncake's whole-program rule derivation has no rule (a `sub_ptr`
pointer-arithmetic intrinsic): `_split_edge` unconditionally, on every
`unflatten`/`flatten` call regardless of AD, and the message helpers only
when their guard branch throws, but Mooncake still derives a rule for every
reachable branch whether it is taken or not. `_split_edge`'s result never
carries a tangent (`Symbol` inputs and outputs), and the message helpers
always throw (a constant `Union{}` result), so a zero-derivative primitive
is sound for each.

Also imports Mooncake primitives for `LogExpFunctions.xlogy`/`xlog1py` on
`Base.IEEEFloat` arguments, fixing issue #99. Mooncake has no rule for
either function, so it derives one from the primal implementation

    xlogy(x, y) = iszero(x) && !isnan(y) ? zero(x * log(y)) : x * log(y)

whose `iszero(x)` branch returns a constant, giving `∂/∂x = 0` at `x == 0`
instead of the correct `log(y)`. This surfaces through
`Distributions.gammalogpdf`, which computes `xlogy(shape - 1, x / scale)`,
so any Gamma log-density differentiated at `shape == 1` gets a wrong
shape-gradient under Mooncake reverse — including the shared-hyperparameter
pooled reconstruction this package's non-centred `pool` builds (a
population-level draw can land a stratum's reconstructed shape on exactly
`1.0`). `LogExpFunctionsChainRulesCoreExt` already ships correct
`ChainRulesCore.rrule`s for both functions, so `@from_rrule` imports them
directly rather than re-deriving the maths.

This is intentional, narrowly-scoped type piracy on functions this package
does not own, matching the workflow Mooncake's own `@from_rrule`/
`@from_chainrules` documentation endorses for closing such gaps from a
downstream package. It should be removed once Mooncake ships its own rule
(reported upstream, see #99).
"""
module ComposedDistributionsMooncakeExt

using ComposedDistributions: _ctor_has_check_args, _split_edge,
                             _throw_unflatten_dimmismatch,
                             _throw_logdensity_dimmismatch,
                             _throw_as_named_dimmismatch,
                             _throw_logpdf_dimmismatch
using LogExpFunctions: xlogy, xlog1py
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_ctor_has_check_args), Any, Tuple}

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(_split_edge), Symbol}

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_unflatten_dimmismatch), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_logdensity_dimmismatch), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_as_named_dimmismatch), Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_logpdf_dimmismatch), Any, Any, Any}

Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{
    typeof(xlogy), Base.IEEEFloat, Base.IEEEFloat}
Mooncake.@from_rrule Mooncake.DefaultCtx Tuple{
    typeof(xlog1py), Base.IEEEFloat, Base.IEEEFloat}

end # module
