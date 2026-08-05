"""
    ComposedDistributionsMooncakeExt

Shields `_ctor_has_check_args` and `_has_params_method` with a Mooncake
`@zero_adjoint`. Both are plain `hasmethod` reflections, lowering to a
`jl_gf_invoke_lookup` foreigncall that Mooncake reverse has no rule for; each
result is a `Bool` constant with respect to the differentiated parameters, so
a zero-adjoint primitive is sound. `_has_params_method` guards the generic
`param_names` derivation (introspection.jl, S1 leaf-name derivation), called
from `leaf_param_names` on every `_leaf_entry`/`_leaf_flatten_values` seam
call (`unflatten`/`flatten`'s hot path, not just a dormant reconstruction
path like `_ctor_has_check_args`), so it needs the same shield to keep the
codec AD-safe under Mooncake reverse.

Also shields `_split_edge` (the flat-vector codec's `Symbol` path-splitter,
`unflatten`/`flatten`'s unconditional hot-path call), `_normalize_param_name`
(the Greek-to-ASCII parameter-name transliteration `param_names` and the
`uncertain`/`update` kwarg-matching call through
`leaf_param_names`/`_normalize_kwarg_names`, same hot path), and the codec's
`DimensionMismatch`-throwing helpers (`_throw_unflatten_dimmismatch`,
`_throw_as_named_dimmismatch`, `_throw_logpdf_dimmismatch`) with
`@zero_derivative`, both forward and reverse. `_split_edge` calls
`Base.split`, `_normalize_param_name` iterates a `String`'s characters
(`Base.normalize` then a `for c in ...` loop), and the `DimensionMismatch`
helpers interpolate their arguments into an error message via `show`. All
recurse into Base's UTF-8 string-indexing continuation machinery, for which
Mooncake's whole-program rule derivation has no rule (a `sub_ptr`
pointer-arithmetic intrinsic): `_split_edge` and `_normalize_param_name`
unconditionally, on every `unflatten`/`flatten` call regardless of AD, and
the message helpers only when their guard branch throws, but Mooncake still
derives a rule for every reachable branch whether it is taken or not.
`_split_edge`/`_normalize_param_name`'s results never carry a tangent
(`Symbol` inputs and outputs), and the message helpers always throw (a
constant `Union{}` result), so a zero-derivative primitive is sound for each.

The `LogExpFunctions.xlogy`/`xlog1py` Mooncake primitives that used to live
here (#99, extended to forward mode in #214) now come from EpiAwareADTools'
`EpiAwareADToolsLogExpFunctionsMooncakeExt`; this package's hard dependencies
on EpiAwareADTools and LogExpFunctions trigger it as soon as Mooncake loads.
They were hoisted to avoid the duplicate registration clash with
DistributionsInference (DistributionsInference#73), and are deleted once
Mooncake ships its own rule (chalk-lab/Mooncake.jl#1241). The trigger this
package supplies is the non-centred `pool`, whose reconstruction can land a
stratum's shape on exactly `1.0`, where Mooncake's derived rule returns the
wrong gradient.
"""
module ComposedDistributionsMooncakeExt

using ComposedDistributions: _ctor_has_check_args, _has_params_method,
                             _normalize_param_name, _split_edge,
                             _throw_unflatten_dimmismatch,
                             _throw_as_named_dimmismatch,
                             _throw_logpdf_dimmismatch
using Mooncake: Mooncake

# Ahead of the DynamicPPL leaf rebuild that will call it (#9).
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_ctor_has_check_args), Any, Tuple}

# Guards the generic `param_names` derivation, live on the `unflatten`/
# `flatten` hot path since the S1 leaf-name derivation (#345).
Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_has_params_method), Any}

# The codec's string-machinery shields, fixing #146 (and, for
# `_normalize_param_name`, #345's Greek-to-ASCII transliteration).
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{typeof(_split_edge), Symbol}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_normalize_param_name), Symbol}

Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_unflatten_dimmismatch), Any, Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_as_named_dimmismatch), Any, Any}
Mooncake.@zero_derivative Mooncake.DefaultCtx Tuple{
    typeof(_throw_logpdf_dimmismatch), Any, Any, Any}

end # module
