"""
    ComposedDistributionsMooncakeExt

Shields `_ctor_has_check_args` with a Mooncake `@zero_adjoint`. Its
`hasmethod` reflection lowers to a `jl_gf_invoke_lookup` foreigncall that
Mooncake reverse has no rule for; the result is a `Bool` constant with
respect to the differentiated parameters, so a zero-adjoint primitive is
sound and keeps any reconstruction built on top of it (e.g. a future
DynamicPPL leaf rebuild, issue #9) AD-safe under Mooncake reverse.
"""
module ComposedDistributionsMooncakeExt

using ComposedDistributions: _ctor_has_check_args
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_ctor_has_check_args), Any, Tuple}

end # module
