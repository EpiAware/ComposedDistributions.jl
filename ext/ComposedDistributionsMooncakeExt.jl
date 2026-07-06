# ComposedDistributions × Mooncake
#
# `_ctor_has_check_args(ctor, vals)` reports (via `hasmethod`) whether a leaf
# distribution constructor accepts a `check_args` keyword, so a leaf
# reconstruction (e.g. the DynamicPPL composer-half extension, issue #9) can
# skip the argument check where supported. Its `hasmethod` lowers to a
# `jl_gf_invoke_lookup` foreigncall that Mooncake reverse has no rule for. The
# result is a `Bool` constant with respect to the sampled parameters (only the
# leaf params carry gradients), so a zero-adjoint primitive runs the primal
# unchanged and returns a zero cotangent, keeping any reconstruction built on
# top of it AD-safe under Mooncake reverse.
module ComposedDistributionsMooncakeExt

using ComposedDistributions: _ctor_has_check_args
using Mooncake: Mooncake

Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{
    typeof(_ctor_has_check_args), Any, Tuple}

end # module
