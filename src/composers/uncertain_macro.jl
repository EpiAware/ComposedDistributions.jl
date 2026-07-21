# The `@uncertain` macro: a syntax front-end over the positional `uncertain`
# family form. It rewrites syntax only — every type decision (a distribution
# argument marks a prior, a `Real` marks a fixed value) stays with the runtime
# `uncertain(::Type{D}, args...)` method in Uncertain.jl.
#
# The rewrite walks an expression post-order (children first) and, for each
# positional call `D(pos_args...)` whose head `D` is a bare Symbol starting with
# an uppercase letter (a distribution-type constructor) and at least one
# positional argument is itself such an uppercase-headed call (a distribution
# literal read as that parameter's prior), replaces it with
# `uncertain(D, pos_args...)`. Lowercase-headed calls (`compose`, `sequential`,
# `affine`, ...) and qualified heads (`Base.Gamma`) are left as calls but still
# recursed into, so nested trees, `compose(...)` front-ends and modifier
# wrappers all reach their leaf constructors. A constructor with only literal
# arguments (`LogNormal(0.5, 0.4)`) has no distribution-valued argument and is
# left unchanged.

# A call node that looks like a distribution literal used as a prior: a `:call`
# with a bare uppercase Symbol head. Guards non-Symbol heads (a qualified
# `Base.Gamma` is an `Expr(:., ...)` head, not a distribution literal here).
function _is_dist_call(a)
    return a isa Expr && a.head === :call && !isempty(a.args) &&
           a.args[1] isa Symbol && isuppercase(first(string(a.args[1])))
end

# Rewrite one already-child-rewritten node: turn `D(pos...)` into
# `uncertain(D, pos...)` when `D` is an uppercase Symbol head and some
# positional argument is a distribution literal. A keyword-carrying call (a
# `:parameters` block among the arguments) is left unrewritten — the positional
# `uncertain` family form takes no keywords — so the explicit `uncertain`
# spelling stays available for that case.
function _maybe_uncertain(ex)
    (ex isa Expr && ex.head === :call && !isempty(ex.args)) || return ex
    head = ex.args[1]
    (head isa Symbol && isuppercase(first(string(head)))) || return ex
    pos = ex.args[2:end]
    any(a -> a isa Expr && a.head === :parameters, pos) && return ex
    any(_is_dist_call, pos) || return ex
    return Expr(:call, :uncertain, head, pos...)
end

# Recursive post-order rewrite: rebuild every child first (so nested leaves are
# rewritten before their parents), then reconsider the node itself. Non-`Expr`
# leaves (Symbols, literals) pass through unchanged.
function _rewrite_uncertain(ex)
    ex isa Expr || return ex
    new_args = Any[_rewrite_uncertain(a) for a in ex.args]
    return _maybe_uncertain(Expr(ex.head, new_args...))
end

@doc "

Read distribution-valued constructor arguments as parameter priors.

`@uncertain expr` rewrites `expr` so that a distribution literal passed as a
positional argument to a distribution constructor becomes that parameter's
prior, the natural spelling of the positional [`uncertain`](@ref) family form.
It walks the whole expression, so it composes with [`compose`](@ref), the
composer verbs and the ModifiedDistributions wrappers.

Each call `D(pos_args...)` whose head `D` is a distribution type (a bare name
beginning with an uppercase letter) and at least one of whose positional
arguments is itself such a distribution literal is rewritten to
`uncertain(D, pos_args...)`. The runtime then sorts each positional argument: a
`UnivariateDistribution` marks that parameter uncertain (a prior), a `Real`
fixes it. So `@uncertain LogNormal(Normal(0.0, 1.0), 0.5)` is
`uncertain(LogNormal, Normal(0.0, 1.0), 0.5)` (`mu` uncertain, `sigma` fixed at
`0.5`). A constructor with only literal arguments (`LogNormal(0.5, 0.4)`) has no
distribution-valued argument and is left unchanged, and lowercase-headed calls
(`compose`, `affine`, ...) are not themselves rewritten, though their arguments
still are, so a modifier wraps the rewritten uncertain leaf.

A keyword-carrying constructor call and a qualified head (`Base.Gamma`) are left
unrewritten; reach those through the explicit [`uncertain`](@ref) constructor.

# Arguments
- `expr`: an expression building a (possibly composed) distribution, with a
  distribution literal in a parameter slot standing for that parameter's prior.

# Examples
```@example
using ComposedDistributions, Distributions

# `shape` uncertain, `scale` fixed at 1.0.
@uncertain Gamma(Normal(0.7, 0.2), 1.0)

# A whole tree: only the Gamma leaf's shape is made uncertain.
@uncertain compose((
    onset = Gamma(LogNormal(log(2.0), 0.2), 1.0),
    admit = LogNormal(0.5, 0.4)))
```

# See also
- [`uncertain`](@ref): the constructor this expands to.
- [`Uncertain`](@ref): the leaf type it builds.
"
macro uncertain(ex)
    return esc(_rewrite_uncertain(ex))
end
