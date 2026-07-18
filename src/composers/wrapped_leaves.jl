# Distributions.jl's `censored(...)` wrapper (`Censored`) reaches full
# leaf-protocol parity with `Truncated` here, mirroring every hook Truncated
# already has across introspection.jl / Shared.jl / varying.jl / Uncertain.jl
# / codec_gen.jl, rather than adding a Censored branch inline to each of those
# files. Kept as its own file so it does not collide with the ongoing
# introspection.jl work elsewhere in the org (design-verbs' CD#202).
#
# Also hosts the tree-level guard: `truncated`/`censored` applied to a WHOLE
# composed distribution's event tree, which is either impossible
# (`Sequential`/`Parallel` are multivariate; Distributions.jl's
# truncated/censored are univariate-only, so plain Julia dispatch already
# errors there) or ill-defined even though it type-checks (`Resolve`/`Compete`
# satisfy `UnivariateDistribution`, but their outcome is a structured named
# event, not the plain scalar the inverse-cdf machinery needs, so `rand` on
# the wrapped result would fail with an unrelated internal Distributions.jl
# error rather than anything informative). Guarded here instead: one clear
# error at construction time, for either wrapper, over any composed node.

# --- Censored leaf-protocol parity with Truncated ---------------------------

free_leaf(d::Distributions.Censored) = free_leaf(d.uncensored)
function rewrap_leaf(d::Distributions.Censored, inner)
    return censored(rewrap_leaf(d.uncensored, inner); lower = d.lower,
        upper = d.upper)
end

uncertain_specs(d::Distributions.Censored) = uncertain_specs(d.uncensored)
extra_leaf_params(d::Distributions.Censored) = extra_leaf_params(d.uncensored)
set_extra_leaf_params(d::Distributions.Censored, ::NamedTuple{()}) = d
function set_extra_leaf_params(d::Distributions.Censored, vals::NamedTuple)
    return censored(set_extra_leaf_params(d.uncensored, vals);
        lower = d.lower, upper = d.upper)
end

shared_tag(d::Distributions.Censored) = shared_tag(d.uncensored)
has_varying(d::Distributions.Censored) = has_varying(d.uncensored)

# --- `censored` pushed inside an `Uncertain` template, mirroring `truncated` -
#
# Without this, `censored(u::Uncertain, ...)` would nest as
# `Censored{Uncertain}`, and the generated codec (which only recognises
# `Uncertain` as the OUTERMOST wrapper, codec_gen.jl's `_leaf_unflatten_expr`)
# would silently treat the whole leaf as fixed, dropping the estimated
# parameter. Mirrors Uncertain.jl's `truncated(d::Uncertain, ...)` method set
# exactly, one method per upstream `censored` signature shape.

function Distributions.censored(d::Uncertain, lower::T,
        upper::T) where {T <: Real}
    return Uncertain(censored(d.template, lower, upper), d.specs)
end
function Distributions.censored(d::Uncertain, lower::Real, ::Nothing)
    return Uncertain(censored(d.template, lower, nothing), d.specs)
end
function Distributions.censored(d::Uncertain, ::Nothing, upper::Real)
    return Uncertain(censored(d.template, nothing, upper), d.specs)
end
Distributions.censored(d::Uncertain, ::Nothing, ::Nothing) = d

# --- forbid truncated()/censored() on a whole composed tree -----------------
#
# One method per upstream signature SHAPE (matching Distributions.jl's own
# `truncated`/`censored` dispatch table exactly, with `AbstractComposedDistribution`
# in place of `UnivariateDistribution`), so each override is strictly more
# specific than the corresponding upstream method in every argument position
# and no method-ambiguity is introduced. A blanket `d, args...` catch-all would
# be ambiguous with upstream's `(d::UnivariateDistribution, l::T, u::T)` for a
# `Resolve`/`Compete` argument (more specific in the first position, less
# specific in the rest), so the shapes are spelled out instead.

function _reject_tree_wrap(verb::String, d)
    throw(ArgumentError(
        "$verb(...) is not supported on a composed distribution's event " *
        "tree ($(nameof(typeof(d)))): Sequential/Parallel combine several " *
        "named child distributions, not a single scalar the inverse-cdf " *
        "machinery can act on, and a Resolve/Compete outcome is a " *
        "structured named event rather than a plain scalar, so $(verb)ing " *
        "the whole node is not well-defined even though it type-checks as " *
        "univariate. $(uppercasefirst(verb)) the individual leaf delays " *
        "instead, e.g. `compose((onset = $verb(Gamma(2.0, 1.0); " *
        "upper = 10.0), ...))`."))
end

function Distributions.truncated(d::AbstractComposedDistribution,
        l::T, u::T) where {T <: Real}
    return _reject_tree_wrap("truncated", d)
end
function Distributions.truncated(d::AbstractComposedDistribution, ::Nothing, ::Nothing)
    _reject_tree_wrap("truncated", d)
end
function Distributions.truncated(d::AbstractComposedDistribution,
        l::Real, u::Real)
    return _reject_tree_wrap("truncated", d)
end
function Distributions.truncated(d::AbstractComposedDistribution;
        lower::Union{Real, Nothing} = nothing, upper::Union{Real, Nothing} = nothing)
    return _reject_tree_wrap("truncated", d)
end
function Distributions.truncated(d::AbstractComposedDistribution,
        ::Nothing, u::Real)
    return _reject_tree_wrap("truncated", d)
end
function Distributions.truncated(d::AbstractComposedDistribution,
        l::Real, ::Nothing)
    return _reject_tree_wrap("truncated", d)
end

function Distributions.censored(d::AbstractComposedDistribution,
        l::T, u::T) where {T <: Real}
    return _reject_tree_wrap("censored", d)
end
function Distributions.censored(d::AbstractComposedDistribution,
        ::Nothing, u::Real)
    return _reject_tree_wrap("censored", d)
end
function Distributions.censored(d::AbstractComposedDistribution,
        l::Real, ::Nothing)
    return _reject_tree_wrap("censored", d)
end
function Distributions.censored(d::AbstractComposedDistribution,
        l::Real, u::Real)
    return _reject_tree_wrap("censored", d)
end
function Distributions.censored(d::AbstractComposedDistribution, ::Nothing, ::Nothing)
    _reject_tree_wrap("censored", d)
end
function Distributions.censored(d::AbstractComposedDistribution;
        lower::Union{Real, Nothing} = nothing, upper::Union{Real, Nothing} = nothing)
    return _reject_tree_wrap("censored", d)
end
