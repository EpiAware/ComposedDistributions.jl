# Composed distributions score a flat vector-valued representation (consumed
# by `logpdf`/AD), but present labelled outputs: a multivariate `rand` as a
# `NamedTuple` keyed by per-value leaf names, `mean`/`var`/`std` of a
# `Parallel` keyed by per-endpoint names. This file wraps the generic
# `_composite_rand` (nesting.jl) realisation by name.

# --- generic realisation ----------------------------------------------------

# The vector-valued realisation of a composer: the generic per-leaf-value draw
# (one value per leaf, a nested composer contributing its own sub-vector). A
# `Resolve` child collapses to its marginal time-to-resolution (its univariate
# `rand`), a `Choose` child to its first alternative.
function _composer_rand(rng::AbstractRNG, d::Union{Sequential, Parallel})
    return _composite_rand(rng, d.components, float(eltype(d)))
end

# Sample an outcome index from the branch probabilities by inverse-CDF (a single
# uniform draw), so no `Categorical` dependency is pulled in. The last outcome
# absorbs any rounding so an index is always returned.
function _sample_branch(rng::AbstractRNG, probs)
    u = rand(rng)
    c = zero(float(eltype(probs)))
    @inbounds for i in 1:(length(probs) - 1)
        c += probs[i]
        u <= c && return i
    end
    return length(probs)
end

# --- the wrap boundary ------------------------------------------------------

# Wrap a vector-valued output `v` into a `NamedTuple` keyed by `names`. The
# lengths must agree (an internal invariant; a mismatch is a bug in the name
# derivation). A result that is already a `NamedTuple` passes through unchanged.
function _as_named(names::Tuple, v::AbstractVector)
    length(names) == length(v) || throw(DimensionMismatch(
        "labelled output has $(length(v)) values but $(length(names)) names " *
        "$(collect(names))"))
    return NamedTuple{names}(Tuple(v))
end
_as_named(::Tuple, v::NamedTuple) = v

# Label a top-level composer realisation: the flat per-value vector wrapped by
# the composer's per-value leaf names.
function _named_composer_rand(rng::AbstractRNG, d)
    return _as_named(_output_names(d), _composer_rand(rng, d))
end

# The output names matching the per-value vector of `rand(d)`: the per-value leaf
# names (one name per leaf value, no latent origin).
_output_names(d::Union{Sequential, Parallel}) = _value_names(d)

# Per-value leaf names of a composer, in the same depth-first layout as the
# generic `_composite_rand`: one name per leaf value, a nested composer recursing
# into its children, a leaf / `Resolve` named by its component name. Names from
# nested levels are joined into a single dotted-underscore path (`:r1_step_1`) so
# a positional default repeated across nesting levels still yields unique
# NamedTuple keys (a leaf at the top level keeps its bare name).
function _value_names(d::Union{Sequential, Parallel})
    out = Symbol[]
    names = component_names(d)
    for i in eachindex(d.components)
        _append_value_names!(out, (names[i],), d.components[i])
    end
    return Tuple(out)
end

function _append_value_names!(out, path::Tuple,
        child::Union{Sequential, Parallel})
    cnames = component_names(child)
    for i in eachindex(child.components)
        _append_value_names!(out, (path..., cnames[i]), child.components[i])
    end
    return out
end
function _append_value_names!(out, path::Tuple, ::Any)
    push!(out, _join_value_path(path))
    return out
end

# Join a value-name path into one `Symbol`: a single-level path keeps its bare
# name (`:a`); a nested path joins with `_` (`:r1_step_1`). This is the
# underscored ("_" separator) event/value namespace, distinct from the dotted
# ("." separator) parameter-path namespace (`_join_path` in introspection.jl).
function _join_value_path(path::Tuple)
    length(path) == 1 ? path[1] :
    Symbol(join(string.(path), "_"))
end

# --- per-endpoint names of a Parallel (the multivariate marginal) -----------
#
# `_endpoint_names(d)` is the tuple of names for the per-endpoint moment vector
# `mean(d::Parallel)` / `var` / `std` produces. It mirrors the
# `_endpoint_moment_vector` walk exactly: one name per collapsed branch
# endpoint, in branch order, a nested `Parallel` flattening its own endpoints
# in. A `Sequential` / `Resolve` / leaf branch collapses to its single endpoint
# and is named by its branch (component) name.

function _endpoint_names(d::Parallel)
    out = Symbol[]
    names = component_names(d)
    for i in eachindex(d.components)
        _append_endpoint_names!(out, names[i], d.components[i])
    end
    return Tuple(out)
end

# A nested `Parallel` branch flattens its own endpoint names in (under the
# nested branch names), mirroring `_append_endpoint_moments!`.
function _append_endpoint_names!(out, ::Symbol, branch::Parallel)
    bnames = component_names(branch)
    for i in eachindex(branch.components)
        _append_endpoint_names!(out, bnames[i], branch.components[i])
    end
    return out
end
# Every other branch (a `Sequential` / `Resolve` / leaf) collapses to one
# endpoint, named by its branch name.
function _append_endpoint_names!(out, name::Symbol, ::Any)
    push!(out, name)
    return out
end

# --- NamedTuple input to logpdf ---------------------------------------------
#
# `logpdf` scores the vector-valued representation; a labelled `NamedTuple` draw
# (as `rand(d)` returns) is accepted and converted to the scored vector by name
# first, so a self-labelling draw round-trips straight back through
# `logpdf(d, rand(d))`. Field order does not matter; the names do.

function logpdf(d::Union{Sequential, Parallel}, x::NamedTuple)
    return logpdf(d, _named_value_vector(d, x))
end

# The per-value vector of a composer from a labelled draw, matched to the
# `_value_names(d)` layout by name.
function _named_value_vector(d::Union{Sequential, Parallel}, x::NamedTuple)
    vnames = _value_names(d)
    for k in keys(x)
        k in vnames || throw(ArgumentError(
            "draw field $(repr(k)) is not a value of this composer; expected " *
            "$(collect(vnames)) (reordering is allowed; names are not)"))
    end
    out = Vector{Float64}(undef, length(vnames))
    for (i, name) in enumerate(vnames)
        haskey(x, name) || throw(ArgumentError(
            "draw is missing required value $(repr(name)); expected " *
            "$(collect(vnames))"))
        out[i] = Float64(x[name])
    end
    return out
end
