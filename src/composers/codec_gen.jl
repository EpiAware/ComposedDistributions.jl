# Generation-time (type-domain) flat-vector <-> nested-NamedTuple codec.
#
# Since #178 PR 1, every composer/wrapper carries its layout-affecting
# structure (names, tags, groups) as TYPE PARAMETERS, not runtime fields. That
# means the whole flat-vector layout -- which leaf owns which slot, the
# shared-tag/pool-group dedup with root-lift, the Resolve stick-breaking
# count -- is a function of `typeof(d)` ALONE. This file walks that type once
# per distinct concrete tree shape (inside `@generated` function bodies) and
# emits code with the slot indices baked in as literals, replacing the old
# runtime `Dict{Symbol, Any}` walk (`params_table` + `_nest_insert!` +
# `_freeze_tree`) that `unflatten` used to re-run on every call. That
# Dict/`Any`-typed walk is #162's root cause (Enzyme's type analysis cannot
# see through a `Dict{Symbol, Any}`/heap-boxed reconstruction); the generated
# walk below produces a concretely-typed (`@inferred`-stable) nested
# NamedTuple instead, which both fixes Enzyme reverse and removes the
# dominant per-evaluation cost (issue #178's spike measured the Dict walk at
# 50-83% of a `logdensity` call).
#
# `update(d, nt)` (ordinary recursion over the nested NamedTuple, ADDING no
# Dict of its own) is UNCHANGED by this file: `reconstruct` below composes the
# new generated `unflatten` with the existing `update`, rather than
# re-deriving `update`'s full leaf-rebuild logic (merge mode, pooled
# reconstruction, stick-breaking, Choose namespacing, extras) a second time.
# That keeps `_update_leaf`/`_uncertain_leaf`/`_reconstruct_pooled_leaf` the
# single source of truth for "how a leaf rebuilds from values" (the #174
# leaf-protocol hooks), exactly as the design review required.

# --- shared guards -----------------------------------------------------------
#
# Moved here from logdensity.jl (the codec's home before this file existed):
# used by the generated `unflatten`/`flatten`/`flat_dimension` below.

# Refuse eagerly when `d` still carries a `Varying` leaf: unlike `Uncertain`,
# whose row already tracks concrete template values, a `Varying` leaf's row
# reports its `reference` only, so the codec would otherwise silently ignore
# the covariate dependence rather than score it.
function _reject_varying(d, what)
    has_varying(d) && throw(ArgumentError(
        "cannot $what a tree with varying leaves; resolve them with " *
        "`instantiate(tree, context)` first"))
    return nothing
end

# Hoisted into its own `@noinline` function (not inlined into `unflatten`'s
# body) for the same Mooncake `show`-in-a-differentiated-function reason the
# rest of the package's dimension guards are: see `logdensity.jl`'s matching
# note. `est` is any object whose `length` is the estimated-row count (a
# `UnitRange` here, since the generated caller has only the count, not the
# table).
@noinline function _throw_unflatten_dimmismatch(x, est, d)
    throw(DimensionMismatch(
        "flat vector has length $(length(x)) but $d has " *
        "$(length(est)) estimated parameters"))
end

# Read the value at `(path..., param)` of a nested NamedTuple. Still used by
# `Pool.jl`'s `_pool_centred_logprior`, which reads a centred pooled member's
# latent straight off the `unflatten`ed `nt` at a runtime-known path (a
# per-evaluation but not per-slot walk, so a plain function is fine here).
function _read_path(nt::NamedTuple, path::Tuple, param::Symbol)
    node = nt
    for k in path
        node = getproperty(node, k)
    end
    return getproperty(node, param)
end

# --- Type-level leaf-protocol companions ------------------------------------
#
# The existing leaf-protocol hooks (`free_leaf`, `_param_names`,
# `extra_leaf_params`, ...) dispatch on an INSTANCE, which the generation-time
# walk does not have (only `typeof(d)`). These companions answer the same
# questions from the TYPE alone; they are additive (the instance-based hooks
# are untouched) and follow the same per-family dispatch table so the two stay
# in lockstep. A leaf-wrapper type without a `_leaf_free_type`/`_extra_names_of`
# method here falls back to the identity/empty default, exactly like the
# instance-based hooks do for an unmapped leaf.

@doc "
The peeled free-delay TYPE of a (possibly wrapped) leaf type, mirroring
`free_leaf` at the type level. The base identity is `L` itself; `Truncated`
peels to its untruncated inner type. A leaf-wrapper extension (censoring,
modifiers) adds its own method on its own type, in step with its instance-based
`free_leaf` method.
" _leaf_free_type(::Type{L}) where {L} = L
_leaf_free_type(::Type{<:Distributions.Truncated{D}}) where {D} = _leaf_free_type(D)

@doc "
The native parameter name labels of a peeled free-delay TYPE, mirroring
`_param_names` at the type level (dispatch table kept in step with it).
Returns `()` for an unmapped family, exactly like the instance-based fallback;
`leaf_param_names`'s positional `:param_i` padding then applies at the
generation-time layer too.
" _param_names_of(::Type) = ()
_param_names_of(::Type{<:Distributions.Normal}) = (:mu, :sigma)
_param_names_of(::Type{<:Distributions.LogNormal}) = (:mu, :sigma)
_param_names_of(::Type{<:Distributions.Gamma}) = (:shape, :scale)
_param_names_of(::Type{<:Distributions.Weibull}) = (:shape, :scale)
_param_names_of(::Type{<:Distributions.Exponential}) = (:scale,)
_param_names_of(::Type{<:Distributions.Uniform}) = (:lower, :upper)

@doc "
The extra (modifier-owned) parameter names of a leaf TYPE, mirroring
`extra_leaf_params`'s key set at the type level. Defaults to `()`; a
leaf-wrapper extension reporting a non-empty `extra_leaf_params` (e.g.
ModifiedDistributions' `thin(...)`) adds its own method.
" _extra_names_of(::Type) = ()
_extra_names_of(::Type{<:Distributions.Truncated{D}}) where {D} = _extra_names_of(D)

# The native parameter arity of a peeled free-delay TYPE: the length of
# `Distributions.params(instance)`. Every ordinary Distributions.jl leaf
# (Gamma, Normal, LogNormal, Weibull, Exponential, Uniform, Beta, ...) reports
# exactly its own struct fields as `params`, so `fieldcount` reads the arity
# straight off the type with no instance needed -- and, unlike
# `Base.return_types`, code reflection of any kind is disallowed inside a
# `@generated` function body, so this must be a structural (not inferential)
# query. A leaf type whose `params` is NOT its own fields 1:1 (a custom
# moment-parameterised wrapper, `_leaf_ctor`'s motivating case) needs its own
# `_params_arity_of` method alongside its `_param_names_of`/`_leaf_ctor`
# override.
_params_arity_of(::Type{L}) where {L} = fieldcount(L)

# The full (native..., extra...) parameter name tuple of a (possibly wrapped)
# leaf TYPE `L`, mirroring `leaf_param_names` at the type level exactly:
# native names come from the PEELED free delay (padding unmapped names
# positionally), but extras are read off `L` itself, UNPEELED -- an extra
# parameter (e.g. `thin`'s reporting probability) is owned by the wrapper, not
# the inner free delay, exactly as `extra_leaf_params(leaf)` (not
# `extra_leaf_params(free_leaf(leaf))`) reads it at the instance level.
#
# `Base.invokelatest` on every one of these type-level hooks: they are called
# from a `@generated` function's GENERATOR (this function is reached from
# `_leaf_unflatten_expr`, itself reached from `unflatten`/`flat_dimension`/
# `flatten`'s generator bodies), and a generator's own world age can be fixed
# BEFORE a package extension (e.g. `ComposedDistributionsModifiedDistributionsExt`)
# has added its `_leaf_free_type`/`_extra_names_of` methods for a wrapper type
# like `Transformed`, silently baking in the un-peeled fallback. Forcing the
# latest world age here is generation-time-only cost (this runs once per
# distinct tree TYPE, never per evaluation) and makes the walk see every
# extension method that is loaded by the time the generated function first
# runs, not just the ones loaded before it was first triggered.
function _leaf_type_param_names(::Type{L}) where {L}
    freeL = Base.invokelatest(_leaf_free_type, L)
    base = Base.invokelatest(_param_names_of, freeL)
    n = Base.invokelatest(_params_arity_of, freeL)
    native = ntuple(n) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
    return (native..., Base.invokelatest(_extra_names_of, L)...)
end

# --- generation-time layout context -----------------------------------------
#
# Mutable, generation-time-only bookkeeping threaded through the whole type
# walk (mirrors the verified spike's `seen`/`idxref`/`tagkeys`/`tagvals`, with
# a second root-lift namespace added for pool groups). Never appears in the
# emitted code itself -- only literal indices and literal name tuples do.
mutable struct _CodecCtx
    idx::Int
    seen_tags::Set{Symbol}
    seen_groups::Set{Symbol}
    tag_keys::Vector{Symbol}
    tag_vals::Vector{Any}
    group_keys::Vector{Symbol}
    group_vals::Vector{Any}
end
_CodecCtx() = _CodecCtx(0, Set{Symbol}(), Set{Symbol}(), Symbol[], Any[],
    Symbol[], Any[])

# --- the shared walk: build an `unflatten` NamedTuple-construction Expr -----
#
# `_unflatten_expr(access, ::Type{T}, ctx)` returns the Expr that constructs
# the NamedTuple entry for the node at `access` (of type `T`), or `nothing`
# when the node is a tag-suppressed leaf occurrence (its value lives at the
# root-lifted tag entry instead, never positionally). `access` is an `Expr`/
# `Symbol` reading the node from the top-level `d` argument.

function _unflatten_expr(access, ::Type{T}, ctx::_CodecCtx) where {T}
    if T <: Sequential || T <: Parallel
        return _composer_unflatten_expr(access, :components, T, ctx)
    elseif T <: Choose
        return _composer_unflatten_expr(access, :alternatives, T, ctx)
    elseif T <: Resolve
        return _resolve_unflatten_expr(access, T, ctx)
    elseif T <: Compete
        return _composer_unflatten_expr(access, :delays, T, ctx)
    elseif T <: Union{Convolved, Difference}
        return _composite_unflatten_expr(access, T, ctx)
    else
        return _leaf_unflatten_expr(access, T, ctx)
    end
end

# The component TYPES of a see-through composite leaf (`Convolved`/
# `Difference` used as a leaf, `convolved_interop.jl`), mirroring
# `_node_children` at the type level: `Convolved{C<:Tuple, Method}`'s
# components are `C`'s own type parameters; `Difference{X, Y, Method}` has
# exactly the two fixed operand types.
_composite_child_types(::Type{<:Convolved{C}}) where {C} = Tuple(C.parameters)
_composite_child_types(::Type{<:Difference{X, Y}}) where {X, Y} = (X, Y)

# A composite leaf's node children are namespaced `component_1, component_2,
# ...` (mirroring `_composite_component_names` in `convolved_interop.jl`,
# computed independently here so this file has no include-order dependency on
# it) and read at runtime through the generic `_node_children` accessor (so
# `Convolved`'s `.components` tuple and `Difference`'s `(.x, .y)` pair share
# one code path, exactly as `_walk_rows!`/`_update` already do).
function _composite_unflatten_expr(access, ::Type{T}, ctx::_CodecCtx) where {T}
    ctypes = _composite_child_types(T)
    keys_out = Symbol[]
    vals_out = Any[]
    for i in eachindex(ctypes)
        child_access = :(ComposedDistributions._node_children($access)[$i])
        e = _unflatten_expr(child_access, ctypes[i], ctx)
        e === nothing && continue
        push!(keys_out, Symbol(:component_, i))
        push!(vals_out, e)
    end
    return :(NamedTuple{$(Tuple(keys_out))}(($(vals_out...),)))
end

# Sequential/Parallel/Choose/Compete share the same shape: named children
# recursed positionally, skipping any `nothing` (tag-suppressed) entry.
function _composer_unflatten_expr(
        access, field::Symbol, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    C = T.parameters[2]
    ctypes = C.parameters
    keys_out = Symbol[]
    vals_out = Any[]
    for i in eachindex(names)
        child_access = :($access.$field[$i])
        e = _unflatten_expr(child_access, ctypes[i], ctx)
        e === nothing && continue
        push!(keys_out, names[i])
        push!(vals_out, e)
    end
    return :(NamedTuple{$(Tuple(keys_out))}(($(vals_out...),)))
end

# `Resolve{names, D, P, S}`: the outcome delays (skipping a `NoEvent` branch,
# which carries no parameters and no entry, mirroring `_walk_rows!`), plus a
# `branch_probs` entry: the K-1 stick coordinates when `S <: Dirichlet` (the
# node's simplex is estimated), else the current fixed per-outcome
# probabilities (read at runtime, not baked in).
function _resolve_unflatten_expr(access, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    D = T.parameters[2]
    S = T.parameters[4]
    dtypes = D.parameters
    keys_out = Symbol[]
    vals_out = Any[]
    for i in eachindex(names)
        dtypes[i] <: NoEvent && continue
        child_access = :($access.delays[$i])
        e = _unflatten_expr(child_access, dtypes[i], ctx)
        e === nothing && continue
        push!(keys_out, names[i])
        push!(vals_out, e)
    end
    bp_expr = if S <: Distributions.Dirichlet
        K = length(names)
        stick_names = ntuple(k -> Symbol(:stick_, k), K - 1)
        stick_vals = map(1:(K - 1)) do _
            ctx.idx += 1
            :(x[$(ctx.idx)])
        end
        :(NamedTuple{$stick_names}(($(stick_vals...),)))
    else
        probs_vals = [:($access.branch_probs[$k]) for k in eachindex(names)]
        :(NamedTuple{$names}(($(probs_vals...),)))
    end
    push!(keys_out, :branch_probs)
    push!(vals_out, bp_expr)
    return :(NamedTuple{$(Tuple(keys_out))}(($(vals_out...),)))
end

# Leaf case: peels a `Shared` tag (root-lift + dedup) and an `Uncertain`
# wrapper (spec keys, including a `Pool` spec), then builds the leaf's own
# `(native..., extra...)` NamedTuple entry -- fixed parameters read from the
# CURRENT instance at runtime, estimated ones from the next `x` slot(s). Every
# leaf gets an entry (fixed or not), matching the old row walk, which emitted a
# fixed leaf's rows too.
function _leaf_unflatten_expr(access, ::Type{L}, ctx::_CodecCtx) where {L}
    if L <: Shared
        tag = L.parameters[1]::Symbol
        D = L.parameters[2]
        inner_access = :($access.dist)
        tag in ctx.seen_tags && return nothing
        push!(ctx.seen_tags, tag)
        inner_entry = _unflatten_expr(inner_access, D, ctx)
        push!(ctx.tag_keys, tag)
        push!(ctx.tag_vals, inner_entry)
        return nothing
    end

    if L <: Uncertain
        Ltempl = L.parameters[2]
        S = L.parameters[3]
        speckeys = S.parameters[1]::Tuple
        specvaltypes = Tuple(S.parameters[2].parameters)
    else
        Ltempl = L
        speckeys = ()
        specvaltypes = ()
    end
    n = Base.invokelatest(_params_arity_of, Base.invokelatest(_leaf_free_type, Ltempl))
    allnames = _leaf_type_param_names(Ltempl)
    vals = Vector{Any}(undef, length(allnames))
    for (i, pname) in enumerate(allnames)
        j = findfirst(==(pname), speckeys)
        if j === nothing
            vals[i] = i <= n ?
                      :(Distributions.params(ComposedDistributions.free_leaf($access))[$i]) :
                      :(ComposedDistributions.extra_leaf_params($access)[$(QuoteNode(pname))].value)
            continue
        end
        specT = specvaltypes[j]
        if specT <: Pool
            group = specT.parameters[1]::Symbol
            noncentred = specT.parameters[2]::Bool
            if !(group in ctx.seen_groups)
                push!(ctx.seen_groups, group)
                hyper_expr = _pool_hyper_unflatten_expr(specT, ctx)
                push!(ctx.group_keys, group)
                push!(ctx.group_vals, hyper_expr)
            end
            ctx.idx += 1
            vals[i] = noncentred ? :((z = x[$(ctx.idx)],)) : :(x[$(ctx.idx)])
        else
            ctx.idx += 1
            vals[i] = :(x[$(ctx.idx)])
        end
    end
    return :(NamedTuple{$allnames}(($(vals...),)))
end

# A pooling group's hyperparameter entry (root-lifted, emitted once at the
# group's first member): the population's own spec'd parameter names, in
# population-param order, each consuming one `x` slot. A population with no
# uncertain specs (fully fixed) contributes an empty NamedTuple, mirroring
# `_pool_hyper_rows!`.
function _pool_hyper_unflatten_expr(specT::Type{<:Pool}, ctx::_CodecCtx)
    P = specT.parameters[3]
    P <: Uncertain || return :(NamedTuple())
    Ptempl = P.parameters[2]
    PS = P.parameters[3]
    speckeys = PS.parameters[1]::Tuple
    pnames = _leaf_type_param_names(Ptempl)
    keys_out = Symbol[]
    vals_out = Any[]
    for pname in pnames
        pname in speckeys || continue
        ctx.idx += 1
        push!(keys_out, pname)
        push!(vals_out, :(x[$(ctx.idx)]))
    end
    return :(NamedTuple{$(Tuple(keys_out))}(($(vals_out...),)))
end

# Merge the root node's own NamedTuple with the root-lifted tag/group entries
# (each keyed by tag/group name), matching the old Dict walk's flat top-level
# namespace. `_validate_tree_names` (called once at `as_logdensity`
# construction, not here) is what actually guards against a name collision
# across the three namespaces; a collision here just silently prefers the
# later `merge` argument, exactly as the old `Dict` insert did.
function _root_merge_expr(root_expr, ctx::_CodecCtx)
    parts = Any[root_expr]
    if !isempty(ctx.tag_keys)
        push!(parts,
            :(NamedTuple{$(Tuple(ctx.tag_keys))}(($(ctx.tag_vals...),))))
    end
    if !isempty(ctx.group_keys)
        push!(parts,
            :(NamedTuple{$(Tuple(ctx.group_keys))}(($(ctx.group_vals...),))))
    end
    length(parts) == 1 && return root_expr
    return :(merge($(parts...)))
end

@doc "

Rebuild the full nested parameter `NamedTuple` from an estimated flat vector.

`unflatten(d, x)` maps the estimated flat vector `x` (the spec'd parameters,
e.g. a draw from a sampler) back to the full nested `NamedTuple`
[`update`](@ref) consumes: each estimated parameter takes its value from `x`,
each fixed parameter its template value. It is the inverse of [`flatten`](@ref),
so `update(d, unflatten(d, x))` collapses every uncertain leaf at the draw while
holding the fixed parameters at the template.

Generated once per distinct tree TYPE from a compile-time layout walk (no
`Dict`, no intermediate `Any`-typed accumulation), so the result is
`@inferred`-concrete and the reverse-mode AD backends (including Enzyme, #162)
differentiate through it.

# Arguments
- `d`: the composed distribution whose table fixes the layout.
- `x`: an estimated flat vector of length [`flat_dimension`](@ref)`(d)`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
# One estimated parameter (onset_admit.shape); the rest stay at the template.
# Public but not exported; reach it by the qualified name.
update(tree, ComposedDistributions.unflatten(tree, [3.0]))
```

# See also
- [`flatten`](@ref): the inverse, nested NamedTuple -> flat vector.
- [`reconstruct`](@ref): flat vector straight to a rebuilt distribution.
- [`update`](@ref): rebuild the distribution from the result.
"
@generated function unflatten(d::T, x::AbstractVector) where {T <:
                                                              AbstractComposedDistribution}
    ctx = _CodecCtx()
    root_expr = _unflatten_expr(:d, T, ctx)
    merged = _root_merge_expr(root_expr, ctx)
    n = ctx.idx
    return quote
        _reject_varying(d, "unflatten")
        length(x) == $n || _throw_unflatten_dimmismatch(x, 1:($n), d)
        $merged
    end
end

@doc "

The estimated parameter dimension of a composed distribution.

`flat_dimension(d)` is the number of scalar ESTIMATED parameters: the count of
[`uncertain`](@ref) specs across the tree, i.e. the [`params_table`](@ref) rows
whose `prior` column carries a spec. A fixed (non-uncertain) leaf contributes
nothing, so a tree with no uncertain leaves has flat dimension 0. It is the
length of the flat vector [`flatten`](@ref) produces and [`unflatten`](@ref)
consumes. Read straight off the same compile-time layout walk `unflatten` uses
(a literal count baked in at generation time), so it cannot drift from the
codec.

# Arguments
- `d`: a composed distribution.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
# Public but not exported; reach it by the qualified name. Only onset_admit's
# shape is uncertain, so the dimension is 1.
ComposedDistributions.flat_dimension(tree)
```

# See also
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec.
"
@generated function flat_dimension(d::T) where {T <: AbstractComposedDistribution}
    ctx = _CodecCtx()
    Base.invokelatest(_unflatten_expr, :d, T, ctx)
    n = ctx.idx
    return quote
        _reject_varying(d, "compute the flat dimension of")
        $n
    end
end

# --- flatten: read an existing nested NamedTuple back to the flat vector ----
#
# Shares the SAME generation-time walk/dedup as `unflatten` (called on `T`,
# not `NT`: the tree TYPE alone fixes which slots are estimated and their
# order), but instead of building NamedTuple-construction expressions it
# builds NamedTuple-READ expressions against the `nt` argument, appended in
# estimated-slot order. Not on the per-gradient hot path (only `unflatten`/
# `reconstruct` are), so the read is a plain generated view rather than a
# further-optimised primitive.
#
# A root-lifted entry (a `Shared` tag or a `Pool` group's hyperparameters) is
# read directly off the LITERAL `nt` argument (`:(nt.$tag)`/`:(nt.$group...)`),
# never off the locally-threaded `nt_access`: that is exactly where
# `unflatten` places it, regardless of how deep the tagged/pooled leaf sits
# structurally. Since the generated function's argument is always named `nt`,
# that root reference needs no extra bookkeeping to thread through the walk.

function _flatten_reads!(exprs::Vector, nt_access, ::Type{T},
        ctx::_CodecCtx) where {T}
    if T <: Union{Sequential, Parallel, Choose, Compete}
        _composer_flatten_reads!(exprs, nt_access, T, ctx)
    elseif T <: Resolve
        _resolve_flatten_reads!(exprs, nt_access, T, ctx)
    elseif T <: Union{Convolved, Difference}
        _composite_flatten_reads!(exprs, nt_access, T, ctx)
    else
        _leaf_flatten_reads!(exprs, nt_access, T, ctx)
    end
    return nothing
end

function _composite_flatten_reads!(
        exprs::Vector, nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    ctypes = _composite_child_types(T)
    for i in eachindex(ctypes)
        _flatten_reads!(
            exprs, :($nt_access.$(Symbol(:component_, i))), ctypes[i], ctx)
    end
    return nothing
end

function _composer_flatten_reads!(
        exprs::Vector, nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    C = T.parameters[2]
    ctypes = C.parameters
    for i in eachindex(names)
        _flatten_reads!(exprs, :($nt_access.$(names[i])), ctypes[i], ctx)
    end
    return nothing
end

function _resolve_flatten_reads!(
        exprs::Vector, nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    D = T.parameters[2]
    S = T.parameters[4]
    dtypes = D.parameters
    for i in eachindex(names)
        dtypes[i] <: NoEvent && continue
        _flatten_reads!(exprs, :($nt_access.$(names[i])), dtypes[i], ctx)
    end
    if S <: Distributions.Dirichlet
        K = length(names)
        bp_access = :($nt_access.branch_probs)
        for k in 1:(K - 1)
            ctx.idx += 1
            push!(exprs, :($bp_access.$(Symbol(:stick_, k))))
        end
    end
    return nothing
end

# Leaf case: a naive `child_access` is passed in even for a `Shared` child
# (built by the caller as if it owned a positional key); when `L <: Shared`
# that access is simply never used below -- a first occurrence reads off the
# literal root `nt.$tag` instead, and a later occurrence returns immediately
# without emitting anything, matching `unflatten`'s suppression exactly.
function _leaf_flatten_reads!(exprs::Vector, nt_access, ::Type{L},
        ctx::_CodecCtx) where {L}
    if L <: Shared
        tag = L.parameters[1]::Symbol
        D = L.parameters[2]
        tag in ctx.seen_tags && return nothing
        push!(ctx.seen_tags, tag)
        _flatten_reads!(exprs, :(nt.$tag), D, ctx)
        return nothing
    end

    if L <: Uncertain
        Ltempl = L.parameters[2]
        S = L.parameters[3]
        speckeys = S.parameters[1]::Tuple
        specvaltypes = Tuple(S.parameters[2].parameters)
    else
        Ltempl = L
        speckeys = ()
        specvaltypes = ()
    end
    allnames = _leaf_type_param_names(Ltempl)
    for pname in allnames
        j = findfirst(==(pname), speckeys)
        j === nothing && continue
        specT = specvaltypes[j]
        access = :($nt_access.$pname)
        if specT <: Pool
            group = specT.parameters[1]::Symbol
            noncentred = specT.parameters[2]::Bool
            if !(group in ctx.seen_groups)
                push!(ctx.seen_groups, group)
                _pool_hyper_flatten_reads!(exprs, group, specT, ctx)
            end
            ctx.idx += 1
            push!(exprs, noncentred ? :($access.z) : access)
        else
            ctx.idx += 1
            push!(exprs, access)
        end
    end
    return nothing
end

# A pooling group's hyperparameters, read off the literal root `nt.<group>`
# (exactly where `unflatten` root-lifts them), in the same population-param
# order `_pool_hyper_unflatten_expr` writes them in.
function _pool_hyper_flatten_reads!(
        exprs::Vector, group::Symbol, specT::Type{<:Pool}, ctx::_CodecCtx)
    P = specT.parameters[3]
    P <: Uncertain || return nothing
    Ptempl = P.parameters[2]
    PS = P.parameters[3]
    speckeys = PS.parameters[1]::Tuple
    pnames = _leaf_type_param_names(Ptempl)
    for pname in pnames
        pname in speckeys || continue
        ctx.idx += 1
        push!(exprs, :(nt.$group.$pname))
    end
    return nothing
end

@doc "

Flatten a nested parameter `NamedTuple` to the estimated flat vector.

`flatten(d, nt)` reads `nt` (keyed like [`params`](@ref)`(d)`, the shape
[`update`](@ref) consumes) at each ESTIMATED [`params_table`](@ref) row (an
[`uncertain`](@ref) spec's parameter) and returns those values as a `Vector`,
in table order restricted to the spec'd rows. A fixed parameter is not read. It
is the inverse of [`unflatten`](@ref): `flatten(d, unflatten(d, x)) == x`.

Shares the same compile-time layout walk `unflatten` uses (a thin generated
view over it), so the two cannot drift apart.

# Arguments
- `d`: the composed distribution whose table fixes the order.
- `nt`: a nested parameter `NamedTuple` keyed like `params(d)`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    onset_admit = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
# The estimated vector is 1-long (onset_admit.shape); round-trip it.
# Public but not exported; reach the codec by the qualified name.
nt = ComposedDistributions.unflatten(tree, [2.0])
ComposedDistributions.flatten(tree, nt)
```

# See also
- [`unflatten`](@ref): the inverse, flat vector -> nested NamedTuple.
- [`flat_dimension`](@ref): the estimated length.
"
@generated function flatten(d::T, nt::NamedTuple) where {T <:
                                                         AbstractComposedDistribution}
    ctx = _CodecCtx()
    exprs = Any[]
    _flatten_reads!(exprs, :nt, T, ctx)
    body = :(Base.vect($(exprs...)))
    return quote
        _reject_varying(d, "flatten")
        $body
    end
end

@doc "

Rebuild a composed distribution straight from its estimated flat vector.

`reconstruct(d, x)` is the flat-vector primary the per-gradient hot path
([`logdensity`](@ref), `as_turing`) routes through: it collapses `d` at the
estimated parameters in `x` (each fixed parameter held at its template value),
equivalent to `update(d, `[`unflatten`](@ref)`(d, x))` but naming the whole
operation as one verb. The nested `NamedTuple` `unflatten` produces is now
generated (concretely typed, no `Dict`), so this differentiates under every
supported AD backend including Enzyme reverse (#162).

# Arguments
- `d`: the composed distribution to rebuild.
- `x`: a flat vector of length [`flat_dimension`](@ref)`(d)`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = uncertain(Gamma(2.0, 1.0);
    shape = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
ComposedDistributions.reconstruct(tree, [3.0])
```

# See also
- [`unflatten`](@ref), [`flatten`](@ref): the flat <-> nested codec.
- [`update`](@ref): the general edit verb this composes.
"
function reconstruct(d::AbstractComposedDistribution, x::AbstractVector)
    return update(d, unflatten(d, x))
end
