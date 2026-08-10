# Generation-time (type-domain) flat-vector <-> nested-NamedTuple codec.
#
# Since #178 PR 1, every composer/wrapper carries its layout-affecting
# structure (names, tags, groups) as type parameters, not runtime fields. That
# means the whole flat-vector layout -- which leaf owns which slot, the
# shared-tag/pool-group dedup with root-lift, the Resolve stick-breaking
# count -- is a function of `typeof(d)` alone. This file walks that type once
# per distinct concrete tree shape (inside `@generated` function bodies) and
# emits code with the slot indices baked in as literals, replacing the old
# runtime `Dict{Symbol, Any}` walk (the `_ParamSink` walk + `_nest_insert!` +
# `_freeze_tree`) that `unflatten` used to re-run on every call. That
# Dict/`Any`-typed walk is #162's root cause (Enzyme's type analysis cannot
# see through a `Dict{Symbol, Any}`/heap-boxed reconstruction); the generated
# walk below produces a concretely-typed (`@inferred`-stable) nested
# NamedTuple instead, which both fixes Enzyme reverse and removes the
# dominant per-evaluation cost (issue #178's spike measured the Dict walk at
# 50-83% of a `logdensity` call).
#
# `update(d, nt)` (ordinary recursion over the nested NamedTuple, adding no
# Dict of its own) is unchanged by this file: `reconstruct` below composes the
# new generated `unflatten` with the existing `update`, rather than
# re-deriving `update`'s full leaf-rebuild logic (merge mode, pooled
# reconstruction, stick-breaking, Choose namespacing, extras) a second time.
# That keeps `_update_leaf`/`_uncertain_leaf`/`_reconstruct_pooled_leaf` the
# single source of truth for "how a leaf rebuilds from values" (the #174
# leaf-protocol hooks), exactly as the design review required.

# --- shared guards -----------------------------------------------------------
#
# Used by the generated `unflatten`/`flatten`/`flat_dimension` below.

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
# rest of the package's dimension guards are: see `nesting.jl`'s matching
# note. `est` is any object whose `length` is the estimated-row count (a
# `UnitRange` here, since the generated caller has only the count, not the
# table).
@noinline function _throw_unflatten_dimmismatch(x, est, d)
    throw(DimensionMismatch(
        "flat vector has length $(length(x)) but $d has " *
        "$(length(est)) estimated parameters"))
end

# Read the value at `(path..., param)` of a nested NamedTuple. Still used by
# `Pool.jl`'s `pool_centred_logprior`, which reads a centred pooled member's
# latent straight off the `unflatten`ed `nt` at a runtime-known path (a
# per-evaluation but not per-slot walk, so a plain function is fine here).
function _read_path(nt::NamedTuple, path::Tuple, param::Symbol)
    node = nt
    for k in path
        node = getproperty(node, k)
    end
    return getproperty(node, param)
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
    elseif T <: AbstractComposedDistribution
        return _generic_node_unflatten_expr(access, T, ctx)
    else
        return _leaf_unflatten_expr(access, T, ctx)
    end
end

# Any OTHER composer node (a downstream type, not one of the five built-ins
# above): read its (names, child types) layout purely from its own type
# parameters -- see `_generic_node_layout`'s docstring for why this must never
# call a method the generator cannot see -- then recurse exactly like
# `_composer_unflatten_expr` does, reading each child at RUNTIME through the
# public `node_children` accessor (a plain call embedded in the returned code,
# not evaluated by the generator itself, so it carries no world-age risk; the
# composite-leaf case above already relies on this same pattern). Landing in
# this branch (rather than falling through to the leaf branch below and
# silently reporting zero estimated parameters) is what #374 closes.
function _generic_node_unflatten_expr(access, ::Type{T}, ctx::_CodecCtx) where {T}
    names, ctypes = _generic_node_layout(T)
    keys_out = Symbol[]
    vals_out = Any[]
    for i in eachindex(names)
        child_access = :(ComposedDistributions.node_children($access)[$i])
        e = _unflatten_expr(child_access, ctypes[i], ctx)
        e === nothing && continue
        push!(keys_out, names[i])
        push!(vals_out, e)
    end
    return :(NamedTuple{$(Tuple(keys_out))}(($(vals_out...),)))
end

# The (names, child types) layout of a downstream composer node, read purely
# from its own type parameters -- no method call, so reading it cannot hit the
# world-age wall a `@generated` function calling a downstream-extensible
# method would (measured: a `@generated` function cannot see a method a
# downstream package defines, even after both packages are fully loaded and
# precompiled -- see `docs/src/developer/interface-contracts.md`). By
# convention, a node wanting codec (`flat_dimension`/`flatten`/`unflatten`/
# `reconstruct`) support declares its own child names (a `Tuple` of `Symbol`,
# matching `component_names`) and its children's types (a `Tuple` type,
# matching `node_children`'s return type) as its first two type parameters,
# in that order -- exactly the shape `Sequential`/`Parallel`/`Choose`/
# `Compete` already use. Throws a clear, actionable error naming the type
# when a node subtyping `AbstractComposedDistribution` does not have that
# shape, rather than silently falling through to the leaf branch (#374's
# root cause).
function _generic_node_layout(::Type{T}) where {T}
    P = T.parameters
    if length(P) < 2 || !(P[1] isa Tuple) || !all(n -> n isa Symbol, P[1]) ||
       !(P[2] isa Type) || !(P[2] <: Tuple)
        throw(ArgumentError(
            "$T subtypes AbstractComposedDistribution but does not expose " *
            "a (names::Tuple{Vararg{Symbol}}, children::Tuple-type) layout " *
            "as its first two type parameters, so the flat-vector codec " *
            "cannot read its layout at compile time. Give it that shape " *
            "(see the \"Writing a new composer node\" developer docs " *
            "section on node_children/node_rebuild), or avoid calling " *
            "flat_dimension/flatten/unflatten/reconstruct on a tree " *
            "containing it."))
    end
    names = P[1]::Tuple
    ctypes = P[2].parameters
    length(names) == length(ctypes) || throw(ArgumentError(
        "$T declares $(length(names)) names but $(length(ctypes)) child " *
        "types; node_children(::$(nameof(T))) must return one child per " *
        "name, matching component_names(::$(nameof(T)))"))
    return names, ctypes
end

# The component types of a see-through composite leaf (`Convolved`/
# `Difference` used as a leaf, `convolved_interop.jl`), mirroring
# `node_children` at the type level: `Convolved{C<:Tuple, Method}`'s
# components are `C`'s own type parameters; `Difference{X, Y, Method}` has
# exactly the two fixed operand types.
_composite_child_types(::Type{<:Convolved{C}}) where {C} = Tuple(C.parameters)
_composite_child_types(::Type{<:Difference{X, Y}}) where {X, Y} = (X, Y)

# A composite leaf's node children are namespaced `component_1, component_2,
# ...` (mirroring `_composite_component_names` in `convolved_interop.jl`,
# computed independently here so this file has no include-order dependency on
# it) and read at runtime through the generic `node_children` accessor (so
# `Convolved`'s `.components` tuple and `Difference`'s `(.x, .y)` pair share
# one code path, exactly as `_walk_rows!`/`_update` already do).
function _composite_unflatten_expr(access, ::Type{T}, ctx::_CodecCtx) where {T}
    ctypes = _composite_child_types(T)
    keys_out = Symbol[]
    vals_out = Any[]
    for i in eachindex(ctypes)
        child_access = :(ComposedDistributions.node_children($access)[$i])
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
# `(native..., extra...)` NamedTuple entry by calling the runtime seam
# (`_leaf_entry`, introspection.jl, S1) with the estimated slot values --
# fixed parameters are read from the current instance inside `_leaf_entry`
# itself (via `leaf_param_values`, entirely generic hooks), so a leaf-wrapper
# extension whose `params` is not its own fields 1:1 needs no type-level
# override for its *fixed* values to come out right (#189's motivating case;
# see `ext_demo`-derived regression test in codec_gen.jl).
#
# `_leaf_entry`'s own substitution contract (introspection.jl) consumes
# `slots` positionally IN `leaf_param_names(leaf)` ORDER, not `speckeys`'s own
# kwargs order (a user can write `uncertain(Gamma(2, 1); scale = ..., shape =
# ...)`, scale first, while Gamma's native order is `(shape, scale)`). Since
# S3 removed the type-level table this generator once used to compute that
# order at generation time, this walk cannot bake per-name literal `x`
# indices in `leaf_param_names` order any more -- so it does not try to:
# `slot_exprs` below is simply the RAW, CONSECUTIVE `k = length(speckeys)`
# positions this leaf occupies in `x`, in NO particular per-name order, and
# `_leaf_entry` resolves the name<->slot correspondence itself at RUNTIME
# (calling `leaf_param_names(leaf)` on the actual instance it is handed,
# ordinary dispatch, no world-age concern since this runs from the returned
# CODE, not the generator). `flatten`'s `_leaf_flatten_reads!` reads the same
# `k`-length block back out in the same runtime-resolved order via
# `_leaf_flatten_values`, so the two stay in lockstep without either knowing
# the order at generation time.
#
# A `Pool` spec needs more than that, though: `_walk_rows!`/`_pool_rows!`
# (introspection.jl/Pool.jl) insert a first-seen group's hyperparameter rows
# at the POOLED PARAMETER'S OWN native-order position within this leaf's row
# sequence -- not hoisted before this leaf's other params (which native-order
# ignorance would otherwise force). `speckeys`/`specvaltypes` are still walked
# directly here to decide WHICH groups this leaf-visit must materialise
# (`ctx.seen_groups` dedup, order-independent) and WHICH spec'd names are
# `Pool`-noncentred (order-independent, by name), and to size the `x` block
# (`_pool_hyper_count`, a population's spec'd-parameter COUNT, not its order --
# generation-time-safe). The actual per-name interleaving -- this leaf's own
# `leaf_param_names` order AND each materialising population's own native
# order -- is resolved from the real instances at RUNTIME, exactly
# `_leaf_entry`'s trick, by `_leaf_entry_grouped` (Pool.jl), so a leaf
# touching no first-seen group keeps using the plain `_leaf_entry` seam above
# unchanged.
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
        S = L.parameters[3]
        speckeys = S.parameters[1]::Tuple
        specvaltypes = Tuple(S.parameters[2].parameters)
    else
        speckeys = ()
        specvaltypes = ()
    end
    pool_names, materialize, materialize_groups, extra_slots = _leaf_pool_layout!(
        ctx, speckeys, specvaltypes)
    slot_exprs = Any[]
    for _ in 1:(length(speckeys) + extra_slots)
        ctx.idx += 1
        push!(slot_exprs, :(x[$(ctx.idx)]))
    end

    if isempty(materialize)
        entry_expr = :(ComposedDistributions._leaf_entry(
            $access, Val($speckeys), ($(slot_exprs...),)))
        isempty(pool_names) && return entry_expr
        return :(ComposedDistributions._wrap_pool_entries(
            $entry_expr, Val($(Tuple(pool_names)))))
    end

    call_expr = :(ComposedDistributions._leaf_entry_grouped(
        $access, Val($speckeys), Val($(Tuple(pool_names))),
        Val($(Tuple(materialize))), ($(slot_exprs...),)))
    for group in materialize_groups
        push!(ctx.group_keys, group)
        push!(ctx.group_vals, :(last($call_expr).$group))
    end
    return :(first($call_expr))
end

# Shared by `_leaf_unflatten_expr`/`_leaf_flatten_reads!`: walks a leaf's
# `speckeys`/`specvaltypes` once to decide (1) `pool_names`, the spec'd names
# that are `Pool`-noncentred (order-independent, by name -- unchanged from
# before this function existed), (2) `materialize`/`materialize_groups`, the
# subset of `speckeys` (and, paired 1:1, their group names) whose pool group
# is first-seen ACROSS THE WHOLE TREE (`ctx.seen_groups` dedup, mutated here
# so a later leaf naming the same group sees it already registered -- kept as
# a SEPARATE parallel vector, not re-derived from `speckeys`/`specvaltypes`
# with `findfirst`, since a `findfirst` result JET cannot prove non-`nothing`
# would poison the caller's indexing), and (3) `extra_slots`, the total
# `x`-slot count every materialising group's population contributes
# (`_pool_hyper_count`, a count only -- see `_leaf_unflatten_expr`'s comment
# for why no order is needed here). `unflatten` and `flatten` call this
# identically so `ctx.seen_groups` advances in lockstep between the two
# generated walks.
function _leaf_pool_layout!(
        ctx::_CodecCtx, speckeys::Tuple, specvaltypes::Tuple)
    pool_names = Symbol[]
    materialize = Symbol[]
    materialize_groups = Symbol[]
    extra_slots = 0
    for (pname, specT) in zip(speckeys, specvaltypes)
        specT <: Pool || continue
        group = specT.parameters[1]::Symbol
        noncentred = specT.parameters[2]::Bool
        if !(group in ctx.seen_groups)
            push!(ctx.seen_groups, group)
            push!(materialize, pname)
            push!(materialize_groups, group)
            extra_slots += _pool_hyper_count(specT)
        end
        noncentred && push!(pool_names, pname)
    end
    return pool_names, materialize, materialize_groups, extra_slots
end

# Merge the root node's own NamedTuple with the root-lifted tag/group entries
# (each keyed by tag/group name), matching the old Dict walk's flat top-level
# namespace. `validate_tree_names` (called once at
# `distribution_to_logdensity` construction, not here) is what actually
# guards against a name collision across the three namespaces; a collision
# here just silently prefers the later `merge` argument, exactly as the old
# `Dict` insert did.
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

Generated once per distinct tree type from a compile-time layout walk (no
`Dict`, no intermediate `Any`-typed accumulation), so the result is
`@inferred`-concrete and the reverse-mode AD backends (including Enzyme)
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

`flat_dimension(d)` is the number of scalar estimated parameters: the count of
[`uncertain`](@ref) specs across the tree, i.e. the [`composed_to_table`](@ref)
`:param` rows whose `prior` column carries a spec. A fixed (non-uncertain) leaf
contributes nothing, so a tree with no uncertain leaves has flat dimension 0.
It is the
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
# Shares the same generation-time walk/dedup as `unflatten` (called on `T`,
# not `NT`: the tree type alone fixes which slots are estimated and their
# order), but instead of building NamedTuple-construction expressions it
# builds NamedTuple-read expressions against the `nt` argument, appended in
# estimated-slot order. Not on the per-gradient hot path (only `unflatten`/
# `reconstruct` are), so the read is a plain generated view rather than a
# further-optimised primitive.
#
# A root-lifted entry (a `Shared` tag or a `Pool` group's hyperparameters) is
# read directly off the literal `nt` argument (`:(nt.$tag)`/`:(nt.$group...)`),
# never off the locally-threaded `nt_access`: that is exactly where
# `unflatten` places it, regardless of how deep the tagged/pooled leaf sits
# structurally. Since the generated function's argument is always named `nt`,
# that root reference needs no extra bookkeeping to thread through the walk.
#
# `d_access` is threaded alongside `nt_access`, mirroring `_unflatten_expr`'s
# own `access` parameter exactly (same child-access expressions at every
# composer/resolve/composite level): since S3 removed the type-level name
# table, the leaf case needs a reachable instance of the ACTUAL leaf to call
# the instance-level `leaf_param_names` on at runtime (`_leaf_flatten_values`,
# introspection.jl) -- `flatten(d::T, nt::NamedTuple)` already receives `d`,
# this walk just needed to carry an access path into it down to each leaf.

function _flatten_reads!(exprs::Vector, d_access, nt_access, ::Type{T},
        ctx::_CodecCtx) where {T}
    if T <: Sequential || T <: Parallel
        _composer_flatten_reads!(exprs, d_access, :components, nt_access, T, ctx)
    elseif T <: Choose
        _composer_flatten_reads!(exprs, d_access, :alternatives, nt_access, T, ctx)
    elseif T <: Resolve
        _resolve_flatten_reads!(exprs, d_access, nt_access, T, ctx)
    elseif T <: Compete
        _composer_flatten_reads!(exprs, d_access, :delays, nt_access, T, ctx)
    elseif T <: Union{Convolved, Difference}
        _composite_flatten_reads!(exprs, d_access, nt_access, T, ctx)
    elseif T <: AbstractComposedDistribution
        _generic_node_flatten_reads!(exprs, d_access, nt_access, T, ctx)
    else
        _leaf_flatten_reads!(exprs, d_access, nt_access, T, ctx)
    end
    return nothing
end

# The read-direction counterpart of `_generic_node_unflatten_expr`: same
# type-parameter layout read, same `node_children` runtime access, appending
# NamedTuple-read expressions instead of building NamedTuple-construction
# ones.
function _generic_node_flatten_reads!(exprs::Vector, d_access, nt_access,
        ::Type{T}, ctx::_CodecCtx) where {T}
    names, ctypes = _generic_node_layout(T)
    for i in eachindex(names)
        _flatten_reads!(exprs,
            :(ComposedDistributions.node_children($d_access)[$i]),
            :($nt_access.$(names[i])), ctypes[i], ctx)
    end
    return nothing
end

function _composite_flatten_reads!(
        exprs::Vector, d_access, nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    ctypes = _composite_child_types(T)
    for i in eachindex(ctypes)
        child_access = :(ComposedDistributions.node_children($d_access)[$i])
        _flatten_reads!(exprs, child_access,
            :($nt_access.$(Symbol(:component_, i))), ctypes[i], ctx)
    end
    return nothing
end

function _composer_flatten_reads!(exprs::Vector, d_access, field::Symbol,
        nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    C = T.parameters[2]
    ctypes = C.parameters
    for i in eachindex(names)
        _flatten_reads!(exprs, :($d_access.$field[$i]),
            :($nt_access.$(names[i])), ctypes[i], ctx)
    end
    return nothing
end

function _resolve_flatten_reads!(
        exprs::Vector, d_access, nt_access, ::Type{T}, ctx::_CodecCtx) where {T}
    names = T.parameters[1]::Tuple
    D = T.parameters[2]
    S = T.parameters[4]
    dtypes = D.parameters
    for i in eachindex(names)
        dtypes[i] <: NoEvent && continue
        _flatten_reads!(exprs, :($d_access.delays[$i]),
            :($nt_access.$(names[i])), dtypes[i], ctx)
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

# Leaf case: a naive `child_access`/`d_access` is passed in even for a
# `Shared` child (built by the caller as if it owned a positional key); when
# `L <: Shared` those accesses are simply never used below -- a first
# occurrence reads off the literal root `nt.$tag` instead (peeling `d_access`
# to `.dist` for the recursion, mirroring `_leaf_unflatten_expr`'s own
# `inner_access`), and a later occurrence returns immediately without emitting
# anything, matching `unflatten`'s suppression exactly.
#
# Routed through the runtime seam (`_leaf_flatten_values`, introspection.jl),
# the read-direction counterpart of `_leaf_unflatten_expr`'s `_leaf_entry`
# call: it calls `leaf_param_names($d_access)` at RUNTIME (ordinary dispatch
# on the actual leaf instance, not the generator, so no world-age concern) to
# resolve the same name<->slot correspondence `unflatten`'s `_leaf_entry` used,
# and unwraps a `Pool`-noncentred slot's `z` field the same way
# `_wrap_pool_entries` wrapped it.
#
# A leaf that materialises a pool group (mirroring `_leaf_unflatten_expr`'s
# `_leaf_entry_grouped` branch exactly, including reusing `_leaf_pool_layout!`
# so `ctx.seen_groups` advances identically to `unflatten`'s walk) is instead
# routed through `_leaf_flatten_grouped` (Pool.jl), the read-direction
# counterpart: it reads each materialising group's hyper `NamedTuple` off the
# literal root `nt.<group>` (exactly where `unflatten` root-lifts it) and
# interleaves those values with this leaf's own, in `leaf_param_names`
# native order, so the flat sequence this leaf-visit contributes lines up
# `_leaf_entry_grouped`'s consumption order value-for-value.
function _leaf_flatten_reads!(exprs::Vector, d_access, nt_access, ::Type{L},
        ctx::_CodecCtx) where {L}
    if L <: Shared
        tag = L.parameters[1]::Symbol
        D = L.parameters[2]
        tag in ctx.seen_tags && return nothing
        push!(ctx.seen_tags, tag)
        _flatten_reads!(exprs, :($d_access.dist), :(nt.$tag), D, ctx)
        return nothing
    end

    if L <: Uncertain
        S = L.parameters[3]
        speckeys = S.parameters[1]::Tuple
        specvaltypes = Tuple(S.parameters[2].parameters)
    else
        speckeys = ()
        specvaltypes = ()
    end
    isempty(speckeys) && return nothing
    pool_names, materialize, _, extra_slots = _leaf_pool_layout!(
        ctx, speckeys, specvaltypes)
    ctx.idx += length(speckeys) + extra_slots

    if isempty(materialize)
        push!(exprs,
            :(ComposedDistributions._leaf_flatten_values(
                $d_access, Val($speckeys), Val($(Tuple(pool_names))),
                $nt_access)...))
    else
        push!(exprs,
            :(ComposedDistributions._leaf_flatten_grouped(
                $d_access, Val($speckeys), Val($(Tuple(pool_names))),
                Val($(Tuple(materialize))), $nt_access, nt)...))
    end
    return nothing
end

@doc "

Flatten a nested parameter `NamedTuple` to the estimated flat vector.

`flatten(d, nt)` reads `nt` (keyed like [`params`](@ref)`(d)`, the shape
[`update`](@ref) consumes) at each estimated [`composed_to_table`](@ref)
`:param` row (an [`uncertain`](@ref) spec's parameter) and returns those
values as a `Vector`, in table order restricted to the spec'd rows. A fixed
parameter is not read. It is the inverse of [`unflatten`](@ref): `flatten(d,
unflatten(d, x)) == x`.

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
    _flatten_reads!(exprs, :d, :nt, T, ctx)
    body = :(Base.vect($(exprs...)))
    return quote
        _reject_varying(d, "flatten")
        $body
    end
end

@doc "

Rebuild a composed distribution straight from its estimated flat vector.

`reconstruct(d, x)` collapses `d` at the estimated parameters in `x`, holding
each fixed parameter at its template value. It is
`update(d, `[`unflatten`](@ref)`(d, x))` named as one verb, and is the
flat-vector primary a per-gradient hot path routes through
(DistributionsInference.jl's `distribution_to_logdensity` and
`distribution_to_turing`).

Being that composition rather than a generated function of its own,
`reconstruct` is not independently shown `@inferred`-concrete; the guarantee
is [`unflatten`](@ref)'s, and `update`'s inferrability is inherited from it.

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
