# ============================================================================
# Batch record sampling and scoring for the multi-child composers
# ============================================================================
#
# `rand(d, n)` draws `n` independent labelled records and stacks them into a
# Tables.jl COLUMN table (a `NamedTuple` of vectors), one column per emitted
# leaf value, keyed by the VALUE-name layout `_value_names(d)` — the same layout
# a single `rand(d)` draw and `logpdf(d, ::NamedTuple)` already use. So the
# simulate/score round trip `logpdf(d, rand(d, n))` holds by construction.
#
# The column layout is per-composer, matching each composer's own single-record
# convention: `Sequential`/`Parallel` records are value-name keyed, a standalone
# one_of record is event-name keyed (`rand(::AbstractOneOf, n)` in Resolve.jl).
# That mirrors their single-record conventions differing, not a new split. Reach
# the event-name view of the schema through [`event_names`](@ref) /
# [`event_tree`](@ref); convert a drawn record's per-step increments to absolute
# positions with [`event_times`](@ref).

# Draw `n` independent labelled records into a Tables.jl column table. The
# `rng`-less form threads the default RNG, matching `rand(::Distribution, n)`.
function Base.rand(rng::AbstractRNG, d::Union{Sequential, Parallel}, n::Int)
    return Tables.columntable([_named_composer_rand(rng, d) for _ in 1:n])
end
function Base.rand(d::Union{Sequential, Parallel}, n::Int)
    return rand(default_rng(), d, n)
end

@doc "

Log density of a batch of labelled records, summed over the batch.

A `Vector` of `NamedTuple` records (e.g. `[rand(d) for _ in 1:n]`) scores each
record through the single-record [`logpdf`](@ref)`(d, ::NamedTuple)` value-name
path and sums. The `AbstractVector{<:NamedTuple}` element type is disjoint from
the flat single-record `logpdf(d, ::AbstractVector{<:Real})` method, so a
scalar-valued record vector is never mistaken for a batch (and vice versa). The
concrete `Sequential`/`Parallel` methods (rather than a `Union`) keep this
strictly more specific than the flat per-type methods, so dispatch is
unambiguous. A column table (as [`rand`](@ref)`(d, n)` returns) is scored by the
`NamedTuple` method's table branch.

See also: [`rand`](@ref), [`event_names`](@ref)
"
function logpdf(d::Sequential, x::AbstractVector{<:NamedTuple})
    return sum(logpdf(d, r) for r in x)
end
function logpdf(d::Parallel, x::AbstractVector{<:NamedTuple})
    return sum(logpdf(d, r) for r in x)
end
