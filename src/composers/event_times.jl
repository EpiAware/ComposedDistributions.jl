# ============================================================================
# event_times / event_increments: per-step increments <-> absolute positions
# ============================================================================
#
# A draw from a composed tree (`rand`) is a record of per-step INCREMENTS,
# keyed by the per-value leaf names (`_value_names`, the same keys `logpdf`
# consumes). `event_times` converts such a record into ABSOLUTE positions
# measured from the tree's origin; `event_increments` is the exact inverse.
#
# The layout is the one the value-name walk already derives, so a caller never
# reconstructs the topology by hand: a `Sequential` threads a running position
# through its steps (chain accumulation), a `Parallel`'s branches each anchor on
# their shared parent position (the origin), and a chain continuing past a
# nested `Parallel` resumes from that parent, matching `_nested_terminal_name`.
# A `Resolve`/`Compete`/`Choose` is a single value slot (the fired outcome), so
# it is one increment from its parent like any leaf. Positions are unitless
# distance from the origin; calendar mapping belongs to the model layer (#269).

# Promote the record's value types to one element type (AD `Dual`s flow too).
_record_valtype(rec::NamedTuple) = promote_type(map(typeof, values(rec))...)

# Accumulate absolute positions. `_accum_time!` pushes each leaf's absolute
# position into `vals` in `_value_names` (depth-first) order and returns the
# subtree's terminal position; `_deaccum_time!` is the inverse, pushing each
# leaf's increment (absolute minus its parent position).

function _accum_time!(vals, rec, base, path, child::Union{Sequential, Parallel})
    cn = component_names(child)
    running = base
    for i in eachindex(child.components)
        pos = _accum_time!(vals, rec, running, (path..., cn[i]),
            child.components[i])
        # A chain threads the running position; parallel branches all share
        # `base`, and the node's terminal is `base` (resume-from-origin).
        child isa Sequential && (running = pos)
    end
    return child isa Sequential ? running : base
end
function _accum_time!(vals, rec, base, path, ::Any)
    name = _join_value_path(path)
    t = base + rec[name]
    push!(vals, t)
    return t
end

function _deaccum_time!(vals, rec, base, path,
        child::Union{Sequential, Parallel})
    cn = component_names(child)
    running = base
    for i in eachindex(child.components)
        pos = _deaccum_time!(vals, rec, running, (path..., cn[i]),
            child.components[i])
        child isa Sequential && (running = pos)
    end
    return child isa Sequential ? running : base
end
function _deaccum_time!(vals, rec, base, path, ::Any)
    name = _join_value_path(path)
    t = rec[name]
    push!(vals, t - base)
    return t
end

@doc "
Convert a drawn record of per-step increments into absolute positions measured
from the composed tree's origin.

A draw from `d` (`rand(d)`) records each event as an increment from its
predecessor. `event_times` accumulates these into absolute positions: chain
steps sum along the chain, parallel branches each measure from the shared
origin, and a resolved/racing node reports the position of the outcome that
fired. The result is keyed exactly like the input record and stays in unitless
distance from the origin. [`event_increments`](@ref) is the inverse.

Pass a `Vector` of records for a batch; each row is transformed independently.

# Arguments
- `d`: the composed tree (a `Sequential` or `Parallel`) the record was drawn
  from; a bare leaf or one_of node errors, having no per-step chain.
- `record`: a single drawn record (`NamedTuple`) of per-step increments, or a
  `Vector` of such records for a batch.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((
    path = sequential(:step_a => LogNormal(0.5, 0.4),
        :step_b => Gamma(2.0, 1.0)),
    side = Gamma(1.5, 1.0)))
event_times(tree, (path_step_a = 1.0, path_step_b = 2.0, side = 3.0))
```

# See also
- [`event_increments`](@ref): the inverse transform.
- [`event_names`](@ref): the flat event layout this reuses.
"
function event_times(d::Union{Sequential, Parallel}, rec::NamedTuple)
    vals = _record_valtype(rec)[]
    _accum_time!(vals, rec, zero(_record_valtype(rec)), (), d)
    return NamedTuple{_value_names(d)}(Tuple(vals))
end
function event_times(d::Union{Sequential, Parallel}, table::AbstractVector)
    [event_times(d, row) for row in table]
end
function event_times(d::UnivariateDistribution, ::Any)
    _throw_bare_record("event_times", d)
end

@doc "
Invert [`event_times`](@ref): convert a record of absolute positions back into
the per-step increments the scorer (`logpdf`) consumes.

`event_increments(d, event_times(d, record)) == record`. Pass a `Vector` of
records for a batch.

# Arguments
- `d`: the composed tree (a `Sequential` or `Parallel`) the record was drawn
  from; a bare leaf or one_of node errors, having no per-step chain.
- `record`: a single record (`NamedTuple`) of absolute positions, or a `Vector`
  of such records for a batch.

# Examples
```@example
using ComposedDistributions, Distributions

tree = sequential(:a => LogNormal(0.5, 0.4), :b => Gamma(2.0, 1.0))
event_increments(tree, (a = 1.0, b = 3.0))
```

# See also
- [`event_times`](@ref): the forward transform.
"
function event_increments(d::Union{Sequential, Parallel}, rec::NamedTuple)
    vals = _record_valtype(rec)[]
    _deaccum_time!(vals, rec, zero(_record_valtype(rec)), (), d)
    return NamedTuple{_value_names(d)}(Tuple(vals))
end
function event_increments(d::Union{Sequential, Parallel}, table::AbstractVector)
    [event_increments(d, row) for row in table]
end
function event_increments(d::UnivariateDistribution, ::Any)
    _throw_bare_record("event_increments", d)
end

# A bare leaf / one_of at the top level has no per-step chain to accumulate;
# these transforms need a composed tree. A clear error beats a MethodError.
@noinline function _throw_bare_record(fn, d)
    throw(ArgumentError(
        "$(fn) needs a composed tree (a Sequential or Parallel); a bare " *
        "$(nameof(typeof(d))) leaf/one_of node has no per-step record to " *
        "transform"))
end
