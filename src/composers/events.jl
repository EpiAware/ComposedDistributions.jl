# The event-skeleton spec: an event-tree topology (names and composition
# structure) with *no* distributions attached. `@events` (in events_macro.jl)
# lowers an operator diagram to one of these, and `update(skeleton; fills...)`
# fills each named hole with a distribution and builds the concrete composed
# tree through the existing `sequential` / `parallel` / `resolve` / `compete`
# verbs. The spec is deliberately Turing- and ModifiedDistributions-free: a fill
# value is any valid leaf (a plain distribution, an `uncertain(...)` leaf, a
# ModifiedDistributions modifier leaf, or a pre-built subtree), so the fill never
# hard-codes to a distribution family. A ModifiedDistributions leaf composes
# through ModifiedDistributions' own `ModifiedDistributionsComposedDistributionsExt`
# because the composer verbs already admit any `UnivariateDistribution`.

# --- the structural spec nodes ---------------------------------------------

@doc "

The supertype of the event-skeleton spec nodes.

An [`EventSkeleton`](@ref) is a tree of these structural nodes: a named
[`Hole`](@ref) leaf, a `→`-chain, a `|`-one_of group, or a `&`-parallel group.
The nodes carry names and composition structure only, no distributions.

# See also
- [`EventSkeleton`](@ref): the skeleton wrapper the nodes sit under.
- [`@events`](@ref): the macro that lowers an operator diagram to them.
"
abstract type AbstractEventSpec end

# A named hole: the fill target. `:name` is the key `update(skeleton; ...)`
# substitutes, and (for a leaf step or branch) the event/edge name of the built
# node.
struct Hole <: AbstractEventSpec
    name::Symbol
end

# A `→` chain of steps flattened into one sequential of all operands.
struct SeqSpec{T <: Tuple} <: AbstractEventSpec
    steps::T
end

# A `|` one_of group. The fill decides whether it becomes a fixed-probability
# `Resolve` (every branch filled with a `(dist, prob)` tuple) or a racing-hazard
# `Compete` (every branch a bare distribution).
struct OneOfSpec{T <: Tuple} <: AbstractEventSpec
    branches::T
end

# A `&` parallel group of independent branches.
struct ParSpec{T <: Tuple} <: AbstractEventSpec
    branches::T
end

# --- names -----------------------------------------------------------------

# The representative name of a spec node: a `Hole`'s own name, a chain's terminal
# event (the destination the chain resolves to), and a group's deterministic
# join of its branch representatives (`_or_` for one_of, `_and_` for parallel).
# This is both the step name a node takes inside an enclosing chain / parallel
# and the outcome name a branch takes inside a one_of. A group's auto-name is
# structural, so the user references only the branch fill keys, never the group
# name.
_spec_name(h::Hole) = h.name
_spec_name(s::SeqSpec) = _spec_name(s.steps[end])
function _spec_name(o::OneOfSpec)
    return Symbol(join((string(_spec_name(b)) for b in o.branches), "_or_"))
end
function _spec_name(p::ParSpec)
    return Symbol(join((string(_spec_name(b)) for b in p.branches), "_and_"))
end

# --- hole inventory --------------------------------------------------------

# Collect every hole name in tree order. Used to validate a fill (every hole
# filled, no unknown key) and to reject a skeleton reusing a name.
_collect_holes!(acc, h::Hole) = (push!(acc, h.name); acc)
function _collect_holes!(acc, s::SeqSpec)
    foreach(x -> _collect_holes!(acc, x), s.steps)
    return acc
end
function _collect_holes!(acc, g::Union{OneOfSpec, ParSpec})
    foreach(x -> _collect_holes!(acc, x), _spec_children(g))
    return acc
end
_spec_children(g::OneOfSpec) = g.branches
_spec_children(g::ParSpec) = g.branches
_hole_names(spec::AbstractEventSpec) = _collect_holes!(Symbol[], spec)

# --- the skeleton wrapper --------------------------------------------------

@doc "

An event-tree topology: named events and their composition structure, with *no*
distributions attached yet.

An `EventSkeleton` is built by [`@events`](@ref) from a readable operator
diagram: `→` (`\\to`) chains events into a [`Sequential`](@ref), `|` branches
into a one_of outcome, `&` runs branches in [`Parallel`](@ref), and parentheses
group for precedence. A bare identifier is an event name and becomes a named
hole. Fill the holes with [`update`](@ref)`(skeleton; name = dist, ...)` to build
the concrete composed tree; whether a `|` node becomes a fixed-probability
[`Resolve`](@ref) or a racing-hazard [`Compete`](@ref) is decided there by the
fill value type.

A skeleton carries names and structure only, so one delay topology is reused
across pathogens or settings by filling it with different distributions. It is
independent of Distributions.jl families and of ModifiedDistributions: the fill
value at each hole is any valid leaf.

# Arguments
- `spec`: the root structural spec node (a [`Hole`](@ref), a `→`-chain, a
  `|`-one_of group, or a `&`-parallel group).

# Examples
```@example
using ComposedDistributions, Distributions

skeleton = @events begin
    onset → admission → (death | discharge)
end
tree = update(skeleton;
    onset = Gamma(2.0, 1.0),
    admission = LogNormal(0.5, 0.4),
    death = (Gamma(1.5, 1.0), 0.3),
    discharge = Gamma(2.0, 1.5))
event_names(tree)
```

# See also
- [`@events`](@ref): the macro that builds a skeleton from an operator diagram.
- [`update`](@ref): fill the holes to build the concrete composed tree.
- [`sequential`](@ref), [`parallel`](@ref), [`resolve`](@ref),
  [`compete`](@ref): the verbs the fill lowers to.
"
struct EventSkeleton{S <: AbstractEventSpec}
    spec::S

    function EventSkeleton(spec::S) where {S <: AbstractEventSpec}
        holes = _hole_names(spec)
        allunique(holes) || throw(ArgumentError(
            "an event skeleton reuses an event name; every hole name must be " *
            "unique, got $(holes)"))
        return new{S}(spec)
    end
end

# The hole (fill-key) names of a skeleton, in tree order.
_hole_names(skel::EventSkeleton) = _hole_names(skel.spec)

function Base.show(io::IO, skel::EventSkeleton)
    print(io, "EventSkeleton(", _render_spec(skel.spec), ")")
    return nothing
end

# Render a spec back to its operator-diagram string (for `show`).
_render_spec(h::Hole) = string(h.name)
function _render_spec(s::SeqSpec)
    return join((_render_spec(x) for x in s.steps), " → ")
end
function _render_spec(o::OneOfSpec)
    return "(" * join((_render_spec(b) for b in o.branches), " | ") * ")"
end
function _render_spec(p::ParSpec)
    return "(" * join((_render_spec(b) for b in p.branches), " & ") * ")"
end

# --- the fill: substitute holes and build the concrete tree ----------------

@doc "

Fill an [`EventSkeleton`](@ref)'s holes with distributions and build the concrete
composed tree.

`update(skeleton; name = fill, ...)` walks the skeleton, substitutes each named
hole with its fill, and lowers each structural node through the matching verb: a
`→`-chain becomes a [`Sequential`](@ref), a `&`-group a [`Parallel`](@ref), and a
`|`-group a fixed-probability [`Resolve`](@ref) or a racing-hazard
[`Compete`](@ref) decided by the fill value type.

A fill value is any valid leaf and is never coerced to a distribution family: a
plain `UnivariateDistribution`, an [`uncertain`](@ref) / `@uncertain` leaf, a
ModifiedDistributions modifier leaf (`affine` / `weighted` / `thin` / censored),
or a pre-built composed subtree. A ModifiedDistributions leaf composes through
the existing extension because the verbs admit any `UnivariateDistribution`.

The one_of (`|`) rule, decided at fill time so `|` stays one syntax:

- every branch filled with a `(dist, prob)` tuple builds a [`Resolve`](@ref);
- every branch filled with a bare distribution builds a [`Compete`](@ref);
- the last branch alone may omit its probability, taking the residual
  `1 - sum(of the others)` (a [`Resolve`](@ref));
- any other mix of `(dist, prob)` and bare branches is an error.

The fill is validated: every hole must be filled (an unfilled hole is named in
the error) and every fill key must name a hole (an unknown key is rejected). A
group's auto-name is structural, so a fill names only the branch holes, never the
group.

# Arguments
- `skeleton`: the [`EventSkeleton`](@ref) whose holes to fill.

# Keyword Arguments
- `fills...`: one `name = fill` per hole, the `name` a hole key and the `fill`
  any valid leaf (or a `(dist, prob)` tuple for a fixed-probability one_of
  branch).

# Examples
```@example
using ComposedDistributions, Distributions

skeleton = @events begin
    onset → admission → (death | discharge)
end
tree = update(skeleton;
    onset = Gamma(2.0, 1.0),
    admission = LogNormal(0.5, 0.4),
    death = (Gamma(1.5, 1.0), 0.3),
    discharge = Gamma(2.0, 1.5))
rand(tree)
```

# See also
- [`@events`](@ref): the macro that builds the skeleton.
- [`EventSkeleton`](@ref): the topology type.
"
function update(skel::EventSkeleton; fills...)
    fill_nt = values(fills)
    holes = _hole_names(skel)
    for h in holes
        haskey(fill_nt, h) || throw(ArgumentError(
            "event skeleton hole `$h` was not filled; pass `$h = <dist>` " *
            "(holes: $(holes))"))
    end
    for k in keys(fill_nt)
        k in holes || throw(ArgumentError(
            "unknown fill key `$k`; the skeleton's holes are $(holes)"))
    end
    return _fill(skel.spec, fill_nt)
end

# A hole substitutes its fill value directly (any valid leaf).
_fill(h::Hole, fills::NamedTuple) = fills[h.name]

# A `→` chain builds a `Sequential` of the filled steps, named by each step's
# representative name.
function _fill(s::SeqSpec, fills::NamedTuple)
    comps = map(x -> _fill(x, fills), s.steps)
    names = map(_spec_name, s.steps)
    return Sequential(comps, names)
end

# A `&` group builds a `Parallel` of the filled branches, named likewise.
function _fill(p::ParSpec, fills::NamedTuple)
    comps = map(x -> _fill(x, fills), p.branches)
    names = map(_spec_name, p.branches)
    return Parallel(comps, names)
end

# A `|` group builds a `Resolve` or `Compete` per the fill value types. A branch
# is filled by its hole (a leaf fill) or, for a nested group branch, by the
# subtree it builds (a bare, non-terminal outcome).
function _fill(o::OneOfSpec, fills::NamedTuple)
    names = map(_spec_name, o.branches)
    vals = map(b -> _branch_fill(b, fills), o.branches)
    outcomes = map(=>, names, vals)
    proby = map(_is_prob_fill, vals)
    if all(proby) || (all(Base.front(proby)) && !last(proby))
        # Every branch a `(dist, prob)` tuple, or every branch but the last (the
        # last taking the residual): a fixed-probability Resolve.
        return resolve(outcomes...)
    elseif !any(proby)
        # Every branch a bare distribution: a racing-hazard Compete.
        return compete(outcomes...)
    end
    throw(ArgumentError(
        "a `|` node fill mixes fixed-probability `(dist, prob)` outcomes with " *
        "bare-distribution outcomes; fill every branch with a `(dist, prob)` " *
        "tuple for a fixed-probability Resolve, or every branch with a bare " *
        "distribution for a racing-hazard Compete (the last branch alone may " *
        "omit its probability, taking the residual). Branches: $(collect(names))"))
end

_branch_fill(h::Hole, fills::NamedTuple) = fills[h.name]
_branch_fill(g::AbstractEventSpec, fills::NamedTuple) = _fill(g, fills)

# A one_of branch fill is a fixed-probability outcome when it is a
# `(dist, prob)` tuple (a two-tuple whose second element is a real probability),
# mirroring the `Resolve` payload shape. Anything else (a bare distribution, an
# uncertain leaf, a modifier leaf, a built subtree) is a bare hazard branch.
_is_prob_fill(::Tuple{Any, <:Real}) = true
_is_prob_fill(::Any) = false
