# The `@events` macro: a syntax front-end that lowers an operator-diagram of
# event names to an `EventSkeleton` (events.jl). It rewrites syntax only. The
# body is walked as pure `Expr` and a constructor call is emitted, with every
# structural type interpolated as its own object (so the emitted code is
# hygienic and needs no name resolution in the caller's scope). No distribution
# ever appears in a skeleton; the fill happens later in `update(skeleton; ...)`.
#
# Operator mapping (parentheses group for precedence):
#   →  (\to)  chains events into a `Sequential`; a nested `→` chain flattens
#             into one sequential of all operands.
#   |         branches into a one_of outcome (a `Resolve`/`Compete` placeholder
#             the fill resolves).
#   &         runs branches in `Parallel`.
# A bare identifier is an event name and becomes a named hole (the fill key).

# Extract the single event-diagram expression from the macro body. A `begin ...
# end` block must hold exactly one expression (line-number nodes aside); a bare
# expression is used directly.
function _events_body_expr(body)
    if body isa Expr && body.head === :block
        stmts = filter(a -> !(a isa LineNumberNode), body.args)
        length(stmts) == 1 || error(
            "@events: the block must contain exactly one event-diagram " *
            "expression; got $(length(stmts))")
        return stmts[1]
    end
    return body
end

# Flatten a chain of one infix operator `op` into its operand expressions,
# splicing nested same-operator calls (so `a → b → c`, whatever the parse
# associativity, yields the flat operand list `[a, b, c]`). A non-`op` node
# stops the flatten and is returned as a single operand (lowered separately,
# possibly recursing into a different operator).
function _events_flatten(ex, op::Symbol)
    if ex isa Expr && ex.head === :call && !isempty(ex.args) &&
       ex.args[1] === op
        return reduce(vcat,
            (_events_flatten(a, op) for a in ex.args[2:end]); init = Any[])
    end
    return Any[ex]
end

# Lower one diagram expression to an `Expr` that builds its spec node. A bare
# Symbol builds a `Hole`; a `→` / `|` / `&` call builds a `SeqSpec` / `OneOfSpec`
# / `ParSpec` over its flattened, recursively lowered operands. The structural
# types are interpolated as objects, so the emitted call resolves them without
# reference to the caller's scope.
function _events_lower(ex)
    if ex isa Symbol
        return Expr(:call, Hole, QuoteNode(ex))
    elseif ex isa Expr && ex.head === :call && !isempty(ex.args)
        op = ex.args[1]
        if op === :→
            return _events_group(SeqSpec, _events_flatten(ex, :→))
        elseif op === :|
            return _events_group(OneOfSpec, _events_flatten(ex, :|))
        elseif op === :&
            return _events_group(ParSpec, _events_flatten(ex, :&))
        end
    end
    return _events_bad(ex)
end

# Build the `T((lowered_children...,))` constructor call for a group node.
function _events_group(T, children)
    return Expr(:call, T, Expr(:tuple, map(_events_lower, children)...))
end

function _events_bad(ex)
    error(
        "@events: unsupported expression `$(ex)`; write event names joined by " *
        "→ (chain), | (one_of) and & (parallel), grouped with parentheses, " *
        "e.g. `onset → admission → (death | discharge)`")
end

@doc "

Declare an event-tree topology as a readable operator diagram.

`@events` lowers an operator diagram of event names to an [`EventSkeleton`](@ref)
carrying structure only, no distributions. Fill the holes later with
[`update`](@ref)`(skeleton; name = dist, ...)` to build the concrete composed
tree, so one delay topology is reused across pathogens or settings.

The operators (parentheses group for precedence):

- `→` (`\\to`, typed `\\to<tab>`) chains events into a [`Sequential`](@ref); a
  nested `→` chain flattens into one sequential of all events.
- `|` branches into a one_of outcome. Whether the node becomes a
  fixed-probability [`Resolve`](@ref) or a racing-hazard [`Compete`](@ref) is
  decided at fill time by the fill value type (see [`update`](@ref)), so `|`
  stays one syntax.
- `&` runs branches in [`Parallel`](@ref).

A bare identifier is an event name and becomes a named hole, the key the fill
substitutes. A nested one_of or parallel group inside a `→` chain is named
deterministically from its branches (`_or_` for one_of, `_and_` for parallel,
e.g. `death | discharge` names its enclosing step `death_or_discharge`); a fill
names only the branch holes, never the group.

# Arguments
- `body`: the event diagram, either a bare expression or a `begin ... end` block
  holding exactly one diagram expression.

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
- [`EventSkeleton`](@ref): the topology type this builds.
- [`update`](@ref): fill the holes to build the concrete tree.
- [`sequential`](@ref), [`parallel`](@ref), [`resolve`](@ref),
  [`compete`](@ref): the verbs the fill lowers to.
"
macro events(body)
    spec = _events_body_expr(body)
    return Expr(:call, EventSkeleton, _events_lower(spec))
end
