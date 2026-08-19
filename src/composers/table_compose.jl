# ============================================================================
# compose(table): rebuilding a tree from its own `composed_to_table` rows
# ============================================================================
#
# `composed_to_table` writes a tree out; this reads one back, closing the round
# trip. The whole thing rests on one property of the table, which is what makes
# it parseable without a registry or a pattern-matched list of special cases:
#
#   AN EDGE IS A REAL TREE POSITION IF AND ONLY IF IT HAS A `:node` ROW.
#
# Every composer node, every leaf and every leaf-wrapper layer emits one. The
# four edges that are NOT tree positions -- a `shared` leaf's tag edge, a pool
# group's hyperparameter edge, a `Resolve`'s `<edge>.branch_probs` block and a
# non-centred pooled parameter's `<leaf>.<param>` latent -- emit only `:param`
# rows. So the two synthetic edges that otherwise look exactly like children
# (`<edge>.branch_probs` and `<leaf>.<param>`) fall out of the child search on
# their own, rather than being matched by name and hoped for. Getting that
# wrong would build a WRONG TREE rather than throw, so it is worth having the
# structure decide it rather than a rule.
#
# The second lever is that the rows already have a verified writer. `update(d,
# table)` sets every value and every spec on a tree of the right shape and is
# an identity on `composed_to_table(d)` for every shape (see
# `_table_to_nested_updates`). So this reader only has to rebuild the
# STRUCTURE -- the node kinds, their names and nesting, each leaf's wrapper
# stack and its fixed attributes -- and then hand the table to `update` for the
# values and specs. Nothing here re-implements spec attachment, `Uncertain`
# layer placement, stick-breaking or pooling.
#
# The three hooks below are how a type says how to rebuild itself. Each is
# dispatched on the TYPE, taken straight from the `node` column, so there is no
# name-to-type registry anywhere and no load-order or world-age hazard: an
# unloaded type simply cannot appear in a table you are holding.

@doc "

Rebuild a leaf distribution from its [`composed_to_table`](@ref) rows.

The reconstruction half of [`leaf_ctor`](@ref), dispatched on the leaf's type
rather than on an instance (there is no instance yet). The default calls the
type with the parameter values positionally, `T(values...)`, which is right
for any Distributions.jl family whose `params` are its constructor arguments —
so a leaf still costs zero methods.

A leaf whose free parameters are not its constructor arguments (one overriding
[`param_names`](@ref)/[`rebuild_leaf`](@ref), e.g. a moment-parameterised
wrapper) needs a method here. The default detects that case rather than
building a wrong leaf silently: it checks the rebuilt leaf reports the same
parameter names and values it was given, and throws naming this function if
not.

# Arguments
- `T`: the leaf type, as carried by the table's `node` column.
- `values`: the leaf's parameter values, keyed and ordered as
  [`leaf_param_names`](@ref) reports them.
- `attrs`: the layer's [`node_attributes`](@ref) entries, as recovered from
  its `:attribute` rows (empty for a plain leaf family).

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.from_table(Gamma, (alpha = 2.0, theta = 1.0), (;))
```

# See also
- [`rewrap_from_table`](@ref): the same job for a leaf-wrapper layer.
- [`node_from_table`](@ref): the same job for a composer node.
- [`composed_to_table`](@ref): the table this reads.
"
function from_table(::Type{T}, values::NamedTuple, ::NamedTuple) where {T}
    leaf = _from_table_construct(T, values)
    _check_leaf_coordinates(T, leaf, values)
    return leaf
end

# `T(values...)` with the failure rewritten to name the hook: a leaf whose
# constructor does not take its `params` positionally (a wrapper taking an
# inner distribution, say) fails here with a `MethodError` that says nothing
# about how to fix it.
function _from_table_construct(::Type{T}, values::NamedTuple) where {T}
    try
        return T(values...)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "compose(table): cannot rebuild a $(T) leaf by calling it with " *
            "its parameter values $(values). Define " *
            "`ComposedDistributions.from_table(::Type{<:$(T)}, values, " *
            "attrs)` for a leaf whose free parameters are not its " *
            "constructor arguments"))
    end
end

# The post-condition that keeps a wrong rebuild from passing silently: the
# leaf we built must report back the same coordinates we put in. A leaf
# overriding `param_names`/`leaf_ctor` (its free parameters are not its
# constructor arguments, in that order) fails this, which is exactly the case
# needing a `from_table` method of its own.
function _check_leaf_coordinates(::Type{T}, leaf, values::NamedTuple) where {T}
    got = leaf_param_names(leaf)
    Tuple(got) == keys(values) && Tuple(params(free_leaf(leaf))) ==
                                  Tuple(values) && return nothing
    throw(ArgumentError(
        "compose(table): rebuilding a $(T) leaf from $(values) gave a leaf " *
        "reporting $(got) = $(params(free_leaf(leaf))) instead, so its free " *
        "parameters are not its constructor arguments in that order. Define " *
        "`ComposedDistributions.from_table(::Type{<:$(T)}, values, attrs)`"))
end

@doc "

Re-apply a leaf-wrapper layer around a rebuilt inner distribution, from its
[`composed_to_table`](@ref) `:attribute` rows.

The reconstruction half of [`node_attributes`](@ref): whatever a wrapper
reports there as its fixed structure comes back here as `attrs`, and this
method puts the layer back around `inner`. Dispatched on the layer's type,
taken from the table's `node` column.

There is no default — a wrapper's structure is its own — so a wrapper type
that appears in a table without a method here throws, naming itself and this
function. The built-in wrappers (`Truncated`, `Censored`, [`Shared`](@ref),
[`Varying`](@ref)) have methods; an [`uncertain`](@ref) layer needs none,
since its specs ride the `prior` column and are applied by
[`update`](@ref)`(d, table)`.

# Arguments
- `T`: the wrapper layer's type, as carried by the table's `node` column.
- `inner`: the already-rebuilt distribution this layer wraps.
- `attrs`: the layer's [`node_attributes`](@ref) entries, recovered from its
  `:attribute` rows.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.rewrap_from_table(
    Truncated, Gamma(2.0, 1.0), (lower = nothing, upper = 10.0))
```

# See also
- [`from_table`](@ref): the same job for the leaf inside the stack.
- [`inner_dist`](@ref): the peel this reverses.
"
function rewrap_from_table(::Type{T}, inner, attrs::NamedTuple) where {T}
    throw(ArgumentError(
        "compose(table): $(T) is a leaf-wrapper layer with no way to rebuild " *
        "it. Define `ComposedDistributions.rewrap_from_table(::Type{<:$(T)}, " *
        "inner, attrs)`, taking the layer's node_attributes back " *
        "($(keys(attrs)) here) and re-applying the layer around `inner`"))
end

@doc "

Rebuild a composer node from its [`composed_to_table`](@ref) rows and its
already-rebuilt children.

The reconstruction half of [`node_rebuild`](@ref), dispatched on the node's
type rather than on an instance (there is no instance yet). The default calls
the type with its children and their names, `T(children, names)` — the shape
[`Sequential`](@ref)/[`Parallel`](@ref) and the documented third-party node
pattern already use — so a node following that pattern still costs only the
three methods of the node contract.

A node whose constructor takes something else needs a method here.
[`Choose`](@ref), [`Compete`](@ref) and [`Resolve`](@ref) have their own, as
do the `Convolved`/`Difference` composites.

# Arguments
- `T`: the node type, as carried by the table's `node` column.
- `names`: the child names, in table order, matching
  [`component_names`](@ref).
- `children`: the rebuilt children, in the same order.
- `attrs`: the node's [`node_attributes`](@ref) entries recovered from its
  `:attribute` rows, plus any node-level parameter block the table records
  separately (a [`Resolve`](@ref)'s `branch_probs`).

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.node_from_table(
    Sequential, (:onset, :admit), (Gamma(2.0, 1.0), LogNormal(0.5, 0.4)), (;))
```

# See also
- [`node_rebuild`](@ref): the same-shape rebuild from an existing instance.
- [`node_attributes`](@ref): the hook feeding `attrs`.
"
function node_from_table(::Type{T}, names::Tuple, children::Tuple,
        ::NamedTuple) where {T}
    try
        return T(children, names)
    catch err
        err isa MethodError || rethrow()
        throw(ArgumentError(
            "compose(table): cannot rebuild a $(T) node by calling it with " *
            "its children and names. Define " *
            "`ComposedDistributions.node_from_table(::Type{<:$(T)}, names, " *
            "children, attrs)`"))
    end
end

# --- the table index -------------------------------------------------------

# Everything the walk needs about the table, computed once: the row indices at
# each edge (in table order, which is the tree's pre-order), the real tree
# edges, and each real edge's children as `name => child edge` in first-seen
# order.
struct _TableIndex{C}
    cols::C
    rows::Dict{Symbol, Vector{Int}}
    real::Set{Symbol}
    children::Dict{Symbol, Vector{Pair{Symbol, Symbol}}}
end

function _table_index(cols, edges, roles)
    rows = Dict{Symbol, Vector{Int}}()
    real = Set{Symbol}()
    order = Symbol[]
    for i in eachindex(edges)
        e = edges[i]
        haskey(rows, e) || push!(order, e)
        push!(get!(rows, e, Int[]), i)
        Symbol(roles[i]) === :node && push!(real, e)
    end
    children = Dict{Symbol, Vector{Pair{Symbol, Symbol}}}()
    for e in order
        e in real || continue
        path = _edge_path(e)
        isempty(path) && continue
        parent = _join_path(path[1:(end - 1)])
        push!(get!(children, parent, Pair{Symbol, Symbol}[]), path[end] => e)
    end
    return _TableIndex(cols, rows, real, children)
end

_col(ix::_TableIndex, name::Symbol) = Tables.getcolumn(ix.cols, name)
function _col(ix::_TableIndex, name::Symbol, default)
    return name in Tables.columnnames(ix.cols) ? Tables.getcolumn(ix.cols,
        name) : default
end

# --- the walk --------------------------------------------------------------

# Rebuild the subtree at `edge`. A real edge is a composer node when it has
# children (every composer has at least one; no leaf ever does — its own
# synthetic `<leaf>.<param>` pool edge carries no `:node` row and so is not a
# child), a leaf-wrapper stack otherwise.
function _build_from_table(ix::_TableIndex, edge::Symbol)
    layers, attrs = _layer_rows(ix, edge)
    isempty(layers) && throw(ArgumentError(
        "compose(table): edge $(repr(edge)) has no `:node` row, so the " *
        "table does not say what kind of node it is"))
    kids = get(ix.children, edge, Pair{Symbol, Symbol}[])
    if !isempty(kids)
        names = Tuple(first(k) for k in kids)
        children = Tuple(_build_from_table(ix, last(k)) for k in kids)
        # The node's own `:attribute` rows win over a node-level parameter
        # block of the same name: an uncertain `Resolve` records its exact
        # probabilities as an attribute BECAUSE its `branch_probs` param rows
        # are stick coordinates, so the attribute is the better of the two.
        node_attrs = merge(_node_level_params(ix, edge), attrs[1])
        return node_from_table(layers[1], names, children, node_attrs)
    end
    layers[1] === NoEvent && return NoEvent()
    return _build_leaf_from_table(ix, edge, layers, attrs)
end

# A real edge's `:node` rows (its composer node, or its leaf-wrapper stack
# outermost first) paired with each layer's own `:attribute` entries. Rows are
# read in table order: a `:node` row opens a layer and the `:attribute` rows
# after it are that layer's, exactly the order `_emit_layers!` writes them. A
# `Pool` attribute row is a parameter spec rather than a layer attribute (it
# is emitted later, among the `:param` rows) and is left to the leaf walk.
function _layer_rows(ix::_TableIndex, edge::Symbol)
    roles, nodes, params_col = _col(ix, :role), _col(ix, :node),
    _col(ix, :param)
    values = _col(ix, :value)
    layers = Any[]
    attrs = NamedTuple[]
    for i in ix.rows[edge]
        role = Symbol(roles[i])
        if role === :node
            push!(layers, nodes[i])
            push!(attrs, (;))
        elseif role === :attribute && nodes[i] !== Pool && !isempty(layers)
            attrs[end] = merge(attrs[end],
                NamedTuple{(params_col[i],)}((values[i],)))
        end
    end
    return layers, attrs
end

# A node's own parameter block, if the table records one at a child-shaped
# edge that is not a tree position (a `Resolve`'s `<edge>.branch_probs`).
# Returned as `blockname = (paramname = value, ...)` to merge into the node's
# attributes. Generic rather than `Resolve`-specific so a downstream node with
# its own node-level parameters reads back the same way.
function _node_level_params(ix::_TableIndex, edge::Symbol)
    roles, params_col, values = _col(ix, :role), _col(ix, :param),
    _col(ix, :value)
    path = _edge_path(edge)
    out = (;)
    for (e, idxs) in ix.rows
        e in ix.real && continue
        p = _edge_path(e)
        length(p) == length(path) + 1 || continue
        p[1:(end - 1)] == path || continue
        block = (;)
        for i in idxs
            Symbol(roles[i]) === :param || continue
            block = merge(block, NamedTuple{(params_col[i],)}((values[i],)))
        end
        isempty(block) && continue
        out = merge(out, NamedTuple{(p[end],)}((block,)))
    end
    return out
end

# Rebuild a (possibly wrapped) leaf: the innermost layer from its parameter
# values, then each wrapper layer re-applied outwards from its own attributes.
# An `Uncertain` layer is skipped — its specs ride the `prior` column and are
# applied by the `update(d, table)` pass, which also places the layer.
function _build_leaf_from_table(ix::_TableIndex, edge::Symbol, layers, attrs)
    tag = _table_shared_tag(layers, attrs)
    values = _leaf_values(ix, tag === nothing ? edge : tag)
    leaf = from_table(layers[end], values, attrs[end])
    for i in (length(layers) - 1):-1:1
        layers[i] === Uncertain && continue
        leaf = rewrap_from_table(layers[i], leaf, attrs[i])
    end
    return leaf
end

# The `tag` attribute of a `Shared` layer in this stack, or `nothing`. A shared
# leaf's parameter rows (and its pooled parameters' `:attribute` rows) are
# written under the tag rather than under its path, and every occurrence of the
# tag reads that one block.
function _table_shared_tag(layers, attrs)
    for i in eachindex(layers)
        layers[i] === Shared && return attrs[i].tag
    end
    return nothing
end

# A leaf's parameter values in `leaf_param_names` order, read off the rows at
# its value edge in table order (the order the walk writes them). A pooled
# parameter has no usable `:param` row — its row is a `Normal(0, 1)` latent or
# a `CentredPoolPrior`-marked value — so its own value comes from the `template`
# entry of its `Pool` `:attribute` row, which sits at exactly that parameter's
# position in the same stream.
function _leaf_values(ix::_TableIndex, edge::Symbol)
    haskey(ix.rows, edge) || throw(ArgumentError(
        "compose(table): no parameter rows at $(repr(edge))"))
    roles, nodes = _col(ix, :role), _col(ix, :node)
    params_col, values = _col(ix, :param), _col(ix, :value)
    out = (;)
    for i in ix.rows[edge]
        role = Symbol(roles[i])
        v = if role === :param && nodes[i] !== Pool
            values[i]
        elseif role === :attribute && nodes[i] === Pool
            values[i].template
        else
            continue
        end
        out = merge(out, NamedTuple{(params_col[i],)}((v,)))
    end
    return out
end

# --- entry point -----------------------------------------------------------

# The round-trip table front end of `compose` (see `compose.jl` for the column
# dispatch). Rebuilds the structure, then hands the same table to `update` for
# every value and spec, so the two directions cannot drift apart.
function _compose_round_trip_table(table)
    cols = Tables.columns(table)
    colnames = Tables.columnnames(cols)
    missing_cols = filter(c -> !(c in colnames), (:edge, :param, :node, :role,
        :value))
    isempty(missing_cols) || throw(ArgumentError(
        "compose(table): a round-trip table needs $(missing_cols) as well as " *
        "$(intersect((:edge, :param, :node, :role, :value), colnames)); got " *
        "columns $(collect(colnames)). The parameter-only projection " *
        "(`role == :param` rows) cannot be composed — it carries no `:node` " *
        "or `:attribute` rows, so it does not describe the tree's structure"))
    edges = Tables.getcolumn(cols, :edge)
    isempty(edges) && throw(ArgumentError("compose(table): the table is empty"))
    ix = _table_index(cols, edges, Tables.getcolumn(cols, :role))
    root = Symbol("")
    root in ix.real || throw(ArgumentError(
        "compose(table): the table has no root `:node` row (an empty `edge`), " *
        "so it is not a whole tree. `composed_to_table` writes one; a " *
        "filtered or partial table cannot be composed"))
    _check_table_namespace(ix)
    # `_table_to_nested_updates` rather than `update(tree, table)`: a column
    # table IS a `NamedTuple`, which `update`'s nested-NamedTuple method
    # matches first and would read as a tree-shaped update keyed `edge`,
    # `param`, ... . Calling the table reader directly says which of the two
    # this is, whatever the table's concrete type.
    return update(_build_from_table(ix, root), _table_to_nested_updates(table))
end

# A `shared` tag doubling as a real tree edge makes two different leaves' rows
# collide under one edge, and the reader cannot tell them apart. `update`'s
# own gate (`validate_tree_names`) catches this on a BUILT tree; here the tree
# does not exist yet, so check the rows directly and refuse with the same
# advice rather than mis-parse.
function _check_table_namespace(ix::_TableIndex)
    roles, nodes = _col(ix, :role), _col(ix, :node)
    params_col, values = _col(ix, :param), _col(ix, :value)
    for i in eachindex(roles)
        Symbol(roles[i]) === :attribute && nodes[i] === Shared &&
        params_col[i] === :tag || continue
        values[i] in ix.real && throw(ArgumentError(
            "compose(table): $(repr(values[i])) is used as both a shared tag " *
            "and a tree edge name, so the two carry rows under one edge and " *
            "cannot be told apart; rename one"))
    end
    return nothing
end
