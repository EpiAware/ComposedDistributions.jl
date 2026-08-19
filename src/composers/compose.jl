@doc "

Build a nested composer stack from a friendly front-end input.

`compose` lowers a NamedTuple, a Tables.jl table, or a nested matrix to the same
[`Sequential`](@ref) / [`Parallel`](@ref) stack. It is a constructor over the
composers, not a new tree type.

# Arguments
- `input`: the front-end to lower, one of the three forms below.

# Inputs

- `NamedTuple` (named, recursive): a [`Parallel`](@ref) over the named children.
  A child that is itself a `NamedTuple` nests as a `Parallel`, a child that is a
  `Vector` or `Tuple` of distributions nests as a [`Sequential`](@ref), and a
  bare `UnivariateDistribution` is a leaf branch.
- Tables.jl AUTHORING table, with `name` and `dist` columns: a
  [`Parallel`](@ref) over the rows, the column-table equivalent of a flat
  `NamedTuple`. An optional `chain` column folds rows sharing a non-zero group
  id into a [`Sequential`](@ref) branch, and an optional `compete`/`prob`
  column pair folds rows sharing a non-zero `compete` id into a
  [`Resolve`](@ref) node whose `prob` entries are the branch probabilities
  (each in ``[0, 1]`` and summing to one per group). One row per branch, a
  whole distribution per `dist` cell: the shape for writing a tree by hand.
- Tables.jl ROUND-TRIP table, with the `edge`/`param`/`node`/`role`/`value`
  columns [`composed_to_table`](@ref) writes: the inverse of that function, so
  `compose(composed_to_table(d)) == d`. One row per structural atom with
  scalar values, so a parameter can be edited in place (in a `DataFrame`, say)
  and composed back. Only an in-memory round trip is supported, as
  [`composed_to_table`](@ref) documents.
- nested `Matrix` of distributions: rows are [`Parallel`](@ref) branches and the
  columns within a row are [`Sequential`](@ref) steps. This orientation is
  canonical, so a one-column matrix is parallel leaf branches (one row each) and
  a one-row matrix is a [`Parallel`](@ref)-of-one wrapping a [`Sequential`](@ref)
  of the row's columns.

A varargs-pairs spelling, `compose(:a => d1, :b => d2, ...)`, is also
available as a convenience over the NamedTuple form (see the
`compose(pairs::Pair{Symbol}...)` method below); the NamedTuple form above
stays primary.

The two table shapes are told apart by their columns, not by a separate
function, and they are deliberately different because they do different jobs:
the authoring shape puts a whole distribution in a cell, the round-trip shape
one scalar per row. A table carrying both column sets is refused as ambiguous.
Only the round-trip shape is an inverse; the authoring shape has none.

# Contract

`compose` *always* returns a composer, never a bare univariate leaf.
A single branch stays a [`Parallel`](@ref)-of-one and a single step a
one-element [`Sequential`](@ref); the wrapper is never collapsed away.
A bare leaf is used directly at the scoring layer, where downstream helpers
such as `record_distributions` and `composed_distribution_model` accept a
bare `UnivariateDistribution`, so callers do not need `compose` to pass one
through.

# Examples
```@example
using ComposedDistributions, Distributions

# A regular 2x2 grid built three ways, all equal.
nt = (r1 = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
    r2 = [Gamma(1.0, 1.0), Gamma(3.0, 1.0)])
table = (name = [:a, :b, :c, :d],
    dist = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4),
        Gamma(1.0, 1.0), Gamma(3.0, 1.0)],
    chain = [1, 1, 2, 2])
mat = [Gamma(2.0, 1.0) LogNormal(0.5, 0.4); Gamma(1.0, 1.0) Gamma(3.0, 1.0)]
compose(nt) == compose(table) == compose(mat)
```

The three front-ends are chosen to build identical stacks for the same
structure, as the example above shows.

# See also
- [`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref): the composers
"
function compose end

# --- NamedTuple front-end --------------------------------------------------
# A NamedTuple maps to a Parallel over its values, each value lowered by
# `_compose_child`. The keys become the branch names, threaded into the
# `Parallel` so `params`/`composed_to_table`/`show` are name-keyed (Option A).
# Structurally this still matches the table and matrix forms (`==` ignores
# names); only the labels differ.
#
# A column table is also a NamedTuple, so a NamedTuple carrying `name`/`dist`
# column vectors is routed to the Tables.jl path instead, letting one
# `(name, dist, chain)` column table build the same stack as a structural
# NamedTuple.
function compose(nt::NamedTuple)
    _is_column_table(nt) && return _compose_table(nt)
    children = map(_compose_child, Tuple(nt))
    return Parallel(children, keys(nt))
end

# --- shared-origin front-end ----------------------------------------------
# `compose(origin; branch = ...)` shares `origin` across the named branches: the
# branches fan out from one origin, so the result is a Sequential whose last step
# is a Parallel of the branch tails. Convolving the stack returns one series per
# branch, each delayed by `origin` convolved with the branch tail (e.g. a shared
# incubation, then a reporting branch and a death branch). The origin is not
# type-bounded (a duck-typed leaf is a valid origin), so this shares its
# positional signature with the table front-end below -- keyword arguments do
# not dispatch -- and the two live in one method there, splitting on whether any
# branch was given. This worker does the fan-out half.
function _compose_origin(origin, nt::NamedTuple)
    tails = map(_compose_child, Tuple(nt))
    return Sequential((_compose_child(origin), Parallel(tails, keys(nt))))
end

# --- Pairs front-end ---------------------------------------------------
# `compose(:a => d1, :b => d2, ...)` is the varargs-pairs spelling of the
# NamedTuple front-end, a thin convenience so CensoredDistributions-style call
# sites (whose `compose` took varargs pairs) still work unmodified against
# this package's NamedTuple-primary `compose`. It just lowers to
# `NamedTuple(pairs)` and forwards, so it inherits the NamedTuple method's
# recursive lowering (a `Pair` value that is itself a `NamedTuple`, vector, or
# tuple nests exactly as it would if written positionally).

@doc "

Varargs-pairs spelling of the [`compose`](@ref) NamedTuple front-end.

`compose(:a => d1, :b => d2, ...)` lowers to `compose((a = d1, b = d2, ...))`
and returns exactly the same stack; this is a convenience spelling for
data-driven or computed branch names, and for callers migrating from
CensoredDistributions' pairs-based `compose`. The NamedTuple form stays the
primary spelling (see the FAQ for the migration note).

# Arguments
- `pairs`: one or more `name => dist_or_composer` pairs, `name` a `Symbol`.

# Examples
```@example
using ComposedDistributions, Distributions

compose(:onset_admit => Gamma(2.0, 1.0), :admit_death => LogNormal(0.5, 0.4)) ==
    compose((onset_admit = Gamma(2.0, 1.0), admit_death = LogNormal(0.5, 0.4)))
```

# See also
- [`compose`](@ref): the primary NamedTuple/table/matrix front-end.
"
function compose(pairs::Pair{Symbol}...)
    isempty(pairs) &&
        throw(ArgumentError("compose(pairs::Pair{Symbol}...) needs ≥1 pair"))
    return compose(NamedTuple(pairs))
end

# A NamedTuple is treated as a column table when it has `name` and `dist`
# fields that are both vectors (the column-table shape), and those vectors
# carry the column roles of a real table: the `:dist` column holds
# distributions and the `:name` column holds row labels (not distributions).
# This disambiguates a genuine `(name, dist)` table from a structural
# NamedTuple whose user-chosen branch keys happen to be `:name`/`:dist`
# carrying distribution vectors, e.g. `(name = [d1, d2], dist = [d3, d4])` —
# two named chain branches, not a table.
function _is_column_table(nt::NamedTuple)
    return _is_authoring_column_table(nt) || _is_round_trip_column_table(nt)
end

function _is_authoring_column_table(nt::NamedTuple)
    haskey(nt, :name) && haskey(nt, :dist) &&
        nt.name isa AbstractVector && nt.dist isa AbstractVector &&
        all(is_composable, nt.dist) && !any(is_composable, nt.name)
end

# The round-trip shape as a `NamedTuple` of columns, which is what
# `Tables.columntable(composed_to_table(d))` (or of the tree itself) gives. Its
# `edge`/`param`/`role` columns hold `Symbol`s, never distributions, so this
# cannot be confused with a structural NamedTuple whose branch names happen to
# be `:edge`/`:param`/`:role` — those would carry composables.
function _is_round_trip_column_table(nt::NamedTuple)
    cols = (:edge, :param, :role)
    all(c -> haskey(nt, c) && nt[c] isa AbstractVector, cols) || return false
    return !any(c -> any(is_composable, nt[c]), cols)
end

# Lower a single front-end value to a composer child. A nested NamedTuple
# recurses (carrying its own keys); a bare vector/tuple of composables becomes a
# Sequential with default `:step_i` names (a plain vector has no names to carry).
# A pre-built multivariate composer value (Sequential/Parallel/Choose, or a
# downstream `AbstractComposedDistribution` node) drops in unchanged, so a
# `compose(...)` result nests as a child and a `Sequential((...), names)` value
# keeps readable step names. Anything else drops in unchanged: a leaf (a plain
# distribution, a `Resolve`, or a duck-typed leaf implementing the univariate
# interface without subtyping it) or a pre-built composer value. The elements
# are not re-checked here -- the composer constructor this feeds is the single
# front door, so `is_composable` is applied once rather than twice.
_compose_child(x) = x
_compose_child(nt::NamedTuple) = compose(nt)
function _compose_child(v::Union{AbstractVector, Tuple})
    return Sequential(map(_compose_child, Tuple(v)))
end

# --- nested Matrix front-end -----------------------------------------------
# A matrix maps to a Parallel over its rows (branches), each row a Sequential
# over its columns (chain steps). A row with a single entry collapses to that
# bare leaf, so a one-column matrix is parallel leaf branches (one row each)
# and a one-row matrix is a single Sequential chain (the lone row's columns),
# matching the NamedTuple/table forms for the same structure.
#
# Names thread through optional keyword arguments (Option A): `names`
# labels the row branches and `step_names` labels the columns within each
# multi-step row. Both fall back to positional defaults (`:branch_i` /
# `:step_j`) when omitted, so the matrix form still works name-free.
function compose(m::AbstractMatrix;
        names = nothing, step_names = nothing)
    nrows, ncols = size(m)
    (nrows >= 1 && ncols >= 1) ||
        throw(ArgumentError("the matrix needs at least one row and column"))
    branch_names = _coerce_names(names, :branch, nrows)
    col_names = ncols == 1 ? nothing : _coerce_names(step_names, :step, ncols)
    branches = ntuple(nrows) do i
        steps = Tuple(m[i, j] for j in 1:ncols)
        ncols == 1 ? steps[1] : Sequential(steps, col_names)
    end
    return Parallel(branches, branch_names)
end

# --- Tables.jl table front-end ---------------------------------------------
# A table with `name` and `dist` columns maps to a Parallel over its rows, the
# column-table equivalent of a flat NamedTuple of leaves. An optional `chain`
# column groups consecutive rows that share a non-zero group id into one
# Sequential branch, so a table can also express the nested chain a NamedTuple
# encodes with a vector value. An optional `compete`/`prob` column pair folds
# the rows sharing a non-zero `compete` group id into one `Resolve` node (the
# `prob` entries its branch probabilities), so the table can also express a
# one_of-outcome set. The generic method accepts any Tables.jl source (a
# column table is also matched by the NamedTuple method, which delegates here);
# the `_compose_table` worker does the shared build.
# Shares its positional signature with the shared-origin front-end above (a
# duck-typed origin cannot be told from a table by dispatch, and keyword
# arguments do not dispatch), so the two are one method splitting on whether any
# branch was given. The `is_composable` check preserves the origin form's own
# "needs a branch" error rather than sending a lone distribution into the table
# reader.
function compose(table; branches...)
    isempty(branches) || return _compose_origin(table, NamedTuple(branches))
    # A table wins over the origin reading: `is_composable` admits anything not
    # positively excluded, and a `Tables.jl` source (a DataFrame, say) is not
    # one of the excluded shapes, so the table check has to come first.
    Tables.istable(table) && return _compose_table(table)
    is_composable(table) &&
        throw(ArgumentError("compose(origin; branches...) needs ≥1 branch"))
    throw(ArgumentError(
        "compose expects a NamedTuple, a Tables.jl table with `name` and " *
        "`dist` columns, or a nested Matrix; got $(typeof(table))"))
end

# `compose` reads TWO table shapes, told apart by their columns rather than by
# a separate verb, and they are deliberately different because they do
# different jobs:
#
#   - the AUTHORING table (`name`/`dist`, optional `chain`/`compete`/`prob`):
#     one row per branch, a whole distribution in the `dist` cell, the
#     `chain`/`compete` ids a compact grouping notation. For writing a tree by
#     hand or from a spreadsheet.
#   - the ROUND-TRIP table (`edge`/`param`/`node`/`role`/`value`, the shape
#     `composed_to_table` writes): one row per structural atom, scalar values,
#     per-parameter editable. The inverse of `composed_to_table`.
#
# Neither subsumes the other: collapsing them would make an author write
# node/attribute/param rows by hand, or make the round-trip shape smuggle whole
# distributions into cells and lose the per-parameter editing that is its
# point. A table carrying both column sets is ambiguous and is refused rather
# than guessed at.
function _compose_table(table)
    cols = Tables.columns(table)
    names = Tables.columnnames(cols)
    authoring = :name in names && :dist in names
    round_trip = :edge in names && :param in names && :role in names
    authoring && round_trip &&
        throw(ArgumentError(
            "compose(table): the table has both the authoring columns " *
            "(`name`/`dist`) and the round-trip columns (`edge`/`param`/`role`), " *
            "so which shape it is meant to be is ambiguous; drop one set"))
    round_trip && return _compose_round_trip_table(table)
    # The parameter-only projection (`edge`/`param`, no `role`) is the shape a
    # reader most plausibly reaches for, and it is exactly the one that cannot
    # work: it carries no `:node` or `:attribute` rows, so it says nothing
    # about the tree's structure. Say that rather than list columns.
    (:edge in names && :param in names) && throw(ArgumentError(
        "compose(table): this looks like the parameter-only projection of a " *
        "composed table (`edge`/`param` but no `role` column), which cannot " *
        "be composed — it carries only `:param` rows, so it records the " *
        "tree's parameters but not its structure. Pass the whole " *
        "composed_to_table(d) instead, or use update(d, table) to write " *
        "these values onto an existing tree"))
    authoring ||
        throw(ArgumentError(
            "compose(table): the table needs either `name` and `dist` " *
            "columns (the authoring shape) or the `edge`/`param`/`node`/" *
            "`role`/`value` columns composed_to_table writes (the round-trip " *
            "shape); got columns $(collect(names))"))
    dists = Tables.getcolumn(cols, :dist)
    row_names = Tables.getcolumn(cols, :name)
    all(is_composable, dists) ||
        throw(ArgumentError(
            "every `dist` entry must be a distribution-like leaf"))
    # A `prob` column only makes sense alongside `compete`, which marks the rows
    # the probabilities apply to; reject it alone rather than silently ignoring.
    (:prob in names && !(:compete in names)) &&
        throw(ArgumentError(
            "a `prob` column needs a `compete` column to mark its outcome set"))
    if :compete in names
        return _compose_table_one_of(dists, row_names,
            Tables.getcolumn(cols, :compete),
            :prob in names ? Tables.getcolumn(cols, :prob) : nothing,
            :chain in names ? Tables.getcolumn(cols, :chain) : nothing)
    end
    if :chain in names
        return _compose_table_chained(
            dists, row_names, Tables.getcolumn(cols, :chain))
    end
    # Flat table: each row is a branch, the `name` column its branch name.
    return Parallel(Tuple(dists), _coerce_names(row_names, :branch, length(dists)))
end

# Fold the rows sharing a non-zero `compete` group id into one `Resolve` node
# (its `prob` entries the branch probabilities); rows with a zero/`missing`
# `compete` id stay ordinary leaf branches (or, with a `chain` column, fold into
# Sequential branches by chain id). Branches appear in first-seen order — the
# row order of each group's first member — so the Parallel reads down the table,
# named by that first row, mirroring `_compose_table_chained`.
function _compose_table_one_of(dists, row_names, compete, prob, chain)
    all(g -> g === missing || g >= 0, compete) || throw(ArgumentError(
        "`compete` group ids must be non-negative or missing"))
    # One first-seen pass assigns each row a branch key: a `compete:id` for a
    # one_of group, else `chain:id` for a chained leaf (a zero/`missing`
    # `compete` and `chain` both make a fresh singleton key). The branch order is
    # the keys' first appearance, and `members` holds each key's rows in order.
    order = Any[]
    members = Dict{Any, Vector{Int}}()
    leaf_counter = 0
    has_compete = false
    for i in eachindex(dists)
        c = compete[i]
        if !(c === missing || c == 0)
            has_compete = true
            key = (:compete, Int(c))
        else
            ch = chain === nothing ? missing : chain[i]
            key = (ch === missing || ch == 0) ? (:leaf, leaf_counter -= 1) :
                  (:chain, Int(ch))
        end
        key in order || push!(order, key)
        push!(get!(members, key, Int[]), i)
    end
    has_compete || throw(ArgumentError(
        "the `compete` column marks no one_of rows (all zero/missing)"))
    branches = map(order) do key
        idx = members[key]
        if key[1] === :compete
            _one_of_from_rows(dists, row_names, prob, idx, key[2])
        elseif length(idx) == 1
            dists[idx[1]]
        else
            steps = Tuple(Symbol(row_names[i]) for i in idx)
            Sequential(Tuple(dists[i] for i in idx), steps)
        end
    end
    branch_names = Tuple(Symbol(row_names[members[key][1]]) for key in order)
    return Parallel(Tuple(branches), branch_names)
end

# Build one `Resolve` node from a compete group's rows: `name => (dist, prob)`
# per row. The constructor validates the branch probabilities sum to one and lie
# in `[0, 1]`; a missing `prob` in a compete row is an error (it is required).
function _one_of_from_rows(dists, row_names, prob, idx, gid)
    prob === nothing && throw(ArgumentError(
        "a `compete` group needs a `prob` column of branch probabilities"))
    outcomes = map(idx) do i
        p = prob[i]
        p === missing && throw(ArgumentError(
            "row $(row_names[i]) is in compete group $gid but has a missing " *
            "`prob`"))
        Symbol(row_names[i]) => (dists[i], p)
    end
    return Resolve(outcomes...)
end

# Group rows by the `chain` column: rows sharing a non-zero group id fold into
# one Sequential branch (in row order); a zero/`missing` group is a leaf branch.
# Branches appear in first-seen group order, matching the NamedTuple value order.
# Each branch is named by the first row of its group; the steps within a chained
# branch are named by their own rows' `name` entries (Option A).
function _compose_table_chained(dists, row_names, groups)
    # Group ids must be non-negative: a zero/`missing` group is a unique leaf,
    # to which a fresh negative id is assigned, so a negative user group would
    # collide with those auto-generated leaf ids.
    all(g -> g === missing || g >= 0, groups) ||
        throw(ArgumentError("`chain` group ids must be non-negative or missing"))
    order = Int[]              # group ids in first-seen order (0 -> unique leaf)
    members = Dict{Int, Vector{Int}}()
    leaf_counter = 0
    for i in eachindex(dists)
        g = groups[i]
        gid = (g === missing || g == 0) ? (leaf_counter -= 1) : Int(g)
        gid in order || push!(order, gid)
        push!(get!(members, gid, Int[]), i)
    end
    branches = map(order) do gid
        idx = members[gid]
        if length(idx) == 1
            dists[idx[1]]
        else
            step_names = Tuple(Symbol(row_names[i]) for i in idx)
            Sequential(Tuple(dists[i] for i in idx), step_names)
        end
    end
    # A chained branch takes the name of its first row; a leaf branch its own.
    branch_names = Tuple(Symbol(row_names[members[gid][1]]) for gid in order)
    return Parallel(Tuple(branches), branch_names)
end
