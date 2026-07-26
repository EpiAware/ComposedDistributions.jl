@doc raw"

A data-selected disjunction over independent named alternatives.

`Choose` holds ``n`` named alternatives ``D_1, \dots, D_n``, each an independent
sub-distribution, and a `selector` naming the data field that picks which
alternative applies to a record. Exactly one alternative is active per record,
chosen by the selector value, *not* by a branch probability and *not* off a
shared origin. This is the disjunctive split that neither [`Parallel`](@ref) (shared
origin, product over branches) nor [`Resolve`](@ref) (shared origin,
probabilistic mixture) expresses: the alternatives are genuinely independent
sub-models with different origins, and the data says which one generated the
record.

Scoring and model dispatch route to the selected alternative: `logpdf(d, x; kind)`
takes the chosen name as the `kind` keyword (no default — a `Choose` has no
single distribution to score without a selection). Sampling has two forms:
`rand(d; kind)` draws the named alternative directly, while a bare `rand(d)` (no
`kind`, the forward-simulation path) samples an alternative uniformly and
returns a self-describing record tagging which was drawn (the `selector` field
set to its name plus the alternative's own draw), so `logpdf(d, rand(d))`
round-trips with no `kind` argument. The selection walk is type-stable: the
selected alternative is found by a hand-rolled recursion over the name tuple
that barriers into the chosen alternative's concrete type, so inference of the
hot-path `logpdf` is preserved.

An alternative may itself be any distribution or a nested composer
([`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref), or another
`Choose`), so a composed tree nests inside a data-selected split. A `Choose`
may also nest the other way, as a child of a `Sequential` / `Parallel` /
[`compose`](@ref) composer: the flat, data-free value path (`logpdf`/`rand`
without a `kind`) commits to its first alternative, so the node's flat width is
that alternative's leaf count; every alternative must share that leaf count for
the nested `Choose` to occupy one fixed flat slot, or the parent's width query
errors.

For prior introspection ([`params_table`](@ref), [`build_priors`](@ref),
[`update`](@ref)) the alternatives' parameters are namespaced per alternative:
independent per-branch params live under their alternative name (`index.…` /
`sourced.…`), so each branch's parameters are inventoried and sampled separately.
A parameter tied across alternatives via [`shared`](@ref)`(:tag, ...)` is keyed
once by its `tag` and is inventoried once and sampled once, so the tied value is
shared by every alternative that uses it.

# Fields
- the alternative names (`Symbol`s) live in the `names` type parameter (read
  with [`component_names`](@ref)).
- `alternatives`: tuple of the alternative distributions, one per name.
- `selector`: the row field name (`Symbol`) whose value selects an alternative.

# See also
- [`choose`](@ref): friendly constructor over `name => dist` pairs
- [`Resolve`](@ref): exactly one of several shared-origin outcomes (mixture)
- [`Parallel`](@ref): independent shared-origin branches (product)
"
struct Choose{names, A <: Tuple} <:
       AbstractComposedDistribution{Multivariate, Continuous}
    "Tuple of the alternative distributions, one per name."
    alternatives::A
    "The row field name (`Symbol`) whose value selects an alternative."
    selector::Symbol

    function Choose{names}(alternatives::A, selector::Symbol) where {
            names, A <: Tuple}
        N = length(names)
        N >= 2 ||
            throw(ArgumentError("Choose needs at least two alternatives"))
        length(alternatives) == N ||
            throw(ArgumentError(
                "Choose needs one alternative per name; got $N names and " *
                "$(length(alternatives)) alternatives"))
        allunique(names) ||
            throw(ArgumentError("Choose alternative names must be unique"))
        all(_is_composable, alternatives) ||
            throw(ArgumentError(
                "every Choose alternative must be a UnivariateDistribution " *
                "or a nested composer"))
        new{names, A}(alternatives, selector)
    end
end

# The names live in the `names` type parameter (like `NamedTuple{names}`); this
# instantiates it directly from the runtime tuple `choose`/`compose` pass, with
# no call-site change from the field-based constructor.
function Choose(names::K, alternatives::A, selector::Symbol) where {
        N, K <: NTuple{N, Symbol}, A <: Tuple}
    return Choose{names}(alternatives, selector)
end

component_names(::Choose{names}) where {names} = names

@doc "

Build a [`Choose`](@ref) data-selected disjunction from `name => dist`
alternatives.

Each alternative is `name => dist`: the alternative name (a `Symbol`) and its
independent sub-distribution. The `selector` keyword names the data field a
record carries to pick an alternative (default `:kind`). At least two
alternatives are required and their names must be unique.

# Arguments
- `alternatives`: the `name => dist` pairs, each an independent sub-distribution
  (a `UnivariateDistribution` or a nested composer). A single named tuple
  `(name = dist, …)` is the equivalent positional spelling for hand-written
  alternatives, kept separate from the `selector` keyword; use Pairs for
  data-driven or computed names.

# Keyword Arguments
- `selector`: the row field name (`Symbol`) whose value picks an alternative
  (default `:kind`).

# Examples
```@example
using ComposedDistributions, Distributions

# An index case (a short delay) vs a sourced case (a longer coupled delay),
# selected by the row's `:kind` field.
d = choose(:index => Gamma(2.0, 1.0),
    :sourced => Gamma(4.0, 1.5))

# Score the alternative the data names.
logpdf(d, 3.0; kind = :index)
```

```@example
using ComposedDistributions, Distributions

# The equivalent named tuple spelling; `selector` stays a keyword.
d = choose((index = Gamma(2.0, 1.0), sourced = Gamma(4.0, 1.5)); selector = :kind)
logpdf(d, 3.0; kind = :index)
```

# See also
- [`Choose`](@ref): the disjunction type
"
function choose(alternatives::Pair...; selector::Symbol = :kind)
    length(alternatives) >= 2 ||
        throw(ArgumentError(
            "choose needs at least two alternatives"))
    # `map`, not `Tuple(gen)`, to keep the name/alternative tuples type-stable
    # and off the `collect_to!` `Array` temporary Enzyme cannot type-analyse
    # (see the `Resolve` constructor).
    names = map(a -> a.first, alternatives)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each Choose alternative name must be a Symbol"))
    dists = map(a -> a.second, alternatives)
    return Choose(names, dists, selector)
end

# Positional NamedTuple spelling: `(a = d1, …)` lowers to `:a => d1, …` Pairs,
# kept separate from the `selector` keyword (the alternatives are positional, so
# `selector` is not consumed as an alternative).
function choose(alternatives::NamedTuple; selector::Symbol = :kind)
    return choose(_nt_pairs(alternatives)...; selector = selector)
end

_n_alternatives(d::Choose) = length(component_names(d))

# --- Type-stable selection -------------------------------------------------
#
# `_pick(d, kind)` returns the alternative distribution whose name `=== kind`,
# by a hand-rolled recursion over the name/alternative tuples. Each step
# compares one `Symbol` and either returns the matching alternative or recurses
# on the tail. The recursion is over tuples of constant length, so the compiler
# union-splits / specialises it and the matching alternative is returned, then a
# downstream `logpdf`/`rand` barriers into its concrete type. This is not a
# runtime `Dict`/type lookup: no boxing, and a `kind` known at the call boundary
# keeps the hot path inferable.
@inline function _pick(d::Choose, kind::Symbol)
    return _pick_recurse(component_names(d), d.alternatives, kind)
end

@inline function _pick_recurse(
        names::Tuple, alternatives::Tuple, kind::Symbol)
    return first(names) === kind ? first(alternatives) :
           _pick_recurse(Base.tail(names), Base.tail(alternatives), kind)
end

# Base case: the name was not found in any alternative.
@inline _pick_recurse(::Tuple{}, ::Tuple{},
    kind::Symbol) = throw(ArgumentError("Choose has no alternative named $(repr(kind))"))

# The length of a realisation is the selected alternative's length. Without a
# selection there is no single length; `length(d)` errors to flag that a
# selection is required (mirroring `logpdf`/`rand`).
function Base.length(::Choose)
    throw(ArgumentError(
        "length(::Choose) needs a selection; the realisation length is the " *
        "selected alternative's. Use `length(ComposedDistributions._pick(" *
        "d, kind))` or pass `kind` to `logpdf`/`rand`."))
end

# The public `params(::Choose)` is positional, mirroring `params(::Resolve)`: a
# tuple of each alternative's `params` in alternative order. The name-keyed nested
# params tree (what prior introspection threads when a `Choose` is a child) goes
# through `_select_params`/`_child_params` in `introspection.jl`, keyed by the
# alternative names so a nested `Choose` yields a name-keyed subtree.
params(d::Choose) = map(params, d.alternatives)

@doc "

Log probability density of the selected alternative at `x`.

`Choose` is a data-selected disjunction, so scoring requires naming the active
alternative through the `kind` keyword; there is no default. The selection walk
is type-stable and the score is the selected alternative's own `logpdf`.

# Examples
```@example
using ComposedDistributions, Distributions

d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
logpdf(d, 3.0; kind = :short)
```

See also: [`Choose`](@ref)
"
# A scalar `x` scores a univariate selected alternative; a vector `x` scores a
# (possibly composer) selected alternative whose realisation is a flat vector.
# Both route through the type-stable `_pick`. Typing `x` keeps these methods
# distinct from the generic multivariate `logpdf(::Distribution, ::AbstractArray)`
# batch methods (the Aqua ambiguity check), since a `Choose`'s active dimension
# is the selected alternative's, not fixed.
function logpdf(
        d::Choose, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return _select_logpdf(d, x, kind)
end
function logpdf(d::Choose, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return _select_logpdf(d, x, kind)
end

function _select_logpdf(d::Choose, x, kind)
    kind === nothing && throw(ArgumentError(
        "logpdf(::Choose, x) needs a `kind` choose the alternative"))
    return logpdf(_pick(d, kind), x)
end

@doc "

Score a self-describing [`Choose`](@ref) record (the shape a bare `rand(d)`
returns).

A bare `rand(d)` draw is a `NamedTuple` whose `selector` field names the drawn
alternative and whose remaining fields are that alternative's labelled draw, so
`logpdf(d, rand(d))` round-trips with no `kind` argument: the selector field is
read to pick the alternative, then the rest of the record is scored under that
alternative's own `logpdf`. A leaf alternative's value rides in the `:value`
field; a composer alternative is scored on its own labelled record fields. A
column table of such records is summed per row.

See also: [`Choose`](@ref), [`rand`](@ref)
"
function logpdf(d::Choose, x::NamedTuple)
    # A column table (a `NamedTuple` of vectors) is a multi-record source: sum
    # the per-record scorer over its rows. A single tagged record scores below.
    Tables.istable(x) &&
        return sum(logpdf(d, r) for r in Tables.namedtupleiterator(x))
    haskey(x, d.selector) || throw(ArgumentError(
        "logpdf(::Choose, record) needs the selector field $(repr(d.selector)) " *
        "to name the drawn alternative; pass `kind` for a bare value instead"))
    kind = x[d.selector]
    kind isa Symbol || throw(ArgumentError(
        "the Choose selector field $(repr(d.selector)) must hold a Symbol " *
        "naming the alternative; got $(typeof(kind))"))
    inner = _drop_named_field(x, d.selector)
    return _choose_record_logpdf(_pick(d, kind), inner)
end

# Score the chosen alternative on its slice of the record (the selector field
# dropped), through the alternative's own `logpdf` so the same shape its `rand`
# produced round-trips: a leaf alternative scores its single `:value` field; a
# composer alternative scores its labelled record.
function _choose_record_logpdf(d::UnivariateDistribution, inner::NamedTuple)
    haskey(inner, :value) || throw(ArgumentError(
        "a leaf Choose alternative record needs a `value` field; got fields " *
        "$(collect(keys(inner)))"))
    return logpdf(d, inner.value)
end
_choose_record_logpdf(d, inner::NamedTuple) = logpdf(d, inner)

# Drop a single named field from a NamedTuple, preserving the order of the
# rest.
function _drop_named_field(row::NamedTuple, field::Symbol)
    ks = filter(!=(field), keys(row))
    return NamedTuple{ks}(map(k -> row[k], ks))
end

@doc "

Probability density of the selected alternative at `x`.

See also: [`logpdf`](@ref)
"
function pdf(d::Choose, x::Real; kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end
function pdf(d::Choose, x::AbstractVector{<:Real};
        kind::Union{Symbol, Nothing} = nothing)
    return exp(logpdf(d, x; kind = kind))
end

@doc "

Sample a [`Choose`](@ref), returning a self-describing record tagging which
alternative was drawn.

Without a `kind` (the forward-simulation path, where no data names the branch)
an alternative is sampled uniformly and the result is a `NamedTuple` carrying
the `selector` field set to the drawn alternative's name plus that
alternative's own draw, so the record identifies which alternative fired and
feeds straight back into [`logpdf`](@ref) with no extra arguments. A leaf
alternative's value is labelled `:value`; a composer alternative contributes
its own flat event-record fields.

With a `kind` (explicit selection) the draw is that alternative's own `rand`
returned directly (a scalar for a leaf, a labelled `NamedTuple` for a
composer), not wrapped in a selector tag: the caller already named the
alternative, so this is the in-tree / committed-selection path
(`logpdf(d, draw; kind)` scores it).

See also: [`Choose`](@ref), [`logpdf`](@ref)
"
function Base.rand(
        rng::AbstractRNG, d::Choose; kind::Union{Symbol, Nothing} = nothing)
    # Explicit selection: return the chosen alternative's raw draw (the caller
    # already knows the kind, so no selector tag is added — the committed /
    # in-tree path).
    kind === nothing || return rand(rng, _pick(d, kind))
    # Forward simulation: pick uniformly and return a self-describing record
    # that tags the drawn alternative (`selector => name`) so
    # `logpdf(d, rand(d))` recovers the alternative with no extra argument.
    chosen = component_names(d)[rand(rng, 1:_n_alternatives(d))]
    return _choose_tagged_record(d, chosen, rand(rng, _pick(d, chosen)))
end

function Base.rand(d::Choose; kind::Union{Symbol, Nothing} = nothing)
    rand(default_rng(), d; kind = kind)
end

# Build the self-describing record of a bare `Choose` draw: the `selector`
# field set to the drawn alternative's `name`, merged with that alternative's
# labelled draw. A leaf alternative's scalar draw is labelled `:value`
# (matching the univariate-leaf record key the vectorised path uses); a
# composer alternative already returns a labelled `NamedTuple` that merges in
# directly. The selector field comes first so the tag is read off the front.
function _choose_tagged_record(d::Choose, name::Symbol, draw)
    return merge(NamedTuple{(d.selector,)}((name,)), _choose_draw_fields(draw))
end
_choose_draw_fields(draw::NamedTuple) = draw
_choose_draw_fields(draw) = (; value = draw)

@doc "

Print a [`Choose`](@ref) node as its selector and named alternatives.

See also: [`Choose`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Choose)
    n = _n_alternatives(d)
    names = component_names(d)
    println(io,
        "Choose node of $n alternatives (selector = $(repr(d.selector)))")
    for k in 1:n
        branch = k == n ? "└─ " : "├─ "
        println(io, "  ", branch, "$(names[k]): $(d.alternatives[k])")
    end
    return nothing
end

function Base.show(io::IO, d::Choose)
    names = component_names(d)
    parts = ["$(names[k])" for k in 1:_n_alternatives(d)]
    print(io, "Choose(", join(parts, " | "),
        "; selector=", repr(d.selector), ")")
    return nothing
end
