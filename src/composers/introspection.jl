# Prior introspection helpers (`params`, `composed_to_table`, `event_names`,
# `event`); see their docstrings below. Implementation note: the `show` and
# `params`/`composed_to_table` traversals are hand-rolled, type-stable
# recursion over the component tuples, not generic tree iterators (not
# type-stable for the heterogeneous composer tree).

# --- node headers ----------------------------------------------------------

# Header label for a composer node (its type plus a count).
_node_header(d::Sequential) = "Sequential ($(length(d.components)) steps)"
_node_header(d::Parallel) = "Parallel ($(length(d.components)) branches)"
_node_header(c::Resolve) = "Resolve ($(_n_branches(c)) outcomes)"
function _node_header(c::Compete)
    return "Compete ($(_n_branches(c)) racing outcomes)"
end

# Is a child a composer (has named children) or a leaf?
_is_composer_dist(::Union{Sequential, Parallel, AbstractOneOf}) = true
_is_composer_dist(::Any) = false

# Named children of a composer as `(name, child[, note])` triples. The note is
# an extra print annotation (a `Resolve` outcome's branch probability), empty
# otherwise. Hand-rolled and type-stable (a tuple comprehension over the
# constant-length component tuple).
function _named_children(d::Union{Sequential, Parallel})
    names = component_names(d)
    return ntuple(i -> (names[i], d.components[i], ""), length(d.components))
end
function _named_children(c::Resolve)
    names = component_names(c)
    return ntuple(length(names)) do i
        (names[i], c.delays[i], "p = $(c.branch_probs[i])")
    end
end
# A racing-hazard node has no per-outcome branch probability (it is derived), so
# its children carry the `racing` annotation instead.
function _named_children(c::Compete)
    names = component_names(c)
    return ntuple(length(names)) do i
        (names[i], c.delays[i], "racing")
    end
end

# --- recursive indented-tree show (hand-rolled, type-stable) ---------------
#
# A nested composed distribution prints as one indented tree, recursing into
# every child so the whole structure is visible at once. Shared `├─ / └─` glyphs
# and indentation: a header line for the root, then each child indented one
# level, composer children recursing and leaf delays printed inline. The compact
# `show(io, d)` one-liners on each type are kept for inline/array display.

# Entry point shared by the three composer `show(::MIME"text/plain")` methods.
function _show_composer_tree(io::IO, d)
    println(io, _node_header(d))
    _show_children(io, d, "")
    return nothing
end

# A leaf's inline label. `show(io, leaf)` is the label, routed through this
# shim so a foreign leaf type whose upstream `show` is a multi-line struct dump
# still gets a one-line label inside the tree (Distributions.jl dumps any
# distribution with a distribution-valued field, `Censored` being the one that
# reaches a tree, over several lines; #282). A wrapper type this package or a
# downstream one OWNS needs no method here — it defines `Base.show` instead.
_leaf_label(leaf) = sprint(show, leaf)

# The label split into lines, trailing blank lines dropped. A label that is
# still multi-line (a foreign type with no shim) then keeps the tree prefix on
# every line rather than breaking out of the tree.
_leaf_label_lines(leaf) = split(rstrip(_leaf_label(leaf), '\n'), '\n')

# Print the named children of `node` under `prefix`. Each child gets a `├─ `
# connector (`└─ ` for the last); a composer child recurses with an extended
# prefix (`│  ` for non-last siblings, three spaces for the last), and a leaf's
# label continuation lines sit under that same prefix.
function _show_children(io::IO, node, prefix::String)
    children = _named_children(node)
    n = length(children)
    for i in 1:n
        last = i == n
        connector = last ? "└─ " : "├─ "
        name, child, note = children[i]
        label = isempty(note) ? "$(name): " : "$(name) ($(note)): "
        child_prefix = prefix * (last ? "   " : "│  ")
        if _is_composer_dist(child)
            println(io, prefix, connector, label, _node_header(child))
            _show_children(io, child, child_prefix)
        else
            lines = _leaf_label_lines(child)
            println(io, prefix, connector, label, first(lines))
            for line in Iterators.drop(lines, 1)
                println(io, child_prefix, line)
            end
        end
    end
    return nothing
end

# --- opt-in detailed inspection ---------------------------------------------
#
# `show` is deliberately compact (structure plus short leaf labels); `inspect`
# is the explicit opt-in for the full detail, recursing the same tree but
# printing each leaf's full `text/plain` representation (every field) under an
# indented prefix.

@doc "

Print a composed distribution's full nested detail.

`inspect(io, d)` walks the same tree as `show` but prints each leaf's full
`text/plain` representation (every field), so it is the opt-in companion to
the compact structural `show`. A composer node prints its header and
recurses; a leaf prints its detailed representation indented under its name.
Writes to `io` (default `stdout`) and returns nothing.

# Arguments
- `io`: the IO stream to print to (default `stdout`).
- `d`: the composed distribution (or bare leaf) to inspect.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
inspect(tree)
```

# See also
- [`event_tree`](@ref): the nested tree of event names
- [`composed_to_table`](@ref): the flat structural inventory
"
function inspect(io::IO, d)
    if _is_composer_dist(d)
        println(io, _node_header(d))
        _inspect_children(io, d, "")
    else
        _inspect_leaf(io, d, "")
    end
    return nothing
end

inspect(d) = inspect(stdout, d)

# Recurse the composer tree like `_show_children`, but print each leaf's full
# `text/plain` detail (rather than its compact one-line label) indented under
# its name.
function _inspect_children(io::IO, node, prefix::String)
    children = _named_children(node)
    n = length(children)
    for i in 1:n
        last = i == n
        connector = last ? "└─ " : "├─ "
        name, child, note = children[i]
        label = isempty(note) ? "$(name): " : "$(name) ($(note)): "
        child_prefix = prefix * (last ? "   " : "│  ")
        if _is_composer_dist(child)
            println(io, prefix, connector, label, _node_header(child))
            _inspect_children(io, child, child_prefix)
        else
            println(io, prefix, connector, label)
            _inspect_leaf(io, child, child_prefix)
        end
    end
    return nothing
end

# A leaf's detail lines, indented under `prefix`, one line at a time (so a
# multi-line detail stays aligned). The lines come from the `_leaf_detail_lines`
# hook so a leaf-wrapper layer can supply richer per-leaf detail.
function _inspect_leaf(io::IO, leaf, prefix::String)
    for line in _leaf_detail_lines(leaf)
        println(io, prefix, line)
    end
    return nothing
end

@doc raw"

Per-leaf `inspect` detail lines for a (possibly wrapped) leaf.

The leaf-detail extension point [`inspect`](@ref) reads through
`_inspect_leaf`: the generic method returns the leaf's full `text/plain` show,
split into lines so a multi-line struct dump stays aligned under its tree
prefix. A leaf-wrapper type (censoring in CensoredDistributions, the modifiers
in ModifiedDistributions) adds its own method dispatching on its own type, to
surface the inner free delay's detail instead of the wrapper's raw struct
dump. Pair with [`uncertain_specs`](@ref), the sibling extension hook for the
`prior` column.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution to render.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.leaf_detail_lines(Gamma(2.0, 1.0))
```

# See also
- [`free_leaf`](@ref), [`rewrap_leaf`](@ref): the sibling leaf-wrapper hooks.
- [`inspect`](@ref): the tree-printing entry point this feeds.
"
leaf_detail_lines(leaf) = split(sprint(show, MIME"text/plain"(), leaf), '\n')
const _leaf_detail_lines = leaf_detail_lines

# --- nested name-keyed params (hand-rolled, type-stable) --------------------

@doc "

Nested, name-keyed parameters of a composed distribution.

Returns a `NamedTuple` keyed by the node names, each value the `params` of that
child (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). A `Resolve` node contributes a name-keyed
NamedTuple of its outcomes plus a `branch_probs` entry. This nested form is for
prior introspection via [`composed_to_table`](@ref); a composed distribution
reconstructs through [`compose`](@ref), not through `Distribution(params...)`.

See also: [`composed_to_table`](@ref), [`event_names`](@ref), [`event`](@ref)
"
function _composed_params(d::Union{Sequential, Parallel})
    names = component_names(d)
    vals = map(_child_params, d.components)
    return NamedTuple{names}(vals)
end

_child_params(c::Union{Sequential, Parallel}) = _composed_params(c)
_child_params(c::Resolve) = _one_of_params(c)
_child_params(c::Compete) = _hazard_one_of_params(c)
_child_params(c::Choose) = _select_params(c)
_child_params(c) = params(c)

# Any OTHER composer node's nested params: name-keyed, recursing into each
# child through `_child_params` (which falls through to plain `params` for a
# leaf child, and back into this method for a further-nested custom node).
# Mirrors `_composed_params` above, generalised off `node_children`/
# `component_names` rather than the `.components` field.
function params(d::AbstractComposedDistribution)
    names = component_names(d)
    vals = map(_child_params, node_children(d))
    return NamedTuple{names}(vals)
end

# A racing-hazard node's nested params: each outcome name -> its delay's params.
# There is no `branch_probs` entry (the winning probability is derived).
function _hazard_one_of_params(c::Compete)
    return NamedTuple{component_names(c)}(map(params, c.delays))
end

# A `Resolve` node's nested params: each outcome name -> its delay's params,
# plus a `branch_probs` entry carrying the outcome probabilities (a fixed node)
# or the K-1 stick coordinates (an uncertain node), so the codec flattens the
# same estimated representation the table lists.
function _one_of_params(c::Resolve)
    outcome_vals = map(params, c.delays)
    outcomes = NamedTuple{component_names(c)}(outcome_vals)
    return merge(outcomes, (; branch_probs = _branch_prob_params(c)))
end

# The nested-params entry for the branch probabilities: the raw probability
# tuple for a fixed node, or the K-1 stick coordinates (keyed `:stick_k`) when
# the node carries an attached `Dirichlet`, so `flatten`/`unflatten` round-trip
# the estimated stick subset.
_branch_prob_params(c::Resolve) = _branch_prob_params(c, c.branch_prob_prior)
_branch_prob_params(c::Resolve, ::Nothing) = c.branch_probs
function _branch_prob_params(c::Resolve, ::Distributions.Dirichlet)
    v = _simplex_to_stick(collect(c.branch_probs))
    names = _stick_param_names(length(component_names(c)))
    return NamedTuple{names}(Tuple(v))
end

# A `Choose` node's nested params: a NamedTuple keyed by the alternative names,
# each value the alternative's own `_child_params` (recursing into a nested
# composer, delegating to `params` at a leaf). The public `params(::Choose)`
# stays positional (mirroring `params(::Resolve)`), but this name-keyed form is
# what the nested params tree threads when a `Choose` is a child, so a `Choose`
# under a `Sequential`/`Parallel` yields a name-keyed subtree rather than a
# positional tuple. Per-branch params are namespaced per alternative
# (`index.…`/`sourced.…`); a tag shared across alternatives via `shared(:tag,...)`
# still appears once per occurrence here and is inventoried/sampled once by
# `composed_to_table`/the prior model.
function _select_params(d::Choose)
    vals = map(_child_params, d.alternatives)
    return NamedTuple{component_names(d)}(vals)
end

# --- censoring-transparent leaves ------------------------------------------
#
# A composed leaf may itself be a censored delay (e.g. a
# `double_interval_censored(Gamma(...))`, i.e. an
# `IntervalCensored{Truncated{PrimaryCensored{Gamma}}}`). The censoring bounds
# (primary event, truncation, interval) are fixed structure, not free
# parameters; only the inner delay's parameters (the `Gamma` alpha/theta) are
# free. `_free_leaf` peels the fixed censoring off to the inner free delay, and
# `_rewrap_leaf` rebuilds the same censoring around a new inner delay. The
# introspection (`composed_to_table`, names) and reconstruction layers go
# through these, so a censored leaf is transparent: its rows show only the
# inner free params and it round-trips by re-censoring the rebuilt delay. A
# plain leaf is the identity for both. The public `Distributions.params` is
# unchanged.

@doc raw"

Innermost free delay of a (possibly wrapped) leaf.

The base identity contract: a plain leaf is its own free leaf, and a `Truncated`
peels to its untruncated inner delay (the truncation bounds are fixed structure,
not free parameters). A wrapper type (censoring in CensoredDistributions, the
modifiers in ModifiedDistributions) adds its own method dispatching on its own
type, so a composed leaf is transparent to the prior/params interface. Pair with
[`rewrap_leaf`](@ref), which rebuilds the same wrapper around a new inner delay.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution to peel.

# Examples
```@example
using ComposedDistributions, Distributions

free_leaf(truncated(Gamma(2.0, 1.0); upper = 10.0))
```

# See also
- [`rewrap_leaf`](@ref): the inverse rebuild.
- [`inner_dist`](@ref): the single-layer peel hook this recurses through.
"

@doc raw"

The next inner distribution layer of a (possibly wrapped) leaf.

The single peel hook the read-through leaf-wrapper hooks recurse through:
a wrapper type (censoring in CensoredDistributions, the modifiers in
ModifiedDistributions, or ComposedDistributions' own `Truncated`/`Shared`/
`Uncertain`/`Varying`) defines ONE method here returning its inner
distribution, and every read-through hook (`free_leaf`, `uncertain_specs`,
`extra_leaf_params`, `shared_tag`) that simply forwards through the wrapper
gets that forwarding for free. A plain leaf's inner is itself — the base
identity is the terminal that stops the peel.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution.

# Examples
```@example
using ComposedDistributions, Distributions

inner_dist(Gamma(2.0, 1.0)) === Gamma(2.0, 1.0)
inner_dist(truncated(Gamma(2.0, 1.0); upper = 10.0))
```

# See also
- [`free_leaf`](@ref): reach the innermost free delay via this hook.
- [`rewrap_leaf`](@ref): the inverse rebuild.
"
inner_dist(leaf) = leaf
inner_dist(d::Truncated) = d.untruncated

@doc raw"

Innermost free delay of a (possibly wrapped) leaf.

The base identity contract: a plain leaf is its own free leaf, and a wrapper
peels one [`inner_dist`](@ref) layer at a time to the inner free delay. A
wrapper type (censoring in CensoredDistributions, the modifiers in
ModifiedDistributions) defines [`inner_dist`](@ref) on its own type and the
peel follows automatically. Pair with [`rewrap_leaf`](@ref), which rebuilds
the same wrapper around a new inner delay.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution to peel.

# Examples
```@example
using ComposedDistributions, Distributions

free_leaf(truncated(Gamma(2.0, 1.0); upper = 10.0))
```

# See also
- [`rewrap_leaf`](@ref): the inverse rebuild.
- [`inner_dist`](@ref): the single-layer peel hook this recurses through.
"
@inline function free_leaf(leaf)
    inner = inner_dist(leaf)
    return inner === leaf ? leaf : free_leaf(inner)
end

@doc raw"

Rebuild the same wrapper around a new inner delay `inner`.

The inverse of [`free_leaf`](@ref): the base identity returns `inner`, and a
`Truncated` re-applies its bounds around the rebuilt inner delay. A wrapper type
adds its own method on its own type, so `rewrap_leaf(leaf, free_leaf_of_new)`
carries the fixed structure across a parameter update.

# Arguments
- `leaf`: the wrapped leaf whose fixed structure is re-applied.
- `inner`: the new inner delay to wrap.

# Examples
```@example
using ComposedDistributions, Distributions

rewrap_leaf(truncated(Gamma(2.0, 1.0); upper = 10.0), Gamma(3.0, 1.5))
```

# See also
- [`free_leaf`](@ref): peel to the inner free delay.
"
rewrap_leaf(leaf, inner) = inner
function rewrap_leaf(d::Truncated, inner)
    return truncated(rewrap_leaf(d.untruncated, inner); lower = d.lower,
        upper = d.upper)
end

@doc raw"

The constructor that rebuilds a leaf's free delay from a positional tuple of
parameter values, in [`leaf_param_names`](@ref) order (excluding any trailing
[`extra_leaf_params`](@ref), which `_update_leaf` splits off and re-attaches
around the rebuild).

The base identity returns the inner delay's type constructor, which is right
for a Distributions.jl family whose `params` are its constructor arguments.

A leaf type whose free parameters are not its native constructor arguments
needs a different rebuild. [`rebuild_leaf`](@ref) is the extension point for
that — an ordinary method taking the leaf and the value tuple, with no
constructor-identity contract to keep — and should be overridden first. Only
override `leaf_ctor` itself when the constructor value is also needed on its
own, e.g. as half of a tied group's identity (see [`leaf_signature`](@ref)).

# Arguments
- `leaf`: the leaf whose free delay is rebuilt.

# Examples
```@example
using ComposedDistributions, Distributions

ctor = ComposedDistributions.leaf_ctor(Gamma(2.0, 1.0))
ctor(3.0, 1.5)
```

# See also
- [`rebuild_leaf`](@ref): the value-tuple rebuild `_update_leaf` actually
  calls; override this first for a non-native leaf.
- [`leaf_signature`](@ref): the tie-identity pairing this constructor feeds.
- [`free_leaf`](@ref): peel to the inner free delay.
- [`rewrap_leaf`](@ref): re-apply the fixed structure around a rebuilt delay.
"
function leaf_ctor(leaf)
    inner = free_leaf(leaf)
    # A bare leaf: its own type constructor rebuilds it. Reached only when no
    # override applies, since an override is the more specific method.
    inner === leaf && return Base.typename(typeof(leaf)).wrapper
    # A wrapper (`Truncated`, `Uncertain`, `Shared`, a censored or modified
    # leaf): recurse rather than read the peeled type directly, so an inner
    # leaf's override is honoured through the wrapper. `_leaf_param_names` peels
    # and then dispatches `param_names` for the same reason; the two must agree
    # or a wrapped leaf would report one set of names and rebuild from another.
    return leaf_ctor(inner)
end

# The underscored alias retained for the package's existing internal callers
# and the leaf-wrapper method definitions (censoring / modifiers); `const`
# makes it the same function object, so dropping the underscore is
# source-compatible.
const _leaf_ctor = leaf_ctor

@doc raw"

Rebuild a leaf's free delay from a positional tuple of parameter values, in
[`leaf_param_names`](@ref) order (excluding any trailing
[`extra_leaf_params`](@ref), which `_update_leaf` splits off and re-attaches
around the rebuild).

This is the reconstruction hook `_update_leaf` actually calls: `update`, the
`unflatten` then `update` posterior read-back, `uncertain`'s pinning path, and
a tied leaf's rebuild all go through it. The default peels one
[`free_leaf`](@ref) layer at a time and recurses, so an override on the inner
leaf is reached through any wrapper (`Truncated`, `Uncertain`, `Shared`, a
censored or modified leaf), exactly like [`leaf_ctor`](@ref)'s own peeling.
Once the leaf is its own free delay, the default composes `leaf_ctor` with the
value tuple, `leaf_ctor(leaf)(vals...)`, so a native-family leaf needs no
method here.

A leaf type whose free parameters are not its native constructor arguments —
a moment-parameterised wrapper reporting a mean and a standard deviation
rather than a shape and a scale, or a type carrying its family in a type
parameter the bare constructor cannot infer positionally — overrides this
directly instead of `leaf_ctor`. Unlike `leaf_ctor`, `rebuild_leaf` carries no
egal-stability contract (see [`leaf_signature`](@ref) for that one): an
ordinary method is enough, no callable struct required. Because the default
peels down to the free delay before dispatching, such an override is honoured
whether the leaf is used bare or under `truncated`, `uncertain`, `shared`, or
any other wrapper.

# Arguments
- `leaf`: the leaf whose free delay is rebuilt.
- `vals`: the leaf's native parameter values, positional, in
  [`leaf_param_names`](@ref) order.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.rebuild_leaf(Gamma(2.0, 1.0), (3.0, 1.5))
```

# See also
- [`leaf_ctor`](@ref): the constructor this default rebuild composes.
- [`leaf_signature`](@ref): the tie-identity pairing, kept separate from
  reconstruction.
"
function rebuild_leaf(leaf, vals::Tuple)
    inner = free_leaf(leaf)
    inner === leaf && return leaf_ctor(leaf)(vals...)
    return rebuild_leaf(inner, vals)
end

@doc raw"

Leaf-level distribution-valued parameter specs, or `nothing` for a fixed leaf.

The uncertain-spec protocol: a `NamedTuple` of a leaf's distribution-valued
parameters (its attached priors), keyed by parameter name, or `nothing` when
the leaf carries no attached prior. The base identity returns `nothing`, and a
`Truncated` peels to its untruncated inner delay's specs (the truncation
bounds are fixed structure, not free parameters). [`Uncertain`](@ref) reports
its own specs (see Uncertain.jl), and a leaf-wrapper type (censoring in
CensoredDistributions, the modifiers in ModifiedDistributions) adds its own
method dispatching on its own type and forwarding to its inner delay's specs,
so an uncertain prior attached under a wrapper still reaches
[`composed_to_table`](@ref)'s `prior` column and [`build_priors`](@ref). Without a
forwarding method the attached prior is silently dropped and the parameter is
treated as fixed.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution to inspect.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.uncertain_specs(Gamma(2.0, 1.0)) === nothing
```

# See also
- [`free_leaf`](@ref), [`rewrap_leaf`](@ref): the sibling leaf-wrapper hooks.
- [`leaf_detail_lines`](@ref): the sibling extension hook for `inspect`
  rendering.
- [`has_uncertain`](@ref): the boolean check built on this protocol.
"
@inline function uncertain_specs(leaf)
    inner = inner_dist(leaf)
    return inner === leaf ? nothing : uncertain_specs(inner)
end

# The underscored alias retained for the package's existing internal callers
# and the leaf-wrapper method definitions (censoring / modifiers); `const`
# makes it the same function object, so dropping the underscore is
# source-compatible.
const _uncertain_specs = uncertain_specs

@doc raw"
The extra, modifier-owned parameters of a leaf, keyed by name.

A `NamedTuple` mapping each extra-parameter name to a `(value, support)`
`NamedTuple`: `value` is the parameter's current value and `support` the
`(lower, upper)` bounds a default prior is derived from. The default (a plain
leaf, no extras) is the empty `NamedTuple` `(;)`, and a `Truncated` peels to its
untruncated inner delay. A modifier layer that owns a free parameter which is
not one of the inner delay's native parameters plugs in by defining this on its
own wrapper type. The thinning factor of `thin(d, p)` (ModifiedDistributions'
`ThinOp`) is the first instance: it reports `(thin = (value = p, support =
(0.0, 1.0)),)`, at which point [`composed_to_table`](@ref) surfaces a `:thin`
row and [`update`](@ref) round-trips it.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution to inspect.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.extra_leaf_params(Gamma(2.0, 1.0))
```

# See also
- [`set_extra_leaf_params`](@ref): the setter dual that rebuilds the leaf.
- [`leaf_param_names`](@ref): appends the extra names after the native ones.
"
@inline function extra_leaf_params(leaf)
    inner = inner_dist(leaf)
    return inner === leaf ? (;) : extra_leaf_params(inner)
end

@doc raw"
Set a leaf's extra, modifier-owned parameters by name and rebuild the leaf.

The setter dual of [`extra_leaf_params`](@ref): `vals` is a `NamedTuple` mapping
each extra name to a new value (the support is fixed structure, not passed), and
the leaf is rebuilt carrying the updated values. The default no-extras method is
the identity on the empty `NamedTuple`, and a `Truncated` re-applies its bounds
around the rebuilt inner delay. A modifier layer that owns an extra parameter
defines this on its own wrapper type, rebuilding around the new value.

# Arguments
- `leaf`: the leaf whose extra parameters are set.
- `vals`: a `NamedTuple` of extra name to new value.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.set_extra_leaf_params(Gamma(2.0, 1.0), (;))
```

# See also
- [`extra_leaf_params`](@ref): reads the extra parameters and their supports.
"
set_extra_leaf_params(leaf, ::NamedTuple{()}) = leaf
set_extra_leaf_params(d::Truncated, ::NamedTuple{()}) = d
function set_extra_leaf_params(d::Truncated, vals::NamedTuple)
    return truncated(set_extra_leaf_params(d.untruncated, vals);
        lower = d.lower, upper = d.upper)
end

# Whether a leaf distribution constructor `ctor` accepts a `check_args` keyword for
# the sampled value tuple `vals`. Dormant reflection for a future leaf
# reconstruction (issue #9) that would skip the argument check (so a sampler
# probing an out-of-support point yields `-Inf` rather than throwing
# mid-gradient) only where the family supports it. Pure reflection returning a
# `Bool` (constant w.r.t. the params), so
# `ComposedDistributionsMooncakeExt` shields it with a Mooncake `@zero_adjoint`,
# keeping the reconstruction AD-safe under Mooncake reverse.
function _ctor_has_check_args(ctor, vals::Tuple)
    return hasmethod(ctor, typeof(vals), (:check_args,))
end

# --- parameter-name introspection for leaves -------------------------------

# Greek-letter fieldnames (`α`, `θ`, ...) transliterated to their English
# spelling, generation-time-only (called from inside `_derived_param_names`'s
# `@generated` body, never at runtime). Julia's parser already normalises
# identifiers to NFC, so no `Unicode` dependency is needed here; every
# fieldname of every constructible exported univariate family in
# Distributions.jl is either plain ASCII or one of these single characters.
# Anything not in the map passes through unchanged -- a non-ASCII `Symbol` is a
# legal `NamedTuple` key, so pass-through is safe, just less friendly.
const _GREEK_TO_LATIN = Dict{Char, String}(
    'α' => "alpha", 'β' => "beta", 'γ' => "gamma", 'δ' => "delta",
    'ε' => "epsilon", 'ζ' => "zeta", 'η' => "eta", 'θ' => "theta",
    'ι' => "iota", 'κ' => "kappa", 'λ' => "lambda", 'μ' => "mu",
    'ν' => "nu", 'ξ' => "xi", 'ο' => "omicron", 'π' => "pi",
    'ρ' => "rho", 'σ' => "sigma", 'ς' => "sigma", 'τ' => "tau",
    'υ' => "upsilon", 'φ' => "phi", 'χ' => "chi", 'ψ' => "psi",
    'ω' => "omega",
    'Α' => "Alpha", 'Β' => "Beta", 'Γ' => "Gamma", 'Δ' => "Delta",
    'Ε' => "Epsilon", 'Ζ' => "Zeta", 'Η' => "Eta", 'Θ' => "Theta",
    'Ι' => "Iota", 'Κ' => "Kappa", 'Λ' => "Lambda", 'Μ' => "Mu",
    'Ν' => "Nu", 'Ξ' => "Xi", 'Ο' => "Omicron", 'Π' => "Pi",
    'Ρ' => "Rho", 'Σ' => "Sigma", 'Τ' => "Tau", 'Υ' => "Upsilon",
    'Φ' => "Phi", 'Χ' => "Chi", 'Ψ' => "Psi", 'Ω' => "Omega")

# Transliterate one fieldname, generation-time-only. Plain ASCII names pass
# straight through with no allocation.
function _transliterate_fieldname(s::Symbol)
    str = String(s)
    isascii(str) && return s
    buf = IOBuffer()
    for c in str
        write(buf, get(_GREEK_TO_LATIN, c, string(c)))
    end
    return Symbol(String(take!(buf)))
end

# The type-domain derivation. `D` is the leaf type, `P` the type of
# `params(leaf)` (a `Tuple` type); both are `Const` at every call site, so the
# body below -- Base type reflection only, nothing user-extensible -- runs
# once per `(D, P)` pair at specialization time and the generated method body
# is a literal tuple of `Symbol`s. `fieldcount(P)` (not `length(params(d))`)
# is what keeps the arity in the type domain.
#
# Falls back to positional `:param_1, :param_2, ...` whenever the leaf's own
# fields don't line up 1:1 with its `params` in order: too few fields, a
# fieldtype mismatch at some slot, or (after transliteration) a name collision.
# That fallback is intentionally silent -- see `param_names`'s docstring for
# how a leaf whose fields do not line up overrides this.
@generated function _derived_param_names(::Type{D}, ::Type{P}) where {D, P}
    n = fieldcount(P)
    positional = ntuple(i -> Symbol(:param_, i), n)
    n == 0 && return :(())
    fns = fieldnames(D)
    length(fns) < n && return :($positional)
    for i in 1:n
        fieldtype(D, i) === fieldtype(P, i) || return :($positional)
    end
    names = ntuple(i -> _transliterate_fieldname(fns[i]), n)
    length(unique(names)) == n || return :($positional)
    return :($names)
end

@doc raw"

The scalar parameter names of a leaf distribution, matched positionally to
`params(leaf)`.

Distributions.jl exposes parameter values through `params` but not their names,
so the names are derived from the leaf's own type: the first `N` fieldnames of
`typeof(leaf)`, transliterated (Greek letters to their English spelling), where
`N` is the arity of `params(leaf)`. This works with no method of its own for
any type whose fields line up 1:1 with `params`, in order -- the common case for
a Distributions.jl-conforming family. A leaf whose fields don't line up (either
because the arity is short, or because a field's declared type disagrees with
the matching `params` slot, or because two transliterated names collide) falls
back to `:param_1, :param_2, ...`. A wrapper (`Truncated`, `Uncertain`,
`Shared`, a censoring/modifier layer) reports its inner leaf's names, read
through [`inner_dist`](@ref).

A leaf type whose free parameters are not aligned with its own fields overrides
this explicitly, in step with [`leaf_ctor`](@ref): the two together fix the
coordinates that `composed_to_table`, `uncertain`, `build_priors` and the flat
codec work in. A moment-parameterised wrapper naming a mean and a standard
deviation, rather than a shape and a scale, is the motivating case.

# Arguments
- the leaf distribution whose parameter names are read.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.param_names(Gamma(2.0, 1.0))
```

# See also
- [`leaf_ctor`](@ref): the matching rebuild.
- [`inner_dist`](@ref): the peel a wrapper's own names follow.
"
function param_names(d)
    inner = inner_dist(d)
    inner === d || return param_names(inner)
    return _derived_param_names(typeof(d), typeof(params(d)))
end

# The underscored alias retained for the package's existing internal callers
# and the leaf-wrapper method definitions (censoring / modifiers); `const`
# makes it the same function object, so dropping the underscore is
# source-compatible.
const _param_names = param_names

@doc raw"
The estimable parameter names of a (possibly wrapped) leaf.

The inner free delay's `param_names`, padding with positional fallbacks
(`:param_1`, ...) so every value has a label even when the leaf's fields don't
line up 1:1 with its `params`, then the names of any
[`extra_leaf_params`](@ref) appended in order. A censored or modified leaf
delegates to its free delay (`free_leaf`), so the fixed wrapper structure
never appears, while a thinning modifier's `:thin` factor rides the trailing
extra-parameter slot. These names are the coordinates
[`composed_to_table`](@ref), [`uncertain`](@ref) and [`build_priors`](@ref)
key on.

# Arguments
- `leaf`: the (possibly wrapped) leaf distribution whose parameter names are
  read.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.leaf_param_names(Gamma(2.0, 1.0))
```

# See also
- [`extra_leaf_params`](@ref): the extra names appended after the native ones.
"
function leaf_param_names(leaf)
    inner = free_leaf(leaf)
    vals = params(inner)
    base = param_names(inner)
    n = length(vals)
    names = ntuple(n) do i
        i <= length(base) ? base[i] : Symbol(:param_, i)
    end
    # Any modifier-owned extra parameters (e.g. a thinned leaf's `:thin` factor)
    # append after the delay params, in `extra_leaf_params` order.
    return (names..., keys(extra_leaf_params(leaf))...)
end
const _leaf_param_names = leaf_param_names

@doc raw"

The `(constructor, parameter-names)` signature [`tie`](@ref) groups leaves by:
tied leaves become one free parameter, so they must share a rebuild and a
parameter structure. The default pairs [`leaf_ctor`](@ref) with
[`leaf_param_names`](@ref), peeling one [`free_leaf`](@ref) layer at a time so
the constructor half is read from the inner leaf's own signature (honouring an
override there) while the parameter-names half stays the outer leaf's
[`leaf_param_names`](@ref) — a wrapper's own name set, including any
[`extra_leaf_params`](@ref) it adds, which a naive full peel would drop.

An override must return an **egal-stable** value: two structurally identical
leaves must return `==`-equal signatures, since a tie compares them. In
particular a constructor closing over a runtime *value* is not egal-stable —
two structurally identical leaves would build two different, unequal,
closures — and would make `tie` wrongly reject two compatible leaves. A
callable closing over a type parameter is egal-stable; prefer a callable
struct over an anonymous closure when a signature needs to carry one. A leaf
that overrides [`rebuild_leaf`](@ref) instead of `leaf_ctor` usually overrides
this too, pairing the leaf's family (a type or a callable struct) with its
parameter names.

# Arguments
- `leaf`: the leaf whose tie-identity signature is read.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.leaf_signature(Gamma(2.0, 1.0))
```

# See also
- [`leaf_ctor`](@ref), [`rebuild_leaf`](@ref): the reconstruction hooks a
  non-native leaf's signature is built from, or replaces.
- [`tie`](@ref): the leaf-grouping verb keyed on this signature.
"
function leaf_signature(leaf)
    inner = free_leaf(leaf)
    fam = inner === leaf ? leaf_ctor(leaf) : first(leaf_signature(inner))
    return (fam, leaf_param_names(leaf))
end

# A diagnostic, never a name selector: `_derived_param_names` decides labels
# from the type alone (unconditionally, so names never depend on values), and
# this catches the one thing the type-level rule cannot see -- a
# contract-conforming type whose field ORDER differs from its `params` order
# with matching field *types* (so the type-level structural check passes but
# the field at slot `i` does not actually hold `params(free)[i]`). Restricted
# to `Real`-valued slots so a vector-parameter leaf cannot trip it spuriously.
# Called where the values are already in hand (`Uncertain`'s inner
# constructor) and by `TestUtils`'s conformance check; never from
# `param_names`/`leaf_param_names`/`_update_leaf`/`_walk_rows!`, which must
# stay on the gradient path with no runtime branch to shield from Mooncake. A
# plain `throw` (never `@warn`, never `try`/`catch`) keeps it AD-safe even if
# it is ever reached from a differentiated path.
function _check_leaf_param_alignment(free)
    fs = fieldnames(typeof(free))
    tvals = params(free)
    n = length(tvals)
    n == 0 && return nothing
    length(fs) < n && return nothing
    for i in 1:n
        v = tvals[i]
        v isa Real || continue
        getfield(free, fs[i]) === v && continue
        throw(ArgumentError(
            "$(typeof(free)) has parameters $(tvals) whose order does not " *
            "match its field order $(fs[1:n]); define " *
            "`ComposedDistributions.param_names` (and `leaf_ctor` in step " *
            "with it) for $(typeof(free))"))
    end
    return nothing
end

# The values matching `leaf_param_names(leaf)` positionally: the free delay's
# native `params`, then each `extra_leaf_params` value, in the same order
# `leaf_param_names` appends their names. Built entirely from the generic,
# instance-level hooks (`free_leaf`, `Distributions.params`,
# `extra_leaf_params`), so it is correct for any leaf type an extension
# defines those on, with no registration step -- the seam `_leaf_entry` and
# `_leaf_flatten_values` (below) read from for the flat codec's leaf case
# (`unflatten`, `flatten`), with no type-level table to keep in sync.
function leaf_param_values(leaf)
    native = params(free_leaf(leaf))
    extras = extra_leaf_params(leaf)
    return (native..., map(v -> v.value, values(extras))...)
end

# A leaf's `unflatten` entry: `leaf_param_names(leaf)` walked in order, taking
# the next value from `slots` for each name present in `speckeys` (an
# `Uncertain` node's estimated parameter names, fixed as a `Val` type
# parameter) and the leaf's own current value from `leaf_param_values`
# otherwise. `speckeys` membership decides *whether* a name is substituted;
# `slots` is consumed strictly in `leaf_param_names` order, NOT `speckeys`'s
# own order (`speckeys` holds the constructor's kwargs order, which can
# differ), so this reproduces the layout the generated codec builds by
# walking the same `leaf_param_names` order and advancing its slot index on
# each match.
function _leaf_entry(leaf, ::Val{speckeys}, slots::Tuple) where {speckeys}
    names = leaf_param_names(leaf)
    vals = leaf_param_values(leaf)
    return NamedTuple{names}(_leaf_substitute(names, vals, speckeys, slots))
end

@inline _leaf_substitute(::Tuple{}, ::Tuple{}, ::Tuple, ::Tuple) = ()
@inline function _leaf_substitute(names::Tuple, vals::Tuple, speckeys::Tuple,
        slots::Tuple)
    hit = names[1] in speckeys
    v = hit ? slots[1] : vals[1]
    rest = hit ? Base.tail(slots) : slots
    return (v, _leaf_substitute(Base.tail(names), Base.tail(vals), speckeys,
        rest)...)
end

# A `Pool`-noncentred entry's dual: after `_leaf_entry` builds the leaf's
# `(native..., extra...)` NamedTuple, wrap whichever of its fields are
# `Pool`-noncentred parameters in a `(z = value,)` sub-NamedTuple, by NAME
# (`pool_names`, a `Val` of the noncentred spec'd names -- generation-time
# information from `speckeys`/`specvaltypes` alone, no `leaf_param_names`
# order needed). Kept as a separate pass over `_leaf_entry`'s own tested
# 3-argument contract (introspection.jl, S1) rather than folded into it, since
# `_leaf_entry` has no notion of `Pool` at all -- codec_gen.jl's
# `_leaf_unflatten_expr` is the only caller that needs the wrap.
function _wrap_pool_entries(
        entry::NamedTuple{names}, ::Val{pool_names}) where {names, pool_names}
    return NamedTuple{names}(_wrap_pool_vals(names, entry, pool_names))
end

@inline _wrap_pool_vals(::Tuple{}, ::NamedTuple, ::Tuple) = ()
@inline function _wrap_pool_vals(
        names::Tuple, entry::NamedTuple, pool_names::Tuple)
    pname = names[1]
    raw = getproperty(entry, pname)
    v = pname in pool_names ? (z = raw,) : raw
    return (v, _wrap_pool_vals(Base.tail(names), entry, pool_names)...)
end

# The read-direction counterpart of `_leaf_entry`/`_wrap_pool_entries`:
# `flatten`'s leaf case. `entry` is the leaf's own already-`unflatten`ed
# NamedTuple sub-tree (keyed by `leaf_param_names(leaf)`, native then extra);
# this extracts the `speckeys`-named values back out, in `leaf_param_names`
# order (`_leaf_entry`'s own substitution order, so the two stay inverse),
# unwrapping a `Pool`-noncentred entry's `z` field the same way
# `_wrap_pool_entries` wrapped it.
#
# The walk order comes from `entry`'s own `keys` rather than from a
# `leaf_param_names(leaf)` call. The two are the same sequence -- `_leaf_entry`
# builds `entry` as `NamedTuple{leaf_param_names(leaf)}` and
# `_wrap_pool_entries` preserves it -- but `entry`'s copy is carried in its
# type, so the spec'd/fixed test below resolves by dispatch. Driven off the
# runtime call instead, inference could not settle how many names pass the
# test and returned a `Union` over result lengths, which propagated out to
# `flatten` as a `Union{Vector{Any}, Vector{Float64}}` return type. `leaf` is
# still taken (unused) so the seam's signature stays the one `codec_gen.jl`
# emits and `leaf_entry_seam.jl` pins.
function _leaf_flatten_values(leaf, ::Val{speckeys}, ::Val{pool_names},
        entry::NamedTuple{names}) where {speckeys, pool_names, names}
    return _leaf_extract(Val(names), Val(speckeys), Val(pool_names), entry)
end

@inline _leaf_extract(::Val{()}, ::Val, ::Val, ::NamedTuple) = ()
@inline function _leaf_extract(
        ::Val{names}, ::Val{speckeys}, ::Val{pool_names},
        entry::NamedTuple) where {names, speckeys, pool_names}
    pname = names[1]
    rest = _leaf_extract(
        Val(Base.tail(names)), Val(speckeys), Val(pool_names), entry)
    pname in speckeys || return rest
    raw = getproperty(entry, pname)
    v = pname in pool_names ? raw.z : raw
    return (v, rest...)
end

# --- node emission (#227 slice 1) -------------------------------------------
#
# The single pre-order walk (`_walk_rows!` below) emits a `:node` row for
# every composer node and every leaf (wrapper) layer, alongside the existing
# `:param` rows. `node_attributes` is the one hook a downstream node or
# leaf-wrapper type defines to control those rows. The other two ingredients
# are derived rather than asked for: a row's structural label is the type's
# own name (`_node_kind`), and a leaf's wrapper layers fold out of the
# `inner_dist` peel (`_leaf_layers`) a wrapper already defines to take part in
# the leaf protocol at all.

# The structural label of a composer node or leaf (wrapper) layer: the `node`
# column of `composed_to_table`. Read straight off the type, so a `Gamma` leaf
# reports `:Gamma` and a `Sequential` node `:Sequential`. Correct by
# construction for every type that can appear in a tree, which is why it is an
# accessor and not a hook a downstream type has to supply.
_node_kind(x) = Base.typename(typeof(x)).name

# A (possibly wrapped) leaf's wrapper layers, outermost to innermost, folded
# out of the single-layer `inner_dist` peel: `truncated(shared(:inc, Gamma(...)))`
# gives its `Truncated`, `Shared` and `Gamma` layers in that order, each then
# emitting its own `:node` row (and any `node_attributes`) at the leaf's one
# edge. The fold terminates at a plain leaf, whose inner is itself, so a
# wrapper with no `inner_dist` method appears as one opaque node row.
function _leaf_layers(x)
    inner = inner_dist(x)
    return inner === x ? (x,) : (x, _leaf_layers(inner)...)
end

@doc raw"

A node or leaf (wrapper) layer's own fixed-structure attributes, for the
`:attribute` rows of [`composed_to_table`](@ref).

Returns a `NamedTuple` of name-value pairs that are fixed structure, not free
parameters (a `Truncated`'s bounds, a `Choose`'s `selector`, a `Shared`'s
tag): each entry becomes one `:attribute` row at the node's edge. The default
is the empty `NamedTuple` `(;)` (no attributes), which is right for a leaf
family whose parameters are already fully covered by the `:param` rows. A
node or leaf-wrapper type with fixed, non-parameter structure overrides this
on its own type.

This is the only node-emission method a downstream type supplies. A row's
structural label is read off the type name, and a wrapped leaf's layers are
peeled through [`inner_dist`](@ref), so neither needs a method of its own.

# Arguments
- `x`: the composer node or (possibly wrapped) leaf layer.

# Examples
```@example
using ComposedDistributions, Distributions

ComposedDistributions.node_attributes(Gamma(2.0, 1.0))
ComposedDistributions.node_attributes(truncated(Gamma(2.0, 1.0); upper = 10.0))
```

# See also
- [`composed_to_table`](@ref): the table this feeds.
- [`inner_dist`](@ref): the peel a wrapper layer's own rows follow.
"
node_attributes(x) = (;)

# `Truncated`'s bounds are fixed structure, not free parameters (its inner
# delay's own parameters ride the `:param` rows as always).
node_attributes(d::Truncated) = (; lower = d.lower, upper = d.upper)

# --- row sinks: a single walk feeding two projections -----------------------
#
# `_walk_rows!` threads one sink object rather than five positional column
# vectors, so the same pre-order walk feeds either the five-column
# parameter-only sink (`_ParamSink`, the walk internal callers run directly
# for a zero-extra-work parameter inventory) or the full seven-column
# node/attribute/param table ([`composed_to_table`](@ref)) with no extra
# traversal. The `_ParamSink` node/attribute push methods are no-ops
# (`@inline`), so a `_ParamSink` walk costs nothing extra on its AD-adjacent
# call sites (e.g. `centred_pool_rows`, called once per gradient evaluation
# for a centred-pooled tree) — this is materialise-once-per-sink, never
# materialise-then-filter.

struct _ParamSink
    edge::Vector{Symbol}
    param::Vector{Symbol}
    value::Vector{Any}
    support::Vector{Any}
    prior::Vector{Any}
end
_ParamSink() = _ParamSink(Symbol[], Symbol[], Any[], Any[], Any[])

struct _FullSink
    edge::Vector{Symbol}
    param::Vector{Symbol}
    node::Vector{Symbol}
    role::Vector{Symbol}
    value::Vector{Any}
    support::Vector{Any}
    prior::Vector{Any}
end
function _FullSink()
    return _FullSink(
        Symbol[], Symbol[], Symbol[], Symbol[], Any[], Any[], Any[])
end

# One `:param` row: a scalar free parameter. `node` names the leaf family
# (or `:Pool`/`:Resolve`) the parameter belongs to; ignored by `_ParamSink`.
function _push_param!(s::_ParamSink, edge::Symbol, param::Symbol,
        ::Symbol, value, support, prior)
    push!(s.edge, edge)
    push!(s.param, param)
    push!(s.value, value)
    push!(s.support, support)
    push!(s.prior, prior)
    return nothing
end
function _push_param!(s::_FullSink, edge::Symbol, param::Symbol, node::Symbol,
        value, support, prior)
    push!(s.edge, edge)
    push!(s.param, param)
    push!(s.node, node)
    push!(s.role, :param)
    push!(s.value, value)
    push!(s.support, support)
    push!(s.prior, prior)
    return nothing
end

# One `:node` row, naming a composer node or a leaf (wrapper) layer; no
# value/support/prior. A no-op on the parameter-only sink.
@inline _push_node!(::_ParamSink, ::Symbol, ::Symbol) = nothing
function _push_node!(s::_FullSink, edge::Symbol, node::Symbol)
    push!(s.edge, edge)
    push!(s.param, Symbol(""))
    push!(s.node, node)
    push!(s.role, :node)
    push!(s.value, nothing)
    push!(s.support, nothing)
    push!(s.prior, nothing)
    return nothing
end

# One `:attribute` row: a node/layer's own fixed-structure attribute (not a
# free parameter). A no-op on the parameter-only sink.
@inline function _push_attr!(::_ParamSink, ::Symbol, ::Symbol, ::Symbol,
        value)
    nothing
end
function _push_attr!(s::_FullSink, edge::Symbol, node::Symbol, param::Symbol,
        value)
    push!(s.edge, edge)
    push!(s.param, param)
    push!(s.node, node)
    push!(s.role, :attribute)
    push!(s.value, value)
    push!(s.support, nothing)
    push!(s.prior, nothing)
    return nothing
end

# Push one `:attribute` row per entry of a `node_attributes(x)` NamedTuple, in
# field order.
function _push_attr_rows!(sink, edge::Symbol, node::Symbol, attrs::NamedTuple)
    for k in keys(attrs)
        _push_attr!(sink, edge, node, k, attrs[k])
    end
    return nothing
end

# A leaf's `:node`/`:attribute` rows, one pair per `_leaf_layers(leaf)` layer.
# A no-op on `_ParamSink`: `_leaf_layers` peels a leaf-wrapper stack (e.g.
# `Truncated`) that only the full table shows, so a parameter-only walk's hot
# path must not build or iterate it at all, not just discard the rows it
# would produce.
@inline _emit_layers!(::_ParamSink, ::Symbol, leaf) = nothing
function _emit_layers!(sink::_FullSink, edge_path::Symbol, leaf)
    for layer in _leaf_layers(leaf)
        kind = _node_kind(layer)
        _push_node!(sink, edge_path, kind)
        _push_attr_rows!(sink, edge_path, kind, node_attributes(layer))
    end
    return nothing
end

# --- composed_to_table (hand-rolled pre-order walk) -------------------------

# A thin wrapper over the flat column table so `composed_to_table(d)` prints as
# an actual table rather than as a bare `NamedTuple` of vectors, while staying a
# first-class Tables.jl source. It forwards the whole Tables.jl column interface
# to the wrapped `NamedTuple`, so `Tables.istable`, `Tables.columns`,
# `Tables.getcolumn` and `DataFrame(tbl)` all work unchanged, and `getproperty`
# forwards `tbl.edge`/`tbl.param`/... to the columns. Only the
# `show(::MIME"text/plain")` is customised, to render a padded ASCII table.

@doc "

A Tables.jl column table of a composed distribution's structure.

The value [`composed_to_table`](@ref) returns: a Tables.jl source (a column
table) that prints as a padded `edge | param | node | role | value | support |
prior` table. It is a thin wrapper over a `NamedTuple` of equal-length column
vectors, forwarding the whole Tables.jl column interface and column access
(`tbl.edge`, `tbl.param`, ...), so `Tables.istable`, `Tables.columns`,
`Tables.getcolumn`, `DataFrame(tbl)` and [`build_priors`](@ref) all consume it
unchanged; only its display is customised.

See also: [`composed_to_table`](@ref), [`build_priors`](@ref).
"
struct ComposedTable{C <: NamedTuple}
    columns::C
end

# Tables.jl source interface: a column table, delegating to the wrapped columns.
Tables.istable(::Type{<:ComposedTable}) = true
Tables.columnaccess(::Type{<:ComposedTable}) = true
Tables.columns(t::ComposedTable) = getfield(t, :columns)
Tables.columnnames(t::ComposedTable) = keys(getfield(t, :columns))
Tables.getcolumn(t::ComposedTable, i::Int) = getfield(t, :columns)[i]
Tables.getcolumn(t::ComposedTable, nm::Symbol) = getfield(t, :columns)[nm]
Tables.schema(t::ComposedTable) = Tables.schema(getfield(t, :columns))
Tables.rowaccess(::Type{<:ComposedTable}) = true
Tables.rows(t::ComposedTable) = Tables.rows(getfield(t, :columns))

# Forward column access (`tbl.edge`, `tbl.param`, ...) to the wrapped columns so
# the table reads like the NamedTuple it wraps.
Base.getproperty(t::ComposedTable, nm::Symbol) = getfield(t, :columns)[nm]
Base.propertynames(t::ComposedTable) = keys(getfield(t, :columns))

# The number of rows (every column is equal length).
function _nrows(t::ComposedTable)
    cols = getfield(t, :columns)
    return isempty(cols) ? 0 : length(first(cols))
end

# A compact one-liner for inline / array display.
function Base.show(io::IO, t::ComposedTable)
    print(io, "ComposedTable($(_nrows(t)) rows)")
    return nothing
end

# A padded ASCII table for `text/plain` display, so `composed_to_table(d)`
# renders as an actual table. Columns are `edge | param | node | role | value |
# support | prior`; each cell is the `string` of the value, columns padded to
# their widest cell (header included).
function Base.show(io::IO, ::MIME"text/plain", t::ComposedTable)
    cols = getfield(t, :columns)
    names = collect(keys(cols))
    n = _nrows(t)
    println(io, "composed table ($n rows)")
    isempty(names) && return nothing
    # Stringify every cell (an absent entry, e.g. a row with no attached
    # prior, renders blank), then size each column to its widest entry.
    cells = [_cell_string.(getindex(cols, nm)) for nm in names]
    headers = string.(names)
    widths = [maximum(length, vcat(headers[j], cells[j]); init = 0)
              for j in eachindex(names)]
    pad(s, w) = s * " "^(w - length(s))
    row(parts) = "  " * join((pad(parts[j], widths[j])
        for j in eachindex(parts)), "  ")
    println(io, row(headers))
    println(io, "  " * join(("─"^w for w in widths), "  "))
    for i in 1:n
        line = row([cells[j][i] for j in eachindex(names)])
        i == n ? print(io, line) : println(io, line)
    end
    return nothing
end

# A blank cell for an absent (`nothing`) entry; `string` otherwise.
_cell_string(x) = x === nothing ? "" : string(x)

@doc "

Flatten a composed distribution's full structure into one node/attribute/
parameter table.

`composed_to_table(d)` returns a Tables.jl column table (a [`ComposedTable`](@ref)
wrapping a `NamedTuple` of equal-length column vectors) with one row per
composer node, per leaf (wrapper) layer, per fixed-structure attribute, and
per scalar free parameter. Columns:

- `edge`: the dotted path of names to the row's node/leaf (e.g. `:onset_admit`),
  or the tag/group for a [`shared`](@ref)/pooled parameter row (see below).
- `param`: the parameter or attribute name; the empty `Symbol` on a `:node`
  row.
- `node`: the type name of the owning composer node or leaf layer (e.g.
  `:Sequential`, `:Gamma`, `:Truncated`).
- `role`: one of `:node`, `:attribute`, `:param`.
- `value`: the current value (a `:param`/`:attribute` row), or `nothing` (a
  `:node` row).
- `support`: the parameter's domain (a `:param` row), else `nothing`.
- `prior`: the attached [`uncertain`](@ref) prior (a `:param` row), else
  `nothing`.

The parameter-only view (the `role == :param` rows, minus the `node`/`role`
columns) is a filter over this table, not a separate function: `filter(row ->
row.role == :param, Tables.rows(composed_to_table(d)))`. `composed_to_table`
recovers structure a parameter-only view alone cannot: whether two branches
with the same names are the same node kind, whether a leaf carries a wrapper
(`Truncated`, `Censored`, ...), and a wrapper's own fixed attributes (a
`Truncated`'s bounds, a `Choose`'s `selector`).

A node row's `edge` is always the row's real dotted tree path. A `:param` row's
`edge` is the real path too, *except* for a [`shared`](@ref)`(:tag, ...)` leaf
(keyed by its tag; later occurrences contribute no further rows, node or
param) or a pooled parameter's hyperparameter rows (keyed by its
[`pool`](@ref) group). So a node row and a `:param` row at the same object can
legitimately disagree on `edge` — do not join the two by `edge` alone; join a
leaf's node rows to its param rows by path prefix instead. A `Resolve`'s own
`branch_probs` param rows are emitted after its children's rows (not
immediately after its own node/attribute rows), so a node's rows are not
generally contiguous in the table.

The composed distribution `d` itself is a Tables.jl source forwarding to this
table (`Tables.columns(d) == Tables.columns(composed_to_table(d))`), so
`DataFrame(tree)` yields the full table; filter its `role` column to `:param`
for the parameter-only projection.

Only an in-memory (DataFrame) round trip is supported: a `Varying` leaf's
`map` attribute and a `Convolved`/`Difference` composite's solver are live
objects, not values a text format (CSV) can carry.

# Arguments
- `d`: the composed distribution to flatten.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = truncated(Gamma(2.0, 1.0); upper = 10.0)))
tbl = composed_to_table(tree)
tbl.node  # :Sequential, :LogNormal, :Truncated, :Gamma, ...
```

# See also
- [`node_attributes`](@ref): the one hook a downstream node or leaf-wrapper
  type defines to control its own rows here.
- [`inner_dist`](@ref): the peel a wrapped leaf's per-layer rows follow.
"
function composed_to_table(d::AbstractComposedDistribution)
    s = _FullSink()
    _walk_rows!(s, Set{Symbol}(), d, ())
    return ComposedTable((edge = s.edge, param = s.param, node = s.node,
        role = s.role, value = s.value, support = s.support, prior = s.prior))
end

# Internal convenience: the historical `params_table(d)` shape, run from the
# `_ParamSink` walk directly (the same zero-extra-work path `centred_pool_rows`
# / `required_parameters` use) rather than by filtering `composed_to_table` —
# an independent computation from the `_FullSink` walk `composed_to_table`
# runs, so the package's own test suite can assert the two stay in lockstep
# (the "golden projection parity" testitem). Kept for the test suite so it
# does not restate the walk at every call site; not part of the public API —
# a user filters `composed_to_table` directly: `filter(row -> row.role ==
# :param, Tables.rows(composed_to_table(d)))`.
function _param_rows(d::AbstractComposedDistribution)
    s = _ParamSink()
    _walk_rows!(s, Set{Symbol}(), d, ())
    return ComposedTable((edge = s.edge, param = s.param,
        value = s.value, support = s.support, prior = s.prior))
end

# --- a composed distribution as a Tables.jl source (#227 slice 1) -----------
#
# A composed tree forwards the whole Tables.jl column interface to its FULL
# table ([`composed_to_table`](@ref), not the parameter-only projection), so
# `DataFrame(tree)` yields the full node/attribute/param table; filter its
# `role` column to `:param` for the parameter-only one. Deliberately no
# `Base.getproperty`/`Base.propertynames` override here (unlike
# [`ComposedTable`](@ref)): a composed distribution's own fields (`.components`,
# `.delays`, ...) must keep working, so only the explicit Tables.jl generic
# functions are forwarded.
Tables.istable(::Type{<:AbstractComposedDistribution}) = true
Tables.columnaccess(::Type{<:AbstractComposedDistribution}) = true
Tables.rowaccess(::Type{<:AbstractComposedDistribution}) = true
Tables.columns(d::AbstractComposedDistribution) = Tables.columns(composed_to_table(d))
function Tables.columnnames(d::AbstractComposedDistribution)
    return Tables.columnnames(composed_to_table(d))
end
function Tables.getcolumn(d::AbstractComposedDistribution, i::Int)
    return Tables.getcolumn(composed_to_table(d), i)
end
function Tables.getcolumn(d::AbstractComposedDistribution, nm::Symbol)
    return Tables.getcolumn(composed_to_table(d), nm)
end
Tables.schema(d::AbstractComposedDistribution) = Tables.schema(composed_to_table(d))
Tables.rows(d::AbstractComposedDistribution) = Tables.rows(composed_to_table(d))

# Pre-order walk over the composer tree. `path` is the tuple of names from the
# root to the current node (the empty tuple at the root, so `_join_path(())`
# `== Symbol("")`; no child name can be empty, so this cannot collide, unlike
# `:root` which would collide with a depth-1 child actually called `root`). A
# composer node emits its own `:node` row, then one `:attribute` row per
# `node_attributes` entry, then recurses into its named children, then any of
# its OWN param rows (a `Resolve`'s `branch_probs`, emitted last — see below).
# A leaf emits one `:node`/`:attribute` row per `_leaf_layers` layer, then one
# `:param` row per scalar free parameter. Hand-rolled recursion to stay
# type-stable.
function _walk_rows!(sink, seen, d::Union{Sequential, Parallel}, path)
    edge = _join_path(path)
    kind = _node_kind(d)
    _push_node!(sink, edge, kind)
    _push_attr_rows!(sink, edge, kind, node_attributes(d))
    names = component_names(d)
    for (name, child) in zip(names, d.components)
        _walk_rows!(sink, seen, child, (path..., name))
    end
    return nothing
end

# A `Choose`'s alternatives each contribute their own rows; a tag shared across
# alternatives is deduped via `seen`, so a parameter tied across the index and
# sourced branches is inventoried once.
function _walk_rows!(sink, seen, d::Choose, path)
    edge = _join_path(path)
    kind = _node_kind(d)
    _push_node!(sink, edge, kind)
    _push_attr_rows!(sink, edge, kind, node_attributes(d))
    for (name, alt) in zip(component_names(d), d.alternatives)
        _walk_rows!(sink, seen, alt, (path..., name))
    end
    return nothing
end

# A `Resolve`'s own `branch_probs` rows are emitted AFTER its children's rows
# (parity with the codec's type-level walk beats a tidy "node, then its own
# rows" ordering — see `_walk_rows!`'s header comment). A `NoEvent` branch
# gets its own `:node` row (so it is visible in `composed_to_table`) and then
# is skipped, unchanged from the params-only behaviour (no param rows).
function _walk_rows!(sink, seen, c::Resolve, path)
    edge = _join_path(path)
    kind = _node_kind(c)
    _push_node!(sink, edge, kind)
    _push_attr_rows!(sink, edge, kind, node_attributes(c))
    for (name, delay) in zip(component_names(c), c.delays)
        child_path = (path..., name)
        if _is_no_event(delay)
            _push_node!(sink, _join_path(child_path), _node_kind(delay))
            continue
        end
        _walk_rows!(sink, seen, delay, child_path)
    end
    bp_edge = _join_path((path..., :branch_probs))
    _branch_prob_rows!(sink, c, bp_edge, c.branch_prob_prior)
    return nothing
end

# The branch-probability rows. A fixed node lists one informational row per
# outcome probability (no attached prior, so it is not estimated). An uncertain
# node (an attached `Dirichlet`) lists its K-1 stick coordinates instead: each a
# scalar estimated parameter `:stick_k` in (0, 1) carrying its stick-breaking
# `Beta` prior, so the codec flattens the simplex through the existing per-row
# scoring with no special-casing.
function _branch_prob_rows!(sink, c::Resolve, edge, ::Nothing)
    sup = (zero(eltype(c.branch_probs)), one(eltype(c.branch_probs)))
    cnames = component_names(c)
    for (k, p) in enumerate(c.branch_probs)
        _push_param!(sink, edge, Symbol(cnames[k]), :Resolve, p, sup, nothing)
    end
    return nothing
end

function _branch_prob_rows!(sink, c::Resolve, edge, prior::Distributions.Dirichlet)
    v = _simplex_to_stick(collect(c.branch_probs))
    betas = _dirichlet_stick_betas(prior)
    names = _stick_param_names(length(component_names(c)))
    for k in eachindex(names)
        _push_param!(sink, edge, names[k], :Resolve, v[k], (0.0, 1.0), betas[k])
    end
    return nothing
end

# A racing-hazard node emits only its outcome delays' parameter rows; there is
# no branch-probability block (the winning probability is derived, not free).
function _walk_rows!(sink, seen, c::Compete, path)
    edge = _join_path(path)
    kind = _node_kind(c)
    _push_node!(sink, edge, kind)
    _push_attr_rows!(sink, edge, kind, node_attributes(c))
    for (name, delay) in zip(component_names(c), c.delays)
        _walk_rows!(sink, seen, delay, (path..., name))
    end
    return nothing
end

# Any OTHER composer node (a downstream type implementing the node-extension
# contract: [`node_children`](@ref) / [`node_rebuild`](@ref) /
# [`component_names`](@ref)) walks generically through those three accessors,
# exactly mirroring the `Sequential`/`Parallel` method above. Dispatch on the
# shared `AbstractComposedDistribution` supertype rather than the closed Union
# above means a node subtyping it with no matching method here falls through
# to THIS method, not silently to the untyped leaf method below — a node
# missing `node_children`/`component_names` gets a `MethodError` naming the
# missing method, not a silent zero-row table (#374).
function _walk_rows!(sink, seen, d::AbstractComposedDistribution, path)
    edge = _join_path(path)
    kind = _node_kind(d)
    _push_node!(sink, edge, kind)
    _push_attr_rows!(sink, edge, kind, node_attributes(d))
    for (name, child) in zip(component_names(d), node_children(d))
        _walk_rows!(sink, seen, child, (path..., name))
    end
    return nothing
end

# Leaf distribution: one `:node`/`:attribute` row per `_leaf_layers` layer,
# then one `:param` row per scalar free parameter. A censored leaf shows only
# its inner free delay's params and that delay's support (the censoring
# bounds are fixed structure, see `free_leaf`); its wrapper layer still gets
# its own node/attribute rows via `_leaf_layers`. A shared-tagged leaf
# (`_shared_tag`) is inventoried once under its tag as the PARAM edge: the
# first occurrence emits the param rows, later occurrences with the same tag
# are skipped so the tied parameter is listed once. The leaf's node/attribute
# rows are emitted BEFORE that dedup check and always use the leaf's real
# path (never the tag), so every shared occurrence gets its own node rows —
# a node row and a param row for the same leaf can legitimately disagree on
# `edge`. An uncertain leaf's attached spec rides each row's `prior` entry
# (`nothing` for a fully fixed parameter), so `build_priors` picks the
# attached prior up without an explicit override.
function _walk_rows!(sink, seen, leaf, path)
    edge_path = _join_path(path)
    _emit_layers!(sink, edge_path, leaf)
    tag = _shared_tag(leaf)
    tag !== nothing && tag in seen && return nothing
    inner = free_leaf(leaf)
    pnames = _leaf_param_names(leaf)
    specs = _uncertain_specs(leaf)
    sup = (minimum(inner), maximum(inner))
    # The native delay params take the inner leaf's own support; each
    # modifier-owned extra parameter (e.g. a thinned leaf's `:thin` factor)
    # appends its current value and its own declared support. With no extras
    # attached this is exactly the plain per-param walk.
    extras = extra_leaf_params(leaf)
    native = params(inner)
    vals = (native..., map(e -> e.value, extras)...)
    sups = (ntuple(_ -> sup, length(native))...,
        map(e -> e.support, extras)...)
    edge = tag === nothing ? edge_path : tag
    tag === nothing || push!(seen, tag)
    leaf_kind = _node_kind(inner)
    for (pname, v, s) in zip(pnames, vals, sups)
        spec = specs === nothing ? nothing : get(specs, pname, nothing)
        # A pooled parameter lowers to the group's shared population
        # hyperparameters (once) plus this member's own latent, all scalar rows.
        if spec isa Pool
            _pool_rows!(sink, seen, spec, edge, pname, v, s)
            continue
        end
        _push_param!(sink, edge, pname, leaf_kind, v, s, spec)
    end
    return nothing
end

# Join a name path to a single dotted `Symbol` (e.g. `(:a, :b)` -> `:a.b`); a
# single-element path keeps its bare name. This is the dotted ("." separator)
# parameter-path namespace (composed_to_table edges / priors), distinct from
# the underscored ("_" separator) event/value namespace (`_join_value_path`,
# `_split_edge_name`).
_join_path(path::Tuple) = Symbol(join(string.(path), "."))

# --- update: nested NamedTuple -> reconstructed distribution ----------------

# Reconstruct a (possibly censored) leaf from a new inner free delay built out
# of `vals`, re-applying the fixed censoring. Mirrors the extension's
# `_reconstruct_leaf` but is Turing-free; argument checks are kept on (this is
# building a concrete distribution, not a gradient hot path).
function _update_leaf(leaf, vals::Tuple)
    # The trailing values are the modifier-owned extra parameters (the trailing
    # rows the walker emits, one per `extra_leaf_params` entry): rebuild the
    # inner delay from the leading native params, then re-attach the updated
    # extras by name. Inert (no split) when the leaf has no extras.
    extras = extra_leaf_params(leaf)
    k = length(extras)
    k == 0 && return rewrap_leaf(leaf, rebuild_leaf(leaf, vals))
    native = vals[1:(end - k)]
    extra_vals = vals[(end - k + 1):end]
    rebuilt = rewrap_leaf(leaf, rebuild_leaf(leaf, native))
    return set_extra_leaf_params(rebuilt,
        NamedTuple{keys(extras)}(extra_vals))
end

@doc "

Update a composed distribution's parameters or replace named nodes — the
single verb for every shape-preserving edit, dispatching on the second
argument.

- a nested `NamedTuple` of parameter values/specs (below): fixes or re-specs
  free parameters, the fine-grained value edit;
- one or more `path => new_node` pairs ([`update`](@ref)`(d, edits::Pair...)`
  in `structural_edits.jl`): replaces whole nodes, coarser than a value edit but
  still same-shape.

For topology edits that change the tree shape, use [`prune`](@ref) or
[`splice`](@ref) instead.

# `update(d, params::NamedTuple)` — set free parameters

`update(d, params)` returns a new distribution of the same structure as `d` with
its parameters set from `params`, a nested NamedTuple mirroring the tree: a
[`Sequential`](@ref)/[`Parallel`](@ref) is keyed by its edge names, a leaf by
its parameter names (as in [`composed_to_table`](@ref)'s `param` column), and a
[`Resolve`](@ref) by its outcome names plus an optional `branch_probs` entry. A
censored leaf is transparent: supply only the inner delay's parameters and the
censoring is carried through.

The value type at a leaf parameter decides what happens (the object-level
spelling of \"distribution in the slot = estimate, value = fix\"):

- a **`Real`** pins the parameter to that fixed value, collapsing any
  [`uncertain`](@ref) spec on it. A NamedTuple of all-`Real` values replaces
  every free parameter (each key required), the plain concrete update;
- a **distribution** makes the parameter [`uncertain`](@ref) with that spec.
  Passing distributions switches to a partial update: only the named parameters
  change (an absent parameter keeps its current spec or fixed value), so
  `update(tree, (onset = (alpha = LogNormal(log(2), 0.2),),))` makes just
  `onset`'s `alpha` uncertain. Promote a whole tree to uncertainty over its
  free parameters with default priors via `update(tree, `
  [`param_priors`](@ref)`(tree))` — the explicit estimate-everything path;
- a **[`pool`](@ref) spec** makes the parameter partially pooled across the
  leaves that name the same group (also a partial update), e.g.
  `update(tree, (onset = (alpha = pool(:district),),))`.

A [`Resolve`](@ref) node's `branch_probs` are a node-level parameter, not a
leaf: attach a simplex-valued `Distributions.Dirichlet` at the `branch_probs`
slot to make them uncertain,
`update(node, (branch_probs = Dirichlet(ones(K)),))`. The `Dirichlet` is the
prior you write; the codec estimates the node through the `Dirichlet`'s K-1
stick-breaking coordinates (labelled `:stick_1 … :stick_{K-1}` in
[`composed_to_table`](@ref) and a fitted chain), each a `Beta` in (0, 1), so every
draw lands on the probability simplex and the gradient is well-defined. The
probabilities are recovered from any draw: a strict `update` from the stick
coordinates (as read back from a chain) collapses the node to concrete
probabilities summing to one (read them with `Distributions.probs`). Promote
attaches a flat `Dirichlet(ones(K))` per `Resolve`.

Read a fitted chain back onto a template with
`DistributionsInference.point_estimate` (or `readback_draws` for every draw)
— this package stays fit-protocol-agnostic, so chain readback lives in
DistributionsInference.jl rather than here; the NamedTuple it returns pairs
directly with `update`.

## Arguments
- `d`: the composed distribution (or bare leaf) to update.
- `params`: a nested NamedTuple keyed like `d`, each leaf value a `Real` (fix)
  or a `UnivariateDistribution` (make uncertain).

## Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
# Concrete values pin the parameters.
tree2 = update(tree, (onset_admit = (alpha = 3.0, theta = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)))
event(tree2, :onset_admit)
# A distribution makes just that parameter uncertain (a partial update).
est = update(tree, (onset_admit = (alpha = LogNormal(log(2.0), 0.2),),))
has_uncertain(est)
```

# `update(d, path => new_node, ...)` — replace nodes

`update(d, path => new_node, ...)` returns a new composed distribution of the
same outer structure as `d` with the node addressed by each `path` replaced by
`new_node`. A `path` is a `Symbol` (a top-level child), a dotted `Symbol`
(`:admit_path.admit_resolution.death`, as in [`event`](@ref) /
[`composed_to_table`](@ref)), or a tuple of edge names from the root (e.g.
`(:admit_path, :admit_resolution, :death)`); the same address [`event`](@ref)
reads is the one this writes. `new_node` may be a leaf distribution or a nested
composer. This shares the recursive reconstruction with the value-update form
above, so the result scores and `rand`s. It preserves the tree shape; for shape
changes use [`prune`](@ref) or [`splice`](@ref).

## Arguments
- `d`: the composed distribution to edit.
- `edits`: one or more `path => new_node` pairs.

## Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tree2 = update(tree, :admit_death => Gamma(3.0, 1.5))
event(tree2, :admit_death)
```

# `update(d, x::AbstractVector)` — set from flat vector

`update(d, x)` is a shorthand for `update(d, unflatten(d, x))`: rebuild the
distribution with parameters read from the flat estimated vector `x`. Each
estimated parameter (an [`uncertain`](@ref) spec in [`composed_to_table`](@ref))
takes its value from the vector, each fixed parameter its template value. This
collapses the tree at the draw and is commonly used to rebuild a distribution
from a sampler output after reading it into the flat coordinate system.

## Arguments
- `d`: the composed distribution to update.
- `x`: a flat vector of estimated parameters, of length [`flat_dimension`](@ref)
  `(d)`.

## Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = uncertain(Gamma(2.0, 1.0);
    alpha = LogNormal(log(2.0), 0.2)),
    admit_death = LogNormal(0.5, 0.4)))
# The one estimated parameter is onset_admit.alpha; the vector is length 1.
# This is equivalent to
# update(tree, ComposedDistributions.unflatten(tree, [3.0])).
result = update(tree, [3.0])
event(result, :onset_admit)
```

# `update(d, table)` — bulk-set from a Tables.jl table

`update(d, table)` reads a [`composed_to_table`](@ref)-shaped Tables.jl table
(any `Tables.istable` source with `edge`/`param` columns) and folds every row
into the tree in one call: a row's `prior` (when present and not `nothing`)
promotes that parameter to [`uncertain`](@ref), otherwise its `value` sets it
— the spreadsheet-style bulk-edit route, an input format for `update` rather
than a separate verb.

## Arguments
- `d`: the composed distribution to update.
- `table`: a Tables.jl table with `edge`/`param` columns and a `value` and/or
  `prior` column.

## Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
update(tree, composed_to_table(tree))   # a no-op round-trip here
```

# See also
- [`uncertain`](@ref)`(tree, ...)`: the promotion-only entry point built on
  the same merge-mode pipeline as this docstring's distribution-valued forms
- [`composed_to_table`](@ref): the flat inventory whose `param` names key the
  leaves; filter to `role == :param` for the parameter-only rows
- [`param_priors`](@ref): default priors for the promote path
- [`flatten`](@ref), [`unflatten`](@ref): the flat <-> nested codec
- [`prune`](@ref), [`splice`](@ref): topology edits that change the shape
"
function update end

# Typed on the abstract root, not the enumerated
# `Union{Sequential, Parallel, AbstractOneOf, Choose}` this used to carry
# (identical body either way): `update(d::AbstractComposedDistribution,
# table)` below is untyped in its second argument (any `Tables.istable`
# source), so for a `(composed distribution, NamedTuple)` call it ties with
# the untyped-first-argument `update(leaf, params::NamedTuple)` fallback
# unless a method exists that is strictly more specific in both arguments.
# The enumerated Union was a strict subtype of `AbstractComposedDistribution`
# so it happened to still win for every composer type that exists today, but
# a future composer subtype not in that Union would have hit a live dispatch
# ambiguity between the table arm and the leaf fallback the moment it called
# `update(new_type, a_namedtuple)`. Typing on the root closes that off for
# every subtype, present and future, not just the four enumerated here.
function update(d::AbstractComposedDistribution, params::NamedTuple)
    return _update(d, params, params, _has_distribution_value(params))
end

function update(leaf, params::NamedTuple)
    return _update(leaf, params, params, _has_distribution_value(params))
end

function update(d::AbstractComposedDistribution, x::AbstractVector)
    # A `Vector{<:NamedTuple}` is both an `AbstractVector` (matching this
    # method, more specific than the untyped table arm below) and
    # `Tables.istable` (a Tables.jl row table) — so it would otherwise
    # silently reach `unflatten`, which expects `Real` elements, instead of
    # the table arm it is actually shaped for. A duck-typed flat vector
    # (`Vector{Any}` holding only `Real`s, say) is not `Tables.istable`
    # (checked on the vector's static element type, not its contents), so
    # this probe does not affect it.
    Tables.istable(x) && return update(d, _table_to_nested_updates(x))
    return update(d, unflatten(d, x))
end

@doc "

Bulk-`update` a composed distribution from a Tables.jl table.

`update(d, table)` reads a [`composed_to_table`](@ref)-shaped Tables.jl table
(any `Tables.istable` source with `edge`/`param` columns, e.g.
`composed_to_table(d)` itself, a `DataFrame`, or a hand-built
`Vector{<:NamedTuple}`) and folds it into the same nested-`NamedTuple`
[`update`](@ref) pipeline used everywhere else: a row's `prior` entry (when
the table carries one and it is not `nothing`) promotes that parameter to
[`uncertain`](@ref); otherwise its `value` entry sets it, so a plain
four-column table (no `prior` column) is a purely concrete bulk write — the
spreadsheet-style workflow this table shape is built to round-trip. This is
`update`'s table input format, not a separate verb: the result is identical
to building the nested `NamedTuple` by hand and calling `update(d, nt)`.

A `Vector{<:NamedTuple}` is `Tables.istable` (a Tables.jl row table) under
both this package's `edge`/`param` convention and
`DistributionsInference`'s dotted-`name` row convention (its
`parameter_rows`) — the two are not interchangeable. Either way it
reaches this same logic (a `Vector{<:NamedTuple}` argument is caught by the
[`update`](@ref)`(d, x::AbstractVector)` method, which checks
`Tables.istable` before treating `x` as a flat numeric vector, and forwards
here). This logic requires `edge` and `param` columns and errors naming the
columns it found otherwise, so a `DistributionsInference`-shaped row vector
is refused loudly rather than silently misread.

A [`composed_to_table`](@ref)-shaped table (a `role` column present, e.g. a
`DataFrame(tree)` — a composed distribution is itself a Tables.jl source over
its full table) is filtered to its `role == :param` rows first, so passing
the whole tree straight to `update` still only writes parameters, never a
`:node`/`:attribute` row. A composed distribution as the second argument
itself (`update(a, b)`) is rejected with a clear error rather than silently
reaching this table path — pass `composed_to_table(b)` explicitly to copy
`b`'s rows.

# Arguments
- `d`: the composed distribution to edit.
- `table`: a Tables.jl table with `edge`/`param` columns and a `value` and/or
  `prior` column (as in [`composed_to_table`](@ref)).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tbl = composed_to_table(tree)
# A bulk concrete write: every `value` re-applied (a no-op read/write
# round-trip here, but the same call works after editing `tbl.value` in
# place or in a DataFrame).
update(tree, tbl)
```

# See also
- [`composed_to_table`](@ref): the table shape this reads.
- [`update`](@ref)`(d, params::NamedTuple)`: the underlying pipeline.
- [`uncertain`](@ref)`(tree, ...)`: the promotion-only entry point.
"
function update(d::AbstractComposedDistribution, table)
    return update(d, _table_to_nested_updates(table))
end

# A composed distribution is now a Tables.jl source in its own right (see
# above), so `update(a, b)` with `b` a tree would otherwise silently reach the
# table arm above and bulk-write `b`'s rows into `a` (both `edge` and `param`
# are present on any composed distribution). Reject it with a clear error
# naming the fix instead: `composed_to_table` makes the intent to copy rows
# explicit.
function update(::AbstractComposedDistribution, other::AbstractComposedDistribution)
    throw(ArgumentError(
        "update(d, table): the second argument is a $(nameof(typeof(other))) " *
        "(a composed distribution), not a table. Composed distributions are " *
        "now Tables.jl sources, so pass composed_to_table(other) explicitly " *
        "if you meant to copy its rows."))
end

# Whether an `update` NamedTuple carries any distribution-valued parameter,
# which switches `update` to *merge* mode: a distribution introduces an uncertain
# spec, a `Real` pins (collapsing any spec), and an absent parameter keeps the
# leaf's current spec or fixed value (so a partial NamedTuple targets only the
# named parameters). Without a distribution anywhere the update is a plain
# concrete replacement (*strict* mode, exact cover), the original behaviour.
# Any distribution counts: a `UnivariateDistribution` spec makes a leaf
# parameter uncertain, and a multivariate simplex prior (a `Dirichlet` at a
# `Resolve`'s `branch_probs`) makes the branch probabilities uncertain, so a
# lone `Dirichlet` update also switches to merge mode.
_has_distribution_value(x::Distribution) = true
_has_distribution_value(x::NamedTuple) = any(_has_distribution_value, values(x))
_has_distribution_value(::Any) = false

# `_update` is the recursive worker. The whole top-level `params` is threaded down
# as the `shared` source: a shared-tagged leaf is keyed at the top level by its
# tag (matching `composed_to_table`'s tag edge), so every occurrence reads the
# one entry; per-node keys are validated against the per-occurrence params with the
# shared tags excluded. `merge` carries the mode (see `_has_distribution_value`);
# the composer recursion is mode-agnostic (it already tolerates absent children),
# only the leaf method and the `Resolve` branch-probability block branch on it.

function _update(d::Union{Sequential, Parallel}, params::NamedTuple, shared,
        merge::Bool)
    names = component_names(d)
    _check_child_keys(params, names, nameof(typeof(d)), shared)
    # A structurally-recursive `map` over the paired (heterogeneous) children/
    # names tuples, not an `ntuple(...) do i ... end` closure indexing
    # `d.components[i]`: each element is built by direct tuple destructuring
    # rather than an index-driven loop body, which is what an `ntuple` closure
    # lowers to before full inlining. Same result; see #223 for why the shape
    # matters (an Enzyme/LLVM-internal crash reverse-differentiating the old
    # form on CI runners, not reproducible locally).
    parts = map(d.components, names) do child, name
        _update(child, _child_params(params, name), shared, merge)
    end
    return node_rebuild(d, parts)
end

# A `Choose` updates each alternative; a tag shared across alternatives reads one
# entry from `shared` and is placed in every occurrence.
function _update(d::Choose, params::NamedTuple, shared, merge::Bool)
    names = component_names(d)
    _check_child_keys(params, names, :Choose, shared)
    alts = ntuple(length(names)) do i
        _update(d.alternatives[i], _child_params(params, names[i]), shared,
            merge)
    end
    return node_rebuild(d, alts)
end

function _update(c::Resolve, params::NamedTuple, shared, merge::Bool)
    names = component_names(c)
    _check_child_keys(params, names, :Resolve, shared;
        optional = (:branch_probs,))
    delays = ntuple(length(names)) do i
        _update(c.delays[i], _child_params(params, names[i]), shared, merge)
    end
    return _update_branch_probs(c, delays, params, merge)
end

# Rebuild the `Resolve` with updated delays, resolving the branch probabilities
# and their attached prior from the update NamedTuple:
#
# - *merge* mode with a `branch_probs = Dirichlet(...)` entry attaches that prior,
#   making the simplex uncertain (the probabilities stay as the point). Without
#   a `branch_probs` entry the node's probabilities and prior are kept.
# - *strict* mode on a node that carries a prior reconstructs the probabilities
#   from the stick coordinates supplied (a draw from the sampler) and collapses
#   the node to concrete structure (drops the prior), mirroring a leaf collapse.
# - *strict* mode on a fixed node replaces the probabilities from concrete
#   per-outcome values, as before.
function _update_branch_probs(c::Resolve, delays, params::NamedTuple,
        merge::Bool)
    names = component_names(c)
    if merge
        haskey(params, :branch_probs) || return Resolve(names, delays,
            c.branch_probs, c.branch_prob_prior)
        bp = params.branch_probs
        # A `Dirichlet` attaches a prior, making the simplex uncertain.
        bp isa Distributions.Dirichlet &&
            return Resolve(names, delays, c.branch_probs, bp)
        # A concrete `NamedTuple` pins the probabilities, exactly as a `Real`
        # value pins a leaf parameter in merge mode: set them from the supplied
        # values (collapsing any prior), reusing the strict-mode handling. This
        # is what makes `update(tree, composed_to_table(tree))` round-trip a
        # fixed-probability `Resolve` when an uncertain leaf elsewhere in the
        # tree has flipped the whole update into merge mode — the fixed node's
        # `branch_probs` come back as a per-outcome `NamedTuple` unrelated to
        # that leaf (#219).
        bp isa NamedTuple && return _update_branch_probs_strict(c, delays, bp)
        throw(ArgumentError(
            "update(Resolve, ...): a `branch_probs` update in merge mode must " *
            "be a `Dirichlet` over the outcomes (making the simplex " *
            "uncertain) or a per-outcome `NamedTuple` (pinning fixed " *
            "probabilities); got a $(typeof(bp))"))
    end
    haskey(params, :branch_probs) || return Resolve(names, delays,
        c.branch_probs, nothing)
    bp = params.branch_probs
    bp isa NamedTuple || throw(ArgumentError(
        "update(Resolve, ...): a strict `branch_probs` update must be a " *
        "NamedTuple (stick coordinates for an uncertain node, or per-outcome " *
        "probabilities for a fixed one); got a $(typeof(bp))"))
    return _update_branch_probs_strict(c, delays, bp)
end

# Set the K probabilities from a concrete per-outcome (fixed node) or stick-
# coordinate (uncertain node) `NamedTuple` and rebuild the node without a prior
# (a concrete branch_probs collapses any uncertainty). Shared by the strict
# path and the merge path's concrete-`NamedTuple` pin (#219).
function _update_branch_probs_strict(c::Resolve, delays, bp::NamedTuple)
    probs = c.branch_prob_prior !== nothing ?
            _reconstruct_branch_probs(c, bp) : _replace_branch_probs(c, bp)
    # `update` is user-facing, so enforce sum-to-one here: the inner `Resolve`
    # constructor deliberately skips it (a prior-sampled/readback weight set is
    # legitimately unnormalised, and `_one_of_logmix` handles it), so a
    # collapse-to-concrete pin that skipped this check would build a node whose
    # `logpdf` silently scores an unnormalised mixture. A stick-reconstructed
    # simplex already sums to one, so a readback round-trip still passes; a
    # hand-supplied fixed set that does not — e.g. `(0.9, 0.9)` — is rejected
    # (#219).
    _validate_branch_probs_sum(probs)
    return Resolve(component_names(c), delays, probs, nothing)
end

# Replace the K probabilities from concrete per-outcome values (a fixed node,
# keyed by outcome name).
function _replace_branch_probs(c::Resolve, bp::NamedTuple)
    names = component_names(c)
    _check_update_keys(bp, names, Symbol("Resolve branch_probs"))
    return ntuple(i -> bp[names[i]], length(names))
end

# Reconstruct the K probabilities from the K-1 stick coordinates supplied (keyed
# `:stick_k`), read in coordinate order so the mapping is independent of the
# NamedTuple's key order.
function _reconstruct_branch_probs(c::Resolve, sticks::NamedTuple)
    names = _stick_param_names(length(component_names(c)))
    _check_update_keys(sticks, names, Symbol("Resolve branch_probs"))
    v = ntuple(k -> sticks[names[k]], length(names))
    return _stick_to_simplex(v)
end

# A racing-hazard node updates each outcome delay; there is no `branch_probs`
# block to update (the winning probability is derived).
function _update(c::Compete, params::NamedTuple, shared, merge::Bool)
    names = component_names(c)
    _check_child_keys(params, names, :Compete, shared)
    delays = ntuple(length(names)) do i
        _update(c.delays[i], _child_params(params, names[i]), shared, merge)
    end
    return node_rebuild(c, delays)
end

# Any OTHER composer node updates its children generically through
# `component_names`/`node_children`/`node_rebuild`, exactly mirroring the
# `Sequential`/`Parallel` method above. See the matching `_walk_rows!` method
# for why dispatching on the shared supertype (rather than a closed Union)
# matters: a node missing a required method gets a `MethodError` here, not a
# silently-wrong update.
function _update(d::AbstractComposedDistribution, params::NamedTuple, shared,
        merge::Bool)
    names = component_names(d)
    _check_child_keys(params, names, nameof(typeof(d)), shared)
    parts = map(node_children(d), names) do child, name
        _update(child, _child_params(params, name), shared, merge)
    end
    return node_rebuild(d, parts)
end

# A no-event marker carries no parameters, so `update` leaves it unchanged.
_update(d::NoEvent, ::NamedTuple, shared, ::Bool) = d

# Leaf: in strict mode take the new concrete values in `_leaf_param_names` order
# and rebuild (collapsing any uncertain leaf); in merge mode introduce/extend
# uncertainty via `_merge_leaf`. A shared-tagged leaf reads its entry from the
# top-level `shared` under its tag; in merge mode an absent tag entry is a no-op
# (an empty merge keeps the leaf), so a partial merge leaves untouched leaves be.
function _update(leaf, params::NamedTuple, shared, merge::Bool)
    tag = _shared_tag(leaf)
    if merge
        updates = tag === nothing ? params : get(shared, tag, NamedTuple())
        return _merge_leaf(leaf, updates)
    end
    leaf_params = tag === nothing ? params : _shared_entry(shared, tag, leaf)
    pnames = _leaf_param_names(leaf)
    # A pooled leaf reconstructs each pooled parameter from the group's shared
    # hyperparameters (read from the top-level group entry, threaded like a
    # shared tag) and the member's own latent, rather than taking scalar values.
    pooled = _pool_specs(leaf)
    pooled === nothing || return _reconstruct_pooled_leaf(
        leaf, leaf_params, shared, pooled, pnames)
    _check_update_keys(leaf_params, pnames, nameof(typeof(leaf)))
    vals = ntuple(i -> leaf_params[pnames[i]], length(pnames))
    return _update_leaf(leaf, vals)
end

# Validate a merge NamedTuple: every key must be a parameter of the leaf, and
# every value a `Real` (fix) or a `UnivariateDistribution` (make uncertain). A
# missing key is fine (that parameter is left untouched).
function _check_merge_keys(updates::NamedTuple, expected::Tuple, what)
    for k in keys(updates)
        k in expected || throw(ArgumentError(
            "update($what, ...) has unexpected parameter $(repr(k)); " *
            "expected $(collect(expected))"))
    end
    for (k, v) in pairs(updates)
        v isa Union{Real, UnivariateDistribution, Pool} || throw(ArgumentError(
            "update($what, ...): the value for $(repr(k)) must be a Real " *
            "(a fixed value), a UnivariateDistribution (an uncertain spec), " *
            "or a `pool(...)` spec (partial pooling); got $(typeof(v))"))
    end
    return nothing
end

# A child's per-occurrence params: a shared-tagged child carries no per-occurrence
# entry (its values live at the top level under its tag), so an absent key is fine
# and an empty NamedTuple is threaded down (the leaf then reads `shared`).
function _child_params(params::NamedTuple, name::Symbol)
    return haskey(params, name) ? params[name] : NamedTuple()
end

# The top-level shared entry for a tag, erroring clearly when it is absent.
function _shared_entry(shared::NamedTuple, tag::Symbol, leaf)
    haskey(shared, tag) || throw(ArgumentError(
        "update(...) is missing the shared parameter $(repr(tag)) " *
        "(a `shared($(repr(tag)), ...)` leaf needs a top-level `$tag` entry)"))
    return shared[tag]
end

# Validate a composer node's child keys. A child key may be absent (a branch
# whose only params are shared carries no per-occurrence entry; the leaf reads
# the top-level shared entry), so missing names are tolerated; an unexpected key
# (not a child name, a shared tag, or an `optional`) errors. With no shared tags
# and no all-shared branch this is the same exact-cover check as before.
function _check_child_keys(params::NamedTuple, names::Tuple, what, shared;
        optional::Tuple = ())
    allowed = (names..., optional..., keys(shared)...)
    extra_keys = filter(k -> !(k in allowed), keys(params))
    isempty(extra_keys) || throw(ArgumentError(
        "update($what, ...) has unexpected keys $(collect(extra_keys)); " *
        "expected $(collect(names))"))
    return nothing
end

# Validate `params` covers exactly `expected` (plus any `optional` keys) at the
# current node, with a clear error naming the node.
function _check_update_keys(params::NamedTuple, expected::Tuple, what;
        optional::Tuple = ())
    have = keys(params)
    missing_keys = filter(k -> !(k in have), expected)
    extra_keys = filter(k -> !(k in expected) && !(k in optional), have)
    isempty(missing_keys) || throw(ArgumentError(
        "update($what, ...) is missing $(collect(missing_keys)); " *
        "expected $(collect(expected))"))
    isempty(extra_keys) || throw(ArgumentError(
        "update($what, ...) has unexpected keys $(collect(extra_keys)); " *
        "expected $(collect(expected))"))
    return nothing
end

function _check_update_keys(params, ::Tuple, what; optional::Tuple = ())
    throw(ArgumentError(
        "update($what, ...) expects a NamedTuple; got $(typeof(params))"))
end

@doc """

Rebuild a composer node around a new children tuple.

`node_rebuild(node, children)` returns a node of the same type and own fixed
structure (a `Choose`'s `selector`, a `Resolve`'s branch probabilities) as
`node`, with its children replaced by `children` (in the same order as
[`component_names`](@ref)`(node)`). Part of the public node-extension
contract, alongside [`node_children`](@ref); see
[Writing a new composer node](@ref new-composer-node). Required from every
composer node: only the node's own constructor knows how to put a new set of
children back together (a leaf recurses no further, so it needs no method).

# Arguments
- `node`: the composer node to rebuild.
- `children`: a `Tuple` of replacement children, one per
  [`component_names`](@ref)`(node)`.

# Examples
```@example
using ComposedDistributions, Distributions

node = compose((onset = Gamma(2.0, 1.0), report = Gamma(1.5, 1.0)))
ComposedDistributions.node_rebuild(node, (LogNormal(0.5, 0.4), Gamma(1.5, 1.0)))
```

# See also
- [`node_children`](@ref): the matching read accessor.
""" function node_rebuild end

# The composers (mirrors the extension's helper, kept core-side so `update`
# is Turing-free). Rebuilds a node of the same type and metadata around a new
# children tuple (steps/branches, one_of outcome delays, Choose
# alternatives); shared by the `update` / `prune` / `splice` structural edits.
function node_rebuild(d::Sequential, components::Tuple)
    Sequential(components, component_names(d))
end
function node_rebuild(d::Parallel, components::Tuple)
    Parallel(components, component_names(d))
end
function node_rebuild(c::Resolve, delays::Tuple)
    Resolve(component_names(c), delays, c.branch_probs,
        c.branch_prob_prior)
end
node_rebuild(c::Compete, delays::Tuple) = Compete(component_names(c), delays)
node_rebuild(d::Choose, alts::Tuple) = Choose(component_names(d), alts, d.selector)

@doc """

The children of a composer node, positionally matching its names.

`node_children(node)` returns the `Tuple` of `node`'s children (steps,
branches, outcome delays or alternatives), in the same order as
[`component_names`](@ref)`(node)`. Part of the public node-extension
contract, alongside [`node_rebuild`](@ref); see
[Writing a new composer node](@ref new-composer-node). Required from every
composer node: only the node's own type knows which field holds them (a leaf
has no children, so it needs no method — every leaf's flat width is its own
free-parameter count, read from [`params`](@ref) instead). The built-ins each
hold their children in a differently-named field (`.components`, `.delays`,
`.alternatives`), which is exactly what this accessor is for: one uniform
name over that variation.

Defining it also gets [`has_varying`](@ref) and [`has_uncertain`](@ref) for
free: both walk a node generically as `any(f, node_children(node))`,
dispatching on the public `AbstractComposedDistribution` root rather than a
closed list of the built-in types, so a third-party node needs no
registration to answer either predicate.

# Arguments
- `node`: the composer node whose children are read.

# Examples
```@example
using ComposedDistributions, Distributions

node = compose((onset = Gamma(2.0, 1.0), report = Gamma(1.5, 1.0)))
ComposedDistributions.node_children(node)
```

# See also
- [`node_rebuild`](@ref): the matching rebuild verb.
- [`component_names`](@ref): the matching names.
- [`has_varying`](@ref), [`has_uncertain`](@ref): the two generic walks built
  on this accessor.
""" function node_children end

node_children(d::Union{Sequential, Parallel}) = d.components
node_children(c::AbstractOneOf) = c.delays
node_children(d::Choose) = d.alternatives

# A transitional alias for this accessor's original (leading-underscore)
# internal name, kept — like `_centred_pool_rows` in `public.jl` — so a
# downstream package reaching in by the old qualified name keeps working.
# `node_children` is the public name.
const _node_children = node_children

# --- build_priors: composed_to_table + flat priors -> nested NamedTuple -----

# Split a dotted edge `Symbol` (`:a.b`) back into its name path (`(:a, :b)`).
# The dotted ("." separator) parameter-path namespace (inverse of `_join_path`),
# distinct from the underscored event/value namespace (`_split_edge_name`).
function _split_edge(edge::Symbol)
    parts = split(string(edge), '.')
    return Tuple(Symbol.(parts))
end

# Insert `value` at the `(path..., leaf)` location of a nested `Dict` tree,
# creating intermediate `Dict`s as needed. Used to assemble the nested prior
# structure from flat table rows before freezing to NamedTuples.
function _nest_insert!(tree::Dict, path::Tuple, leaf::Symbol, value)
    node = tree
    for k in path
        node = get!(node, k, Dict{Symbol, Any}())::Dict{Symbol, Any}
    end
    node[leaf] = value
    return nothing
end

# Freeze a nested `Dict{Symbol}` tree into nested `NamedTuple`s (leaves, the
# prior objects, are left untouched).
_freeze_tree(x) = x
function _freeze_tree(d::Dict{Symbol})
    ks = Tuple(keys(d))
    return NamedTuple{ks}(map(k -> _freeze_tree(d[k]), ks))
end

# --- update(d, table): a Tables.jl table folded to a nested update NamedTuple

# `update(d, table)`'s reader: a `composed_to_table`-shaped Tables.jl table ->
# the nested NamedTuple `update`/`_update` consume. Reuses the exact
# `_split_edge`/`_nest_insert!`/`_freeze_tree` assembly `build_priors` uses,
# so the table -> tree shape is identical; only the per-row value picked
# differs (a row's own `prior`/`value`, no override/default machinery — this
# is a plain bulk write, not prior assembly). Requires `Tables.istable` and
# `edge`/`param` columns, erroring by column name on either miss so a
# differently-shaped row table (e.g. DistributionsInference's dotted-`name`
# `parameter_rows` convention, DI#20) is refused loudly rather than silently
# misread — both shapes are `Tables.istable`, so this check is the only thing
# that tells them apart. A `role` column (a `composed_to_table`-shaped table,
# or a `DataFrame(tree)` of one) is filtered to its `:param` rows first, so a
# `:node`/`:attribute` row (whose `value`/`prior` are both absent) never
# reaches the "neither a usable prior nor value" error below; `Symbol(role)`
# so a String `"param"` (a CSV/DataFrame round trip) still matches. A table
# with no `role` column (a plain four/five-column hand-built shape) is
# unfiltered, exactly as before.
function _table_to_nested_updates(table)
    Tables.istable(table) || throw(ArgumentError(
        "update(d, table) needs a Tables.jl table (composed_to_table-shaped: " *
        "edge/param columns, plus value and/or prior); got $(typeof(table)). " *
        "Pass a NamedTuple for a single targeted edit, or an " *
        "AbstractVector{<:Real} for a flat parameter vector."))
    cols = Tables.columns(table)
    colnames = Tables.columnnames(cols)
    (:edge in colnames && :param in colnames) || throw(ArgumentError(
        "update(d, table) needs `edge` and `param` columns (as produced by " *
        "composed_to_table); got columns $(collect(colnames))"))
    edges = Tables.getcolumn(cols, :edge)
    params_col = Tables.getcolumn(cols, :param)
    has_role = :role in colnames
    role_col = has_role ? Tables.getcolumn(cols, :role) : nothing
    has_value = :value in colnames
    has_prior = :prior in colnames
    values_col = has_value ? Tables.getcolumn(cols, :value) : nothing
    prior_col = has_prior ? Tables.getcolumn(cols, :prior) : nothing
    tree = Dict{Symbol, Any}()
    for i in eachindex(edges)
        has_role && Symbol(role_col[i]) !== :param && continue
        entry = if has_prior && prior_col[i] !== nothing
            prior_col[i]
        elseif has_value
            values_col[i]
        else
            throw(ArgumentError(
                "update(d, table) row $(i) (edge=$(edges[i]), " *
                "param=$(params_col[i])) has neither a usable `prior` nor a " *
                "`value` entry"))
        end
        _nest_insert!(tree, _split_edge(edges[i]), params_col[i], entry)
    end
    return _freeze_tree(tree)
end

# --- parameter-derived default priors (brms-style family defaults) ----------
#
# The default prior is classified from the parameter's own natural domain, not
# the leaf's variate support: a location-family delay (`Normal`, `Affine(Normal)`)
# has unbounded variate support, but its scale parameter still lives on the
# positive half-line, so a `minimum(dist)`/`maximum(dist)` rule would wrongly
# give it an unconstrained prior with mass on negative scale.

# Location parameters live on the whole line (a `Normal`/`LogNormal` `mu`, a
# `Uniform` bound), so they get an unconstrained default.
function _is_location_param(p::Symbol)
    p === :mu || p === :location || p === :loc || p === :lower || p === :upper
end

# Scale/shape/rate-type parameters are positive by construction (the `sigma` of a
# `Normal`/`LogNormal`, the `shape`/`scale` of a `Gamma`/`Weibull`, the `scale`
# of an `Exponential`, and the common positive parameter names of related
# families), so they get a positive-truncated default regardless of the leaf's
# variate support.
function _is_positive_param(p::Symbol)
    p === :sigma || p === :scale || p === :rate || p === :shape ||
        p === :alpha || p === :beta || p === :theta || p === :nu ||
        p === :k || p === :df || p === :mean || p === :sd
end

@doc "

Pick a default prior for a parameter row, brms-style.

`default_prior(row)` is the per-row default [`build_priors`](@ref) uses for rows
the user does not override. `row` is a `(; edge, param, value, support)`
NamedTuple (a [`composed_to_table`](@ref) `:param` row); the prior family
follows the parameter's own natural domain (classified by name), not the
leaf's variate support:

- a probability parameter, support `[0, 1]` (a `branch_probs` row) ->
  `Uniform(0, 1)`.
- a scale/shape/rate-type parameter (`:sigma`, `:scale`, `:shape`, `:rate`, ...)
  -> `truncated(Normal(value, scale); lower = 0)`, positive by construction even
  for a location-family delay (a `Normal`/`Affine(Normal)` `sigma`).
- a location parameter (`:mu`, `:location`, a `Uniform` bound) ->
  `Normal(value, scale)`, unconstrained since the location lives on the whole
  line even for a positive-support delay.
- otherwise, an unmapped name falls back to the variate support: a non-negative
  support -> `truncated(Normal(value, scale); lower = 0)`, else
  `Normal(value, scale)`.

The spread `scale` defaults to `max(abs(value), 1)`, a weakly-informative width
that scales with the parameter's magnitude.

# Arguments
- `row`: a [`composed_to_table`](@ref) `:param` row `(; edge, param, value,
  support)`.

# Examples
```@example
using ComposedDistributions, Distributions

# A positive scale parameter -> a positive-truncated default.
default_prior((; edge = :onset_admit, param = :scale,
    value = 1.0, support = (0.0, Inf)))
```

!!! note \"DistributionsInference's `with_priors`\"
    `DistributionsInference.with_priors` applies the
    same support-derived heuristic generically, over any fit-protocol
    object's `parameter_rows` (a flat, dotted-`name` row schema), not just a
    `ComposedDistributions` tree. It is a separate implementation, not a
    thin wrapper over this one: `DistributionsInference` depends on
    `ComposedDistributions`, not the reverse, so this package's own
    `default_prior`/[`build_priors`](@ref) cannot delegate to it without
    inverting that dependency. The two stay independent, parallel
    implementations of the same heuristic for their respective row shapes.

# See also
- [`build_priors`](@ref): assembles the nested prior NamedTuple, using this as
  the per-row default and accepting overrides.
"
function default_prior(row)
    lo, hi = row.support
    scale = max(abs(float(row.value)), one(float(row.value)))
    if lo == 0 && hi == 1
        return Distributions.Uniform(0, 1)
    elseif _is_positive_param(row.param)
        return Distributions.truncated(
            Distributions.Normal(row.value, scale); lower = 0)
    elseif _is_location_param(row.param)
        return Distributions.Normal(row.value, scale)
    elseif lo >= 0 && isinf(hi)
        return Distributions.truncated(
            Distributions.Normal(row.value, scale); lower = 0)
    else
        return Distributions.Normal(row.value, scale)
    end
end

@doc "

Assemble the nested prior `NamedTuple` from a [`composed_to_table`](@ref)
inventory.

`build_priors(table; priors, default)` turns the flat parameter table into the
nested `NamedTuple` that a downstream `composed_parameters_model` (and
[`update`](@ref)) expect, so users define priors against the flat table rows
rather than by hand-matching the tree. `table` may be a plain `edge`/`param`/
`value`/`support`/`prior` table, or a [`composed_to_table`](@ref)-shaped table
carrying a `role` column (e.g. `composed_to_table(tree)` or `DataFrame(tree)`
itself); a `role` column is filtered to its `:param` rows first, the same way
[`update`](@ref)`(d, table)` reads one.

For each `:param` row the prior is chosen in order:
1. a user `priors` override for that `(edge, param)`, if present, else
2. the row's attached `prior` (an [`uncertain`](@ref) parameter's spec rides
   the table's `prior` column), if present, else
3. `default(row)`, the per-row default (support-derived [`default_prior`](@ref)
   unless a different `default` function is given).

By default every row gets a sensible support-derived prior, so
`build_priors(composed_to_table(tree))` alone yields a complete prior
NamedTuple. A user overrides only the parameters they care about (brms-style
partial override) through `priors`.

`row` is a `NamedTuple` `(; edge, param, value, support)` (the table's columns
for that row), so a custom `default` can pick a prior from the parameter's
`support`.

# Arguments
- `table`: a [`composed_to_table`](@ref) inventory (any Tables.jl column table
  with `edge`, `param`, `value`, `support` columns, optionally a `role`
  column).

# Keyword Arguments
- `priors`: per-parameter overrides, either a `(edge, param) => prior` mapping
  (e.g. a `Dict`) or a nested `NamedTuple` keyed like the tree
  (`(onset_admit = (alpha = prior,),)`); only the listed parameters are
  overridden (default: empty).
- `default`: a function `row -> prior` for rows not overridden (default:
  [`default_prior`](@ref), deriving the prior family from the parameter's
  support).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
tbl = composed_to_table(tree)
# Support-derived defaults everywhere, overriding only one parameter.
nested = build_priors(tbl;
    priors = (onset_admit = (alpha = truncated(Normal(2, 0.5); lower = 0),),))
nested.onset_admit.alpha
```

# See also
- [`composed_to_table`](@ref): the flat inventory keyed against.
- [`default_prior`](@ref): the support-derived per-row default.
- `composed_parameters_model` (downstream), [`update`](@ref): consume the result.
"
function build_priors(table; priors = Dict{Tuple{Symbol, Symbol}, Any}(),
        default = default_prior)
    Tables.istable(table) || throw(ArgumentError(
        "build_priors(table) needs a Tables.jl table (composed_to_table-" *
        "shaped: edge/param columns, plus value and support); " *
        "got $(typeof(table))."))
    cols = Tables.columns(table)
    colnames = Tables.columnnames(cols)
    (:edge in colnames && :param in colnames) || throw(ArgumentError(
        "build_priors(table) needs `edge` and `param` columns (as produced " *
        "by composed_to_table); got columns $(collect(colnames))"))
    edges = Tables.getcolumn(cols, :edge)
    params_col = Tables.getcolumn(cols, :param)
    values = Tables.getcolumn(cols, :value)
    supports = Tables.getcolumn(cols, :support)
    # A `role` column (a `composed_to_table`-shaped table) is filtered to its
    # `:param` rows first, the same way `update(d, table)` reads one — a
    # `:node`/`:attribute` row carries no usable `value`/`support`. A table
    # with no `role` column (a plain hand-built shape) is unfiltered.
    has_role = :role in colnames
    role_col = has_role ? Tables.getcolumn(cols, :role) : nothing
    # The attached-prior column (an uncertain parameter's spec); tolerate its
    # absence so a hand-built four-column table keeps working.
    attached = :prior in colnames ? Tables.getcolumn(cols, :prior) : nothing
    tree = Dict{Symbol, Any}()
    for i in eachindex(edges)
        has_role && Symbol(role_col[i]) !== :param && continue
        edge = edges[i]
        param = params_col[i]
        ovr = _prior_override(priors, edge, param)
        prior = if ovr !== nothing
            ovr
        elseif attached !== nothing && attached[i] !== nothing
            attached[i]
        elseif default !== nothing
            row = (; edge = edge, param = param,
                value = values[i], support = supports[i])
            default(row)
        else
            throw(ArgumentError(
                "no prior for ($edge, $param) and no default supplied"))
        end
        _nest_insert!(tree, _split_edge(edge), param, prior)
    end
    return _freeze_tree(tree)
end

# A user override for `(edge, param)`, or `nothing` if none. Accepts a mapping
# keyed by the `(edge, param)` pair (a `Dict`) or a nested `NamedTuple` keyed
# like the tree (descend the edge path, then the param). Missing keys return
# `nothing` so the row falls through to the default.
function _prior_override(priors::NamedTuple, edge::Symbol, param::Symbol)
    node = priors
    for name in _split_edge(edge)
        node isa NamedTuple && haskey(node, name) || return nothing
        node = node[name]
    end
    node isa NamedTuple && haskey(node, param) || return nothing
    return node[param]
end

function _prior_override(priors, edge::Symbol, param::Symbol)
    key = (edge, param)
    return haskey(priors, key) ? priors[key] : nothing
end

@doc "

Build the nested prior `NamedTuple` straight from a composed distribution.

`param_priors(tree; priors, default)` is a thin convenience over
[`build_priors`](@ref)`(`[`composed_to_table`](@ref)`(tree))`: it reads the
full structural inventory of the composed distribution `tree` (filtered to
its `:param` rows by [`build_priors`](@ref)) and assembles the nested prior
`NamedTuple` in one call, forwarding the same keyword surface. It adds no
prior logic of its own.

The result is spec-shaped (a nested NamedTuple of distributions keyed like the
tree), so it feeds [`update`](@ref) directly: `update(tree, param_priors(tree))`
promotes every free parameter to [`uncertain`](@ref) with its default prior —
the explicit estimate-everything path under uncertain-first (a bare tree
estimates nothing). Pass `priors` to swap in your own spec for named
parameters.

# Arguments
- `tree`: a composed distribution from [`compose`](@ref).

# Keyword Arguments
- `priors`: per-parameter overrides, either a `(edge, param) => prior` mapping
  or a nested `NamedTuple` keyed like the tree; only the listed parameters are
  overridden (default: empty).
- `default`: a function `row -> prior` for rows not overridden (default:
  [`default_prior`](@ref)).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)))
priors = param_priors(tree)
priors.onset_admit.alpha
```

# See also
- [`build_priors`](@ref): the underlying table-based assembly.
- [`composed_to_table`](@ref): the structural inventory read internally.
"
function param_priors(tree; kwargs...)
    priors = build_priors(composed_to_table(tree); kwargs...)
    return _attach_branch_prob_priors(priors, tree)
end

# The nested prior NamedTuple carries no node identity, so the flat `Dirichlet`
# default for an uncertain branch-probability simplex is injected by walking the
# tree alongside the built priors: at each `Resolve` the `branch_probs` entry is
# set to the node's own attached `Dirichlet` if it has one, else a flat
# `Dirichlet(ones(K))`, so `update(tree, param_priors(tree))` promotes the
# simplex to uncertain with a sensible default (the branch probabilities are
# recovered from any draw). `Compete` (winning probability derived) and `Choose`
# (data-selected) have no node-level probability parameter, so nothing is
# injected for them.
function _attach_branch_prob_priors(nt::NamedTuple, d)
    ks = keys(nt)
    vals = map(ks) do k
        if k === :branch_probs && d isa Resolve
            _promote_branch_prior(d)
        else
            child = _prior_child_node(d, k)
            child === nothing ? nt[k] : _attach_branch_prob_priors(nt[k], child)
        end
    end
    return NamedTuple{ks}(vals)
end
_attach_branch_prob_priors(x, d) = x

function _promote_branch_prior(c::Resolve)
    return c.branch_prob_prior === nothing ?
           Distributions.Dirichlet(ones(length(component_names(c)))) :
           c.branch_prob_prior
end

# The child tree node under name `k` for the prior walk, or `nothing` when `k`
# is a leaf parameter name (not a child node), so the walk stops descending.
function _prior_child_node(d::Union{Sequential, Parallel}, k::Symbol)
    names = component_names(d)
    i = findfirst(==(k), names)
    return i === nothing ? nothing : d.components[i]
end
function _prior_child_node(c::AbstractOneOf, k::Symbol)
    i = findfirst(==(k), component_names(c))
    return i === nothing ? nothing : c.delays[i]
end
function _prior_child_node(d::Choose, k::Symbol)
    i = findfirst(==(k), component_names(d))
    return i === nothing ? nothing : d.alternatives[i]
end
_prior_child_node(::Any, ::Symbol) = nothing

# --- name introspection ----------------------------------------------------

@doc "

The *flat* event names of a composed distribution.

`event_names(d)` returns the tuple of event names in flat depth-first order:
the root origin event followed by one target event per leaf edge.
An inner composer's events are exposed, so `compose((path = [a, b],))` lists the
inner `(:onset, ...)` events rather than just the `(:path,)` edge. Event names
are derived from the edge names (an edge `:onset_admit` gives origin `:onset` and
target `:admit`); a positional default edge contributes `:event_i`. These event
names key a data row, distinct from the nested edge/child structure of
[`event_tree`](@ref) (whose first level is the top-level child names).

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((onset_admit = LogNormal(1.5, 0.4),
    admit_death = Gamma(2.0, 1.0)))
event_names(tree)
```

# See also
- [`event_tree`](@ref): the *nested* tree of event names
- [`event`](@ref): fetch a child or subtree by name path
- [`composed_to_table`](@ref): the flat structural table
"
function event_names(d::Union{Sequential, Parallel, AbstractOneOf})
    return _flat_event_names(d)
end
# A `Choose` has no single flat layout (the active alternative is data-selected),
# so its flat event names are its alternative names.
event_names(d::Choose) = component_names(d)

@doc "

The *nested* tree of event names of a composed distribution.

`event_tree(d)` returns the event-name structure as data: a nested `NamedTuple`
keyed by child name down to the leaves, mirroring the tree. Its first level is
the top-level child names (the old top-level `event_names` result); a
[`Sequential`](@ref)/[`Parallel`](@ref)/[`Choose`](@ref) child recurses to its
own nested NamedTuple, a [`Resolve`](@ref) child to its outcome names, and a
leaf to its own name. Pair with [`event_names`](@ref) for the *flat* per-event
layout that matches `rand`/`mean`/`var`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((admit_path = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))),
    onset_recover = Gamma(3.0, 1.0)))
event_tree(tree)
```

# See also
- [`event_names`](@ref): the *flat* per-event names
- [`event`](@ref): fetch a child or subtree by name path
"
function event_tree(d::Union{Sequential, Parallel})
    names = component_names(d)
    vals = ntuple(i -> _event_tree_child(names[i], d.components[i]),
        length(names))
    return NamedTuple{names}(vals)
end

function event_tree(c::AbstractOneOf)
    names = component_names(c)
    vals = ntuple(i -> _event_tree_child(names[i], c.delays[i]),
        length(names))
    return NamedTuple{names}(vals)
end

function event_tree(d::Choose)
    names = component_names(d)
    vals = ntuple(i -> _event_tree_child(names[i], d.alternatives[i]),
        length(names))
    return NamedTuple{names}(vals)
end

# A composer child recurses to its own nested NamedTuple; a leaf is keyed by its
# parent under its own name, so its value is just that name (the leaf event).
function _event_tree_child(
        ::Symbol, c::Union{Sequential, Parallel, AbstractOneOf, Choose})
    event_tree(c)
end
_event_tree_child(name::Symbol, ::Any) = name

# No child named `name` at this level: raise the same "no child named ...;
# have [...]" style `edit`/`prune`/`splice`/`_pick` use, rather than a bare
# `KeyError` with no hint about the valid alternatives.
function _no_event_child_error(d, name::Symbol, names::Tuple)
    throw(ArgumentError(
        "event($(nameof(typeof(d))), ...): no child named $(repr(name)); " *
        "have $(collect(names))"))
end

# Direct-child lookup by a single (un-dotted) name. Internal so the public
# `event` can split a dotted path before descending.
function _event_child(d::Union{Sequential, Parallel}, name::Symbol)
    names = component_names(d)
    idx = findfirst(==(name), names)
    idx === nothing && _no_event_child_error(d, name, names)
    return d.components[idx]
end

function _event_child(c::AbstractOneOf, name::Symbol)
    names = component_names(c)
    idx = findfirst(==(name), names)
    idx === nothing && _no_event_child_error(c, name, names)
    return c.delays[idx]
end

function _event_child(d::Choose, name::Symbol)
    names = component_names(d)
    idx = findfirst(==(name), names)
    idx === nothing && _no_event_child_error(d, name, names)
    return d.alternatives[idx]
end

@doc "

Fetch a composed distribution's child (event/edge), or descend a name path.

`event(d, path...)` returns the sub-distribution of `d` at the named location: a
single `Symbol` fetches a direct child (a branch of a [`Parallel`](@ref), a step
of a [`Sequential`](@ref), an outcome delay of a [`Resolve`](@ref), or an
alternative of a [`Choose`](@ref)); multiple `Symbol`s, or a single dotted-path
`Symbol` (`:admit_path.admit_death`, as in [`composed_to_table`](@ref)'s `edge`
column), descend the tree one name per step. Throws an `ArgumentError` naming
the valid children if a name along the path is not a child at that level
(mirroring [`update`](@ref)/[`prune`](@ref)/[`splice`](@ref)).

# Arguments
- `d`: the composed distribution to look up a child of (or descend into).
- `path`: one or more edge/event names (`Symbol`s) from `d` down to the target,
  or a single dotted-path `Symbol`.

# Examples
```@example
using ComposedDistributions, Distributions

tree = compose((admit_path = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4))),
    onset_recover = Gamma(3.0, 1.0)))
event(tree, :onset_recover)
event(tree, :admit_path, :admit_death)
```

# See also
- [`event_names`](@ref): list a node's flat event names
- [`event_tree`](@ref): the nested tree of event names
"
function event(d)
    # No name is an error: `event` needs at least one name to fetch.
    throw(ArgumentError("event needs at least one name"))
end

# A single name: a dotted `Symbol` (`:a.b`) splits into its steps and descends;
# a bare name is a single direct-child lookup.
function event(d, name::Symbol)
    steps = _split_edge(name)
    node = d
    for step in steps
        node = _event_child(node, step)
    end
    return node
end

# Two or more names descend the tree one step per name.
function event(d, name1::Symbol, name2::Symbol, rest::Symbol...)
    node = _event_child(d, name1)
    for name in (name2, rest...)
        node = _event_child(node, name)
    end
    return node
end
