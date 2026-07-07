module ComposedDistributionsFlexiChainsExt

# ComposedDistributions x DynamicPPL x FlexiChains: read a fitted chain's
# parameters back onto a composed-distribution template. `chain_to_params`
# (declared with its docstring in src/composers/readback.jl) reduces a
# FlexiChains chain to the nested NamedTuple `update` consumes; `param_draws`
# keeps every draw; `strip_prefix` drops the outer submodel prefix from a
# chain's parameter names. Loaded only when both DynamicPPL and FlexiChains are
# available.

using ComposedDistributions: ComposedDistributions, Sequential, Parallel,
                             Resolve, Compete, Choose, component_names,
                             _shared_tag, _leaf_param_names, _collect_shared,
                             _uncertain_specs, free_leaf
import ComposedDistributions: chain_to_params, update, strip_prefix,
                              param_draws
using Distributions: params
using DynamicPPL: VarName
using FlexiChains: FlexiChains
using Statistics: mean

# `AbstractPPL.unprefix` removes one leading prefix from a `VarName`. It is the
# parent module of `VarName` (re-exported through DynamicPPL) and is reached
# that way rather than as a direct dependency, keeping the weakdep set
# unchanged.
const _AbstractPPL = parentmodule(VarName)

# Read a chain's free parameters into a `String`-keyed lookup, so the tree-walk
# matches each template parameter by its dotted name instead of rebuilding a
# `VarName` optic per parameter. The chain keys are the submodel-prefixed
# `VarName`s (e.g. `d.onset_admit.shape`); their `string` is the dotted path
# the walk forms.
#
# The reduction is configurable: `summary` is any `AbstractVector -> scalar`
# (default `mean`, so `median`, `mode`, `x -> quantile(x, 0.9)` all work),
# applied to each parameter's draws. `draw` keeps the single-iteration
# shortcut (an exact index). `draws` selects a subset of iterations to reduce
# over: a range / index vector (positional), or a predicate `i -> Bool` over
# the iteration index (e.g. warmup drop / thinning); `nothing` uses every draw.
#
# Only scalar parameters belong in the lookup. A latent fit can also carry
# vector-valued per-record VarNames whose draws are `Vector`s; the template
# walk never requests them. Filtering on the column element type keeps the
# scalars and skips the vector params, so the reduction is safe to read.
function _value_lookup(chain, draw, draws, summary)
    vns = [vn for vn in FlexiChains.parameters(chain)
           if eltype(chain[vn]) <: Real]
    if draw !== nothing
        return Dict(string(vn) => vec(chain[vn])[draw] for vn in vns)
    end
    sel = _draw_indices(chain, draws)
    return Dict(string(vn) => summary(_select_draws(chain[vn], sel))
    for vn in vns)
end

# The iteration indices a `draws` selector picks out. `nothing` is every
# iteration; a predicate filters the index range; anything else (a range /
# index vector) is taken as the indices directly.
_draw_indices(chain, ::Nothing) = Colon()
function _draw_indices(chain, draws)
    if draws isa Function
        n = length(vec(chain[first(FlexiChains.parameters(chain))]))
        return [i for i in 1:n if draws(i)]
    end
    return collect(draws)
end

_select_draws(col, ::Colon) = vec(col)
_select_draws(col, sel) = vec(col)[sel]

# Look up one parameter by its dotted name (`prefix.path...`); `nothing` if
# absent (e.g. a `Resolve`'s branch probabilities kept fixed in the template).
_read_value(lookup, key) = get(lookup, key, nothing)

# Form the dotted name a submodel-sampled parameter carries: the `~`-bound
# `prefix` then the edge path and parameter name (e.g. prefix `:d`, path
# `(:onset_admit, :shape)` -> `"d.onset_admit.shape"`), matching `string(vn)`.
# An empty prefix (`Symbol("")`) drops the leading segment, so a
# `strip_prefix`ed chain reads back at the bare edge path
# (`"onset_admit.shape"`).
function _dotted(prefix::Symbol, path::Tuple)
    prefix === Symbol("") && return join(string.(path), ".")
    return join(string.((prefix, path...)), ".")
end

# Build the nested NamedTuple by walking the template, so its key order
# matches `params(template)` (deterministic, not Dict-ordered). Each node
# reads its children/parameters from `lookup` at its running dotted name.

function _node_params(d::Union{Sequential, Parallel}, lookup, prefix, path)
    names = component_names(d)
    vals = map(zip(names, d.components)) do (name, child)
        _node_params(child, lookup, prefix, (path..., name))
    end
    return NamedTuple{names}(Tuple(vals))
end

function _node_params(c::Resolve, lookup, prefix, path)
    delays = map(zip(c.names, c.delays)) do (name, delay)
        _node_params(delay, lookup, prefix, (path..., name))
    end
    base = NamedTuple{c.names}(Tuple(delays))
    # Branch probabilities only appear in the chain when sampled (a prior was
    # supplied); otherwise omit so `update` keeps the template's fixed values.
    bp = map(c.names) do name
        _read_value(lookup, _dotted(prefix, (path..., :branch_probs, name)))
    end
    any(x -> x === nothing, bp) && return base
    return merge(base, (; branch_probs = NamedTuple{c.names}(Tuple(bp))))
end

# A racing-hazard node has only its outcome-delay params in the chain (no
# branch_probs block, since the winning probability is derived).
function _node_params(c::Compete, lookup, prefix, path)
    delays = map(zip(c.names, c.delays)) do (name, delay)
        _node_params(delay, lookup, prefix, (path..., name))
    end
    return NamedTuple{c.names}(Tuple(delays))
end

# A `Choose` walks its alternatives by name (mirroring the core
# `params`/`update` traversal), so a posterior reads back onto a Choose
# template. The nested NamedTuple is keyed by the alternative names; a
# shared-tagged leaf inside an alternative reads its values from the top level
# under its tag (see the leaf method below), so a tie across alternatives maps
# to the one chain entry.
function _node_params(d::Choose, lookup, prefix, path)
    vals = map(zip(d.names, d.alternatives)) do (name, alt)
        _node_params(alt, lookup, prefix, (path..., name))
    end
    return NamedTuple{d.names}(Tuple(vals))
end

# Leaf: read each free parameter (in `_leaf_param_names` order) from `lookup`.
# Under uncertain-first the chain contains exactly the ESTIMATED (spec'd)
# parameters, so a parameter present in the chain is read from it, and an absent
# FIXED parameter falls back to the template's value (it was never sampled). A
# spec'd parameter missing from the chain signals a chain that does not match
# the template (wrong prefix, or not the chain that produced it), so that errors
# rather than silently substituting the template.
#
# A shared-tagged leaf (`shared(:tag, ...)`) is deduped in the chain: it is
# sampled once under its tag (`<prefix>.<tag>.<param>`), not per occurrence,
# matching `params_table`'s tag edge. So an occurrence reads its values at the
# bare tag (ignoring its branch path); every occurrence then maps to the one
# chain entry, and the nested NamedTuple keys the group once at the top level
# under its tag (read back by `update`).
function _node_params(leaf, lookup, prefix, path)
    tag = _shared_tag(leaf)
    keypath = tag === nothing ? path : (tag,)
    pnames = _leaf_param_names(leaf)
    specs = _uncertain_specs(leaf)
    tvals = params(free_leaf(leaf))
    vals = ntuple(length(pnames)) do i
        p = pnames[i]
        key = _dotted(prefix, (keypath..., p))
        v = _read_value(lookup, key)
        if v !== nothing
            v
        elseif specs !== nothing && haskey(specs, p)
            throw(ArgumentError("leaf parameter $key not found in chain"))
        else
            tvals[i]
        end
    end
    return NamedTuple{pnames}(Tuple(vals))
end

# Read every shared group once from the chain into a top-level `tag => values`
# NamedTuple, mirroring how the core `update` reads a shared leaf from the
# top-level tag entry. `_collect_shared` gives the first-occurrence leaf per
# tag (its inner family fixes the parameter names), deduped in pre-order.
function _shared_params(template, lookup, prefix)
    groups = _collect_shared(template)
    isempty(groups) && return NamedTuple()
    entries = map(groups) do (tag, leaf)
        tag => _node_params(leaf, lookup, prefix, ())
    end
    return NamedTuple(entries)
end

function chain_to_params(template, chain; prefix::Symbol = :d, draw = nothing,
        draws = nothing, summary = mean)
    lookup = _value_lookup(chain, draw, draws, summary)
    tree = _node_params(template, lookup, prefix, ())
    # A shared-tagged leaf is sampled once under its tag, so add a top-level
    # `tag` entry for each shared group; the core `update` reads each
    # occurrence from it (per-occurrence entries in `tree` are tolerated).
    return merge(tree, _shared_params(template, lookup, prefix))
end

# The total number of draws in the chain (length of any parameter's draw
# vector). Used to materialise a `nothing` (every-draw) selector into the
# concrete iteration indices `param_draws` maps over.
function _n_draws(chain)
    return length(vec(chain[first(FlexiChains.parameters(chain))]))
end

# Vectorised per-draw read: one nested NamedTuple per selected iteration, each
# equal to `chain_to_params(template, chain; draw = i)`. `param_draws` is the
# all-draws counterpart of the reducing `chain_to_params`: it keeps every draw
# so a tutorial maps `update` over the result for per-draw distributions /
# trajectories, instead of looping `update` per draw or hand-indexing
# `@varname`. `draws` (a range / index vector, or a predicate over the
# iteration index) restricts to a subset; `nothing` keeps every draw. Reuses
# `chain_to_params(...; draw = i)` per index, so each entry matches the
# existing single-draw read exactly.
function param_draws(template, chain::FlexiChains.FlexiChain;
        prefix::Symbol = :d, draws = nothing)
    sel = _draw_indices(chain, draws)
    idx = sel isa Colon ? (1:_n_draws(chain)) : sel
    return [chain_to_params(template, chain; prefix = prefix, draw = i)
            for i in idx]
end

# Update a composed distribution directly from a fitted chain, so callers
# write `update(template, chain)` rather than threading `chain_to_params` by
# hand. Reads the chain into the nested NamedTuple (posterior means, or a
# single `draw`) at the submodel `prefix`, then reconstructs through the core
# `update`.
@doc "

Update a composed distribution's parameters straight from a fitted chain.

`update(template, chain)` reads `chain` (sampled through a
`~ to_submodel(...)`-based parameters model) into the nested NamedTuple and
rebuilds `template` with those values, so the workflow is one call instead of
`update(template, chain_to_params(template, chain))`. By default it reduces
each parameter's draws with `mean`; pass any `summary` reduction, restrict to a
subset of draws with `draws` (a range / index vector, or a predicate over the
iteration index), or pass `draw=i` for a single iteration. The `prefix`
keyword names the submodel variable the parameters were sampled under
(default `:d`).

This method is available only when both `DynamicPPL` and `FlexiChains` are
loaded.

# Arguments
- `template`: the composed distribution the chain's parameters were sampled
  against.
- `chain`: the fitted `FlexiChains` chain to read parameter values from.

# Keyword Arguments
- `prefix`: the submodel variable name the parameters were sampled under
  (default `:d`).
- `summary`: the reduction `AbstractVector -> scalar` applied to each
  parameter's draws (default `mean`).
- `draws`: a subset of iterations to reduce over (a range / index vector, or a
  predicate over the iteration index); `nothing` uses every draw.
- `draw`: a single iteration index to read (overrides `summary`/`draws`).

# See also
- [`chain_to_params`](@ref): the nested NamedTuple this reads.
- [`update`](@ref): the NamedTuple-keyed reconstruction this delegates to.
"
function update(template, chain::FlexiChains.FlexiChain;
        prefix::Symbol = :d, draw = nothing, draws = nothing, summary = mean)
    params = chain_to_params(template, chain; prefix = prefix, draw = draw,
        draws = draws, summary = summary)
    return update(template, params)
end

# Remove the single leading `prefix` from one parameter VarName, leaving the
# rest of the path (the edge-path namespacing) intact. A VarName that does not
# carry the prefix (`AbstractPPL.unprefix` throws an `ArgumentError`) is
# returned unchanged, so a plain top-level variable sampled alongside the
# submodel is left alone rather than erroring.
function _strip_one(vn::VarName, prefix::VarName)
    try
        return _AbstractPPL.unprefix(vn, prefix)
    catch err
        err isa ArgumentError && return vn
        rethrow()
    end
end

# `strip_prefix(chain; prefix = :d)`: map every parameter VarName through
# `_strip_one`, dropping the outer submodel prefix. `FlexiChains.map_parameters`
# touches only the parameter keys, so the chain's extras (log-densities,
# sampler stats) and draw values are preserved.
function strip_prefix(chain::FlexiChains.FlexiChain; prefix::Symbol = :d)
    pre = VarName{prefix}()
    return FlexiChains.map_parameters(vn -> _strip_one(vn, pre), chain)
end

end
