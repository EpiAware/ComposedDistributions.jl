# ComposedDistributions × ModifiedDistributions
#
# A modified leaf (`Affine` / `Weighted` / `Transformed` / `Modified`) is a
# delay wrapper: its modification (scale/shift, likelihood weight, forward
# transform, hazard effect) is fixed structure, not a free parameter. So
# inside a composed tree it must peel like any other leaf wrapper —
# `free_leaf` reaches the inner free delay, `rewrap_leaf` rebuilds the
# modifier around a new inner delay, and `_shared_tag` sees a tag through it.
# `Transformed`'s `ThinOp` is the one exception: `thin`'s reporting
# probability is a FREE parameter (CensoredDistributions'
# `forward_transform.jl` precedent), so it also plugs into the core's
# `extra_leaf_params`/`set_extra_leaf_params` hooks as a `:thin` entry,
# surfacing as a `:thin` row in `params_table` and round-tripping through
# `update`. This seam needs both packages' types, so it lives here (the
# function owner — ComposedDistributions for `free_leaf`/`rewrap_leaf`/
# `shared_tag`/`extra_leaf_params`/`set_extra_leaf_params`,
# ModifiedDistributions for `get_dist` — plus at least one type at the seam,
# so no piracy).
#
# The modifier payloads are read through ModifiedDistributions' public
# accessors (`get_dist`/`get_scale`/`get_shift`/`get_weight`/`get_effect`/
# `get_link`/`get_op`/`get_factor`, ModifiedDistributions#61) rather than
# struct fields, so a field rename upstream cannot silently break this seam.
module ComposedDistributionsModifiedDistributionsExt

import ComposedDistributions: free_leaf, rewrap_leaf, shared_tag,
                              uncertain_specs, extra_leaf_params,
                              set_extra_leaf_params, leaf_mean, leaf_var,
                              instantiate, has_varying
using ComposedDistributions: Shared, AbstractContext
using Distributions: mean, var
import ModifiedDistributions: get_dist
using ModifiedDistributions: Affine, Weighted, Transformed, Modified,
                             ThinOp, affine, modify, get_scale, get_shift,
                             get_weight, get_effect, get_link, get_op,
                             get_factor

# --- free_leaf: reach the inner free delay through a modifier ---------------

free_leaf(d::Affine) = free_leaf(get_dist(d))
free_leaf(d::Weighted) = free_leaf(get_dist(d))
free_leaf(d::Transformed) = free_leaf(get_dist(d))
free_leaf(d::Modified) = free_leaf(get_dist(d))

# --- rewrap_leaf: rebuild the modifier around a new inner delay --------------

function rewrap_leaf(d::Affine, inner)
    return affine(rewrap_leaf(get_dist(d), inner);
        scale = get_scale(d), shift = get_shift(d))
end
function rewrap_leaf(d::Weighted, inner)
    return Weighted(rewrap_leaf(get_dist(d), inner), get_weight(d))
end
function rewrap_leaf(d::Transformed, inner)
    return Transformed(rewrap_leaf(get_dist(d), inner), get_op(d))
end
function rewrap_leaf(d::Modified, inner)
    return modify(rewrap_leaf(get_dist(d), inner), get_effect(d);
        link = get_link(d))
end

# --- _shared_tag: see a shared tag through a modifier -----------------------

shared_tag(d::Affine) = shared_tag(get_dist(d))
shared_tag(d::Weighted) = shared_tag(get_dist(d))
shared_tag(d::Transformed) = shared_tag(get_dist(d))
shared_tag(d::Modified) = shared_tag(get_dist(d))

# --- _uncertain_specs: see uncertain parameters through a modifier ----------
#
# A modifier over an `uncertain(...)` leaf must still expose the attached
# parameter specs, so `params_table`'s prior column and the marginal `rand`
# see through the modifier exactly like the tag protocol does.

uncertain_specs(d::Affine) = uncertain_specs(get_dist(d))
uncertain_specs(d::Weighted) = uncertain_specs(get_dist(d))
uncertain_specs(d::Transformed) = uncertain_specs(get_dist(d))
uncertain_specs(d::Modified) = uncertain_specs(get_dist(d))

# --- overall moments: use the modifier's own moment, not the free leaf's -----
#
# The per-leaf moment defaults to `mean(free_leaf(leaf))`, which is the free
# delay's — right for the parameter surface, but wrong for a moment when the
# modifier itself changes the distribution. `Affine` has correct analytic
# moments (`scale*mean + shift`, `scale^2*var`) and its `rand` honours the
# transform, so its overall moment must too. `Weighted` / `Transformed` delegate
# their moments straight to the inner delay, so their free-leaf moment already
# agrees — no method needed.

leaf_mean(d::Affine) = mean(d)
leaf_var(d::Affine) = var(d)

# `Modified` has no analytic moment yet (blocked on ModifiedDistributions#44's
# numeric cumulative-hazard path), and `free_leaf` peels it to the inner delay —
# so the default `mean(free_leaf(d))` would silently return the UNMODIFIED
# delay's moment, understating the hazard modification. Error informatively
# instead: a chain containing a hazard-modified step has no overall moment until
# #44 lands (draw the marginal with `rand` meanwhile).
function leaf_mean(d::Modified)
    throw(ArgumentError(
        "a hazard-modified (`Modified`) leaf has no analytic mean; the " *
        "modified moment needs numeric cumulative-hazard integration " *
        "(ModifiedDistributions#44). Draw the marginal with `rand` for a " *
        "Monte-Carlo moment, or exclude the modified step."))
end
function leaf_var(d::Modified)
    throw(ArgumentError(
        "a hazard-modified (`Modified`) leaf has no analytic variance; the " *
        "modified moment needs numeric cumulative-hazard integration " *
        "(ModifiedDistributions#44). Draw the marginal with `rand` for a " *
        "Monte-Carlo moment, or exclude the modified step."))
end

# --- extra_leaf_params / set_extra_leaf_params: surface a thin(...) reporting
# probability as a free parameter -------------------------------------------
#
# `thin(d, p)` (a `Transformed` carrying a `ThinOp`) is NOT a fixed-structure
# modifier like the others: `p` enters the per-record likelihood, so it must
# be inventoried by `params_table` and round-tripped by `update` like any
# other leaf parameter (see `src/composers/introspection.jl`'s hook
# docstring, and CensoredDistributions' `forward_transform.jl` for the
# precedent this mirrors). It plugs into the generic extra-parameter protocol
# as a `:thin` entry on `[0, 1]`. The peel-through modifiers
# (`Affine`/`Weighted`/`Modified`) forward to their inner delay so a thinned
# leaf still reports its factor underneath any of them.

extra_leaf_params(d::Affine) = extra_leaf_params(get_dist(d))
extra_leaf_params(d::Weighted) = extra_leaf_params(get_dist(d))
extra_leaf_params(d::Modified) = extra_leaf_params(get_dist(d))
function extra_leaf_params(d::Transformed)
    op = get_op(d)
    return op isa ThinOp ?
           (thin = (value = get_factor(op), support = (0.0, 1.0)),) :
           extra_leaf_params(get_dist(d))
end

# Empty-`NamedTuple` identity methods disambiguate the modifier forwards below
# from the core's generic `set_extra_leaf_params(leaf, ::NamedTuple{()})` (both
# would match a modifier with no extras); setting no extras is the identity.
set_extra_leaf_params(d::Affine, ::NamedTuple{()}) = d
set_extra_leaf_params(d::Weighted, ::NamedTuple{()}) = d
set_extra_leaf_params(d::Modified, ::NamedTuple{()}) = d
set_extra_leaf_params(d::Transformed, ::NamedTuple{()}) = d

function set_extra_leaf_params(d::Affine, vals::NamedTuple)
    return affine(set_extra_leaf_params(get_dist(d), vals);
        scale = get_scale(d), shift = get_shift(d))
end
function set_extra_leaf_params(d::Weighted, vals::NamedTuple)
    return Weighted(set_extra_leaf_params(get_dist(d), vals), get_weight(d))
end
function set_extra_leaf_params(d::Modified, vals::NamedTuple)
    return modify(set_extra_leaf_params(get_dist(d), vals),
        get_effect(d); link = get_link(d))
end
function set_extra_leaf_params(d::Transformed, vals::NamedTuple)
    op = get_op(d)
    return op isa ThinOp ? Transformed(get_dist(d), ThinOp(vals.thin)) :
           Transformed(set_extra_leaf_params(get_dist(d), vals), op)
end

# --- get_dist: the composed `Shared` tag is transparent to the unwrap protocol

get_dist(d::Shared) = d.dist

# --- instantiate / has_varying: resolve a Varying leaf through a modifier ---
#
# A modifier wrapping a `Varying` leaf must peel through so the inner Varying
# resolves at a `Context`; without these methods `instantiate` falls to the
# identity `instantiate(::UnivariateDistribution, ctx) = d`, silently scoring
# against the reference, and `has_varying` falls to `false`, so the guard
# never fires. Mirrors the `Shared` descent.

function instantiate(d::Affine, ctx::AbstractContext)
    affine(instantiate(get_dist(d), ctx);
        scale = get_scale(d), shift = get_shift(d))
end
function instantiate(d::Weighted, ctx::AbstractContext)
    Weighted(instantiate(get_dist(d), ctx), get_weight(d))
end
function instantiate(d::Transformed, ctx::AbstractContext)
    Transformed(instantiate(get_dist(d), ctx), get_op(d))
end
function instantiate(d::Modified, ctx::AbstractContext)
    modify(instantiate(get_dist(d), ctx), get_effect(d); link = get_link(d))
end

has_varying(d::Affine) = has_varying(get_dist(d))
has_varying(d::Weighted) = has_varying(get_dist(d))
has_varying(d::Transformed) = has_varying(get_dist(d))
has_varying(d::Modified) = has_varying(get_dist(d))

end # module
