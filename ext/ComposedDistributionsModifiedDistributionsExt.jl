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
# `_thin_factor`/`_set_thin_factor` hooks, surfacing as a `:thin` row in
# `params_table` and round-tripping through `update`. This seam needs both
# packages' types, so it lives here (the function owner —
# ComposedDistributions for `free_leaf`/`rewrap_leaf`/`_shared_tag`/
# `_thin_factor`/`_set_thin_factor`, ModifiedDistributions for `get_dist` —
# plus at least one type at the seam, so no piracy).
module ComposedDistributionsModifiedDistributionsExt

import ComposedDistributions: free_leaf, rewrap_leaf, _shared_tag,
                              _uncertain_specs, _thin_factor, _set_thin_factor,
                              _leaf_mean, _leaf_var, instantiate, has_varying
using ComposedDistributions: Shared, AbstractContext
using Distributions: mean, var
import ModifiedDistributions: get_dist
using ModifiedDistributions: Affine, Weighted, Transformed, Modified,
                             ThinOp, affine, modify

# --- free_leaf: reach the inner free delay through a modifier ---------------

free_leaf(d::Affine) = free_leaf(d.dist)
free_leaf(d::Weighted) = free_leaf(d.dist)
free_leaf(d::Transformed) = free_leaf(d.dist)
free_leaf(d::Modified) = free_leaf(d.dist)

# --- rewrap_leaf: rebuild the modifier around a new inner delay --------------

function rewrap_leaf(d::Affine, inner)
    return affine(rewrap_leaf(d.dist, inner); scale = d.scale, shift = d.shift)
end
function rewrap_leaf(d::Weighted, inner)
    return Weighted(rewrap_leaf(d.dist, inner), d.weight)
end
function rewrap_leaf(d::Transformed, inner)
    return Transformed(rewrap_leaf(d.dist, inner), d.op)
end
function rewrap_leaf(d::Modified, inner)
    return modify(rewrap_leaf(d.dist, inner), d.effect; link = d.link)
end

# --- _shared_tag: see a shared tag through a modifier -----------------------

_shared_tag(d::Affine) = _shared_tag(d.dist)
_shared_tag(d::Weighted) = _shared_tag(d.dist)
_shared_tag(d::Transformed) = _shared_tag(d.dist)
_shared_tag(d::Modified) = _shared_tag(d.dist)

# --- _uncertain_specs: see uncertain parameters through a modifier ----------
#
# A modifier over an `uncertain(...)` leaf must still expose the attached
# parameter specs, so `params_table`'s prior column and the marginal `rand`
# see through the modifier exactly like the tag protocol does.

_uncertain_specs(d::Affine) = _uncertain_specs(d.dist)
_uncertain_specs(d::Weighted) = _uncertain_specs(d.dist)
_uncertain_specs(d::Transformed) = _uncertain_specs(d.dist)
_uncertain_specs(d::Modified) = _uncertain_specs(d.dist)

# --- overall moments: use the modifier's own moment, not the free leaf's -----
#
# The per-leaf moment defaults to `mean(free_leaf(leaf))`, which is the free
# delay's — right for the parameter surface, but wrong for a moment when the
# modifier itself changes the distribution. `Affine` has correct analytic
# moments (`scale*mean + shift`, `scale^2*var`) and its `rand` honours the
# transform, so its overall moment must too. `Weighted` / `Transformed` delegate
# their moments straight to the inner delay, so their free-leaf moment already
# agrees — no method needed.

_leaf_mean(d::Affine) = mean(d)
_leaf_var(d::Affine) = var(d)

# `Modified` has no analytic moment yet (blocked on ModifiedDistributions#44's
# numeric cumulative-hazard path), and `free_leaf` peels it to the inner delay —
# so the default `mean(free_leaf(d))` would silently return the UNMODIFIED
# delay's moment, understating the hazard modification. Error informatively
# instead: a chain containing a hazard-modified step has no overall moment until
# #44 lands (draw the marginal with `rand` meanwhile).
function _leaf_mean(d::Modified)
    throw(ArgumentError(
        "a hazard-modified (`Modified`) leaf has no analytic mean; the " *
        "modified moment needs numeric cumulative-hazard integration " *
        "(ModifiedDistributions#44). Draw the marginal with `rand` for a " *
        "Monte-Carlo moment, or exclude the modified step."))
end
function _leaf_var(d::Modified)
    throw(ArgumentError(
        "a hazard-modified (`Modified`) leaf has no analytic variance; the " *
        "modified moment needs numeric cumulative-hazard integration " *
        "(ModifiedDistributions#44). Draw the marginal with `rand` for a " *
        "Monte-Carlo moment, or exclude the modified step."))
end

# --- _thin_factor / _set_thin_factor: surface a thin(...) reporting
# probability as a free parameter -------------------------------------------
#
# `thin(d, p)` (a `Transformed` carrying a `ThinOp`) is NOT a fixed-structure
# modifier like the others: `p` enters the per-record likelihood, so it must
# be inventoried by `params_table` and round-tripped by `update` like any
# other leaf parameter (see `src/composers/introspection.jl`'s hook
# docstring, and CensoredDistributions' `forward_transform.jl` for the
# precedent this mirrors). The peel-through modifiers (`Affine`/`Weighted`/
# `Modified`) forward to their inner delay so a thinned leaf still reports its
# factor underneath any of them.

_thin_factor(d::Affine) = _thin_factor(d.dist)
_thin_factor(d::Weighted) = _thin_factor(d.dist)
_thin_factor(d::Modified) = _thin_factor(d.dist)
function _thin_factor(d::Transformed)
    return d.op isa ThinOp ? d.op.factor : _thin_factor(d.dist)
end

function _set_thin_factor(d::Affine, p)
    return affine(_set_thin_factor(d.dist, p); scale = d.scale, shift = d.shift)
end
function _set_thin_factor(d::Weighted, p)
    return Weighted(_set_thin_factor(d.dist, p), d.weight)
end
function _set_thin_factor(d::Modified, p)
    return modify(_set_thin_factor(d.dist, p), d.effect; link = d.link)
end
function _set_thin_factor(d::Transformed, p)
    return d.op isa ThinOp ? Transformed(d.dist, ThinOp(p)) :
           Transformed(_set_thin_factor(d.dist, p), d.op)
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
    affine(
        instantiate(d.dist, ctx); scale = d.scale, shift = d.shift)
end
function instantiate(d::Weighted, ctx::AbstractContext)
    Weighted(instantiate(d.dist, ctx), d.weight)
end
function instantiate(d::Transformed, ctx::AbstractContext)
    Transformed(instantiate(d.dist, ctx), d.op)
end
function instantiate(d::Modified, ctx::AbstractContext)
    modify(
        instantiate(d.dist, ctx), d.effect; link = d.link)
end

has_varying(d::Affine) = has_varying(d.dist)
has_varying(d::Weighted) = has_varying(d.dist)
has_varying(d::Transformed) = has_varying(d.dist)
has_varying(d::Modified) = has_varying(d.dist)

end # module
