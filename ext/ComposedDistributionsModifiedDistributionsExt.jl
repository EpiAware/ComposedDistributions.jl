# ComposedDistributions × ModifiedDistributions
#
# A modified leaf (`Affine` / `Weighted` / `Transformed`) is a delay wrapper: its
# modification (scale/shift, likelihood weight, forward transform) is fixed
# structure, not a free parameter. So inside a composed tree it must peel like any
# other leaf wrapper — `free_leaf` reaches the inner free delay, `rewrap_leaf`
# rebuilds the modifier around a new inner delay, and `_shared_tag` sees a tag
# through it. This seam needs both packages' types, so it lives here (the
# function owner — ComposedDistributions for `free_leaf`/`rewrap_leaf`/
# `_shared_tag`, ModifiedDistributions for `get_dist` — plus at least one type at
# the seam, so no piracy).
module ComposedDistributionsModifiedDistributionsExt

import ComposedDistributions: free_leaf, rewrap_leaf, _shared_tag,
                              _uncertain_specs
using ComposedDistributions: Shared
import ModifiedDistributions: get_dist
using ModifiedDistributions: Affine, Weighted, Transformed, affine

# --- free_leaf: reach the inner free delay through a modifier ---------------

free_leaf(d::Affine) = free_leaf(d.dist)
free_leaf(d::Weighted) = free_leaf(d.dist)
free_leaf(d::Transformed) = free_leaf(d.dist)

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

# --- _shared_tag: see a shared tag through a modifier -----------------------

_shared_tag(d::Affine) = _shared_tag(d.dist)
_shared_tag(d::Weighted) = _shared_tag(d.dist)
_shared_tag(d::Transformed) = _shared_tag(d.dist)

# --- _uncertain_specs: see uncertain parameters through a modifier ----------
#
# A modifier over an `uncertain(...)` leaf must still expose the attached
# parameter specs, so `params_table`'s prior column and the marginal `rand`
# see through the modifier exactly like the tag protocol does.

_uncertain_specs(d::Affine) = _uncertain_specs(d.dist)
_uncertain_specs(d::Weighted) = _uncertain_specs(d.dist)
_uncertain_specs(d::Transformed) = _uncertain_specs(d.dist)

# --- get_dist: the composed `Shared` tag is transparent to the unwrap protocol

get_dist(d::Shared) = d.dist

end # module
