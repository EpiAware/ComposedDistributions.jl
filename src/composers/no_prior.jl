# ============================================================================
# NoPrior: the "free, no prior yet" spec marker
# ============================================================================
#
# `no_prior()` is a spec value alongside a `Distribution` and a `pool(...)`
# spec wherever `uncertain`/`update` accept one (`Uncertain.jl`'s `specs`, a
# `Resolve`'s `branch_prob_prior`): it says "this parameter is estimated" with
# no opinion on what its prior should be. ComposedDistributions describes
# structure — supports and which parameters are free (see the package
# docstring) — choosing a prior is DistributionsInference.jl's job, not this
# one's, so `uncertain(tree)` (bare) marks every free parameter `no_prior()`
# rather than guessing a default.
#
# Defined first, before any composer type, so every file below (`Resolve.jl`'s
# `branch_prob_prior`, `introspection.jl`'s row walk, `Uncertain.jl`'s specs,
# `codec_gen.jl`'s generated codec) can dispatch on it directly rather than
# widening a `Union` at the call site.
#
# The flat codec treats a marker spec exactly like an attached prior — it
# occupies one flat-vector slot and marks the parameter estimated (see
# `codec_gen.jl`: the generated `unflatten`/`flatten` walk is spec-value-blind
# outside the `Pool` case, so a marker needs no codec change at all). What it
# cannot do is draw a value: `Uncertain`'s `rand` refuses it, naming the
# parameter, and `Resolve`'s `rand` (`Resolve.jl`'s
# `_check_branch_probs_resolved`) carries the same guard on its
# `branch_prob_prior` (#366).

@doc raw"

A singleton marker: this parameter is estimated, with no prior chosen yet.

`no_prior()` builds the marker. It is a spec value alongside a `Distribution`
(a prior) and a [`pool`](@ref)`(...)` spec (partial pooling) wherever
[`uncertain`](@ref)/[`update`](@ref) accept one: it says a parameter is free
without saying what its prior is. Every place a spec drives structure
([`params_table`](@ref)'s `prior` column, [`has_uncertain`](@ref), the flat
codec) treats `NoPrior` exactly like an attached prior — it occupies a
flat-vector slot and marks the parameter estimated. What it cannot do is
*draw*: `rand` on an [`Uncertain`](@ref) leaf, or a [`Resolve`](@ref) node,
that still carries an unresolved `NoPrior` marker is refused, naming the
parameter (or, for `Resolve`, the node) and how to fix it (attach a prior, or
collapse first with [`update`](@ref)).

Bare `uncertain(tree)` (no `params`/keywords) is built on this marker: it
promotes every currently-fixed free parameter to `no_prior()`, the explicit
estimate-everything path with no guessed prior anywhere — including a
[`Resolve`](@ref)'s branch-probability simplex, which accepts `no_prior()` at
its `branch_probs` slot exactly like an attached `Dirichlet`.

# Examples
```@example
using ComposedDistributions, Distributions

# Mark a parameter estimated with no prior chosen yet.
u = uncertain(Gamma(2.0, 1.0); shape = no_prior())
has_uncertain(u)
```

# See also
- [`no_prior`](@ref): the constructor.
- [`uncertain`](@ref): attaches a spec (a prior, `no_prior()`, or `pool(...)`).
- [`Uncertain`](@ref): the leaf a spec attaches to.
"
struct NoPrior end

@doc raw"

Mark a parameter estimated, with no prior chosen yet.

`no_prior()` builds the [`NoPrior`](@ref) marker: a spec value wherever
[`uncertain`](@ref)/[`update`](@ref) accept one, saying a parameter is free
without choosing its prior. `uncertain(tree)` (bare) applies it to every
currently-fixed free parameter; write it explicitly to mark just one.

# Examples
```@example
using ComposedDistributions, Distributions

uncertain(Gamma(2.0, 1.0); shape = no_prior())
```

# See also
- [`NoPrior`](@ref): the marker type.
- [`uncertain`](@ref): attaches it (or a real prior) as a spec.
"
no_prior() = NoPrior()

Base.show(io::IO, ::NoPrior) = print(io, "no_prior()")

# A zero-field immutable struct is a singleton (every instance is `===`), so
# the default `==`/`hash` (which fall back to identity for a type with no
# fields to compare) already treat every `NoPrior()` as equal — no explicit
# methods needed.

# A marker spec counts as "distribution-valued" for `update`/`uncertain`'s
# merge-mode switch, exactly like a `Pool` spec (`Pool.jl`): writing
# `no_prior()` into a slot introduces a spec, not a plain value.
_has_distribution_value(::NoPrior) = true

# There is nothing to draw from an unresolved marker. `Uncertain`'s own `rand`
# (`Uncertain.jl`) checks for this up front and names the parameter in its
# error, so this method is reached only if a marker reaches `rand` some other
# way (e.g. nested inside a `Pool` population) — a defensive fallback, not the
# primary error path.
function Base.rand(::AbstractRNG, ::NoPrior)
    throw(ArgumentError(
        "cannot draw from no_prior(): it marks a parameter estimated with " *
        "no prior chosen yet; attach one with uncertain(tree; param = " *
        "prior, ...) or update(tree, table) before drawing"))
end
