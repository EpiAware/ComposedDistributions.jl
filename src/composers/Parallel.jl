@doc raw"

Independent branches composed from any univariate distributions.

`Parallel` places ``n`` branch distributions ``D_1, \dots, D_n`` off one origin,
with the realisation the vector of branch values ``[v_1, \dots, v_n]``. A branch
may itself be a [`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref),
[`Compete`](@ref) or [`Choose`](@ref) composer, so trees nest recursively and
the nesting is the tree.

`logpdf` is the sum of the per-branch log-densities,

```math
\log f(v_1, \dots, v_n) = \sum_{i=1}^{n} \log f_{D_i}(v_i).
```

The branches are independent here: this is the plain generic composition. The
shared-origin coupling (where every branch shares one latent primary event) is a
censored specialisation layered on top elsewhere, not part of this type.

# Fields
- `components`: tuple of the branch distributions (each univariate or a nested
  composer).
- the branch names (`Symbol`s), one per component, live in the `names` type
  parameter (read with [`component_names`](@ref)); the `compose` front-ends
  thread the user's names through, positional construction assigns
  `:branch_1, :branch_2, ...`.

# See also
- [`Sequential`](@ref): a chain of additive steps
- [`Resolve`](@ref): exactly one of several outcomes
"
struct Parallel{names, C <: Tuple} <:
       AbstractMultiChild{Continuous}
    "Tuple of the branch distributions (each univariate or a nested composer)."
    components::C

    function Parallel{names}(components::C) where {names, C <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Parallel needs at least one branch"))
        all(_is_composable, components) ||
            throw(ArgumentError(
                "every Parallel branch must be a UnivariateDistribution or " *
                "a nested composer"))
        length(names) == length(components) ||
            throw(ArgumentError(
                "Parallel names must match the number of components"))
        all(n -> n isa Symbol, names) ||
            throw(ArgumentError("every Parallel name must be a Symbol"))
        allunique(names) ||
            throw(ArgumentError("Parallel branch names must be unique"))
        new{names, C}(components)
    end
end

# The names live in the `names` type parameter (like `NamedTuple{names}`); this
# instantiates it directly from the runtime tuple `compose`/`parallel` pass,
# with no call-site change from the field-based constructor.
function Parallel(components::C, names::NTuple{N, Symbol}) where {
        C <: Tuple, N}
    return Parallel{names}(components)
end

# Positional construction assigns default `:branch_i` names.
function Parallel(components::C) where {C <: Tuple}
    return Parallel(components, _default_names(:branch, length(components)))
end

# A zero-arg call has no method through the variadic `Parallel(c1, cs...)`
# front-end below, so it would otherwise raise a bare `MethodError` rather
# than the inner constructor's friendly "needs at least one branch".
Parallel() = throw(ArgumentError("Parallel needs at least one branch"))

@doc "

Compose univariate distributions into [`Parallel`](@ref) branches.

Each argument is a branch distribution; the realisation is the vector of branch
values. Pass branches as positional arguments or a single vector/tuple. Any
[`Sequential`](@ref), [`Resolve`](@ref), [`Compete`](@ref) or [`Choose`](@ref)
child nests.

# Examples
```@example
using ComposedDistributions, Distributions

d = Parallel(Gamma(2.0, 1.0), LogNormal(1.0, 0.5))
rand(d)
```

# See also
- [`Parallel`](@ref): the composer type
"
Parallel(c1, cs...) = Parallel((c1, cs...))
Parallel(components::AbstractVector) = Parallel(Tuple(components))

@doc "

Compose univariate distributions into [`Parallel`](@ref) branches.

Lowercase verb mirroring [`sequential`](@ref) / [`resolve`](@ref): the public
constructor for a [`Parallel`](@ref) branch set. Pass branch distributions
positionally (default names `:branch_1, :branch_2, ...`) or `name => dist` pairs
to name the branches; a branch may itself be a [`Sequential`](@ref),
[`Resolve`](@ref), [`Compete`](@ref), [`Choose`](@ref) or nested set. Prefer
this verb over the bare struct constructor.

# Arguments
- `branches`: the branch distributions, either as positional distributions or as
  `name => dist` pairs naming each branch. A single named tuple
  `(name = dist, …)` is the equivalent positional spelling for hand-written
  branches; use Pairs for data-driven or computed names.

# Examples
```@example
using ComposedDistributions, Distributions

d = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
event_names(d)
```

```@example
using ComposedDistributions, Distributions

# The equivalent named tuple spelling.
d = parallel((admit = Gamma(2.0, 1.0), notif = LogNormal(1.0, 0.5)))
event_names(d)
```

# See also
- [`Parallel`](@ref): the composer type
- [`sequential`](@ref), [`resolve`](@ref), [`compete`](@ref): the sibling constructors
- [`compose`](@ref): the NamedTuple/table/matrix front-end
"
function parallel(branches::Pair...)
    length(branches) >= 1 ||
        throw(ArgumentError("parallel needs at least one branch"))
    names = Tuple(b.first for b in branches)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each parallel branch name must be a Symbol"))
    dists = Tuple(b.second for b in branches)
    return Parallel(dists, names)
end

parallel(b1, bs...) = Parallel((b1, bs...))
parallel(branches::AbstractVector) = Parallel(Tuple(branches))

# Positional NamedTuple spelling: `(a = d1, …)` lowers to `:a => d1, …` Pairs.
parallel(branches::NamedTuple) = parallel(_nt_pairs(branches)...)

# Total number of leaf values in a realisation (sum over nested children).
Base.length(d::Parallel) = _nleaves(d.components)

function Base.eltype(::Type{<:Parallel{names, C}}) where {names, C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

# Branch names, one per component.
component_names(::Parallel{names}) where {names} = names

@doc "

Nested, name-keyed parameters of the branches.

Returns a `NamedTuple` keyed by the branch names, each value the `params` of that
branch (recursing into nested composers; a leaf delegates to its standard/
extended `Distributions.params`). This nested form is for prior introspection
via [`params_table`](@ref); a composed distribution reconstructs through
[`compose`](@ref), not through `Distribution(params...)`.

See also: [`params_table`](@ref), [`event_names`](@ref), [`event`](@ref)
"
params(d::Parallel) = _composed_params(d)

@doc "

Log probability density of a branch-value vector, summed over branches.

See also: [`Parallel`](@ref)
"
function logpdf(d::Parallel, x::AbstractVector)
    length(x) == length(d) || _throw_logpdf_dimmismatch(d, x, "branch")
    return _composite_logpdf(d.components, x)
end

@doc "

Log probability density of a branch-value vector admitting `missing` branches:
`missing` in a slot means that branch was not observed (the ecosystem-wide
convention), scoring the observed branches and integrating out the rest (each
unobserved branch's own marginal contributes zero log density).

See also: [`Parallel`](@ref)
"
function logpdf(d::Parallel, x::AbstractVector{>:Missing})
    length(x) == length(d) || _throw_logpdf_dimmismatch(d, x, "branch")
    return _composite_logpdf(d.components, x)
end

@doc "

Probability density of a branch-value vector.

See also: [`logpdf`](@ref)
"
pdf(d::Parallel, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Sample a branch realisation as a `NamedTuple` keyed by the per-branch value
names: one entry per leaf branch, a nested
[`Sequential`](@ref)/[`Parallel`](@ref) branch contributing its own sub-values
under dotted-joined names, and a [`Resolve`](@ref) branch contributing its own
collapsed scalar.

See also: [`Parallel`](@ref)
"
Base.rand(rng::AbstractRNG, d::Parallel) = _named_composer_rand(rng, d)

Base.rand(d::Parallel) = rand(default_rng(), d)
sampler(d::Parallel) = d

@doc "

Print a [`Parallel`](@ref) composer as a recursive indented tree, descending
into any nested composer children so the whole structure is shown at once.

See also: [`Parallel`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Parallel)
    _show_composer_tree(io, d)
    return nothing
end

function Base.show(io::IO, d::Parallel)
    print(io, "Parallel(", join(string.(d.components), " | "), ")")
    return nothing
end
