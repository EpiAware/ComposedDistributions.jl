@doc raw"

A chain of independent steps composed from any univariate distributions.

`Sequential` links events ``E_0 \to E_1 \to \dots \to E_k`` through independent
step distributions ``D_1, \dots, D_k``. A realisation is the flat vector of step
values ``[v_1, \dots, v_k]`` (one value per step). A step may itself be a
[`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
or [`Choose`](@ref) composer, in which case it contributes its own flat
sub-vector, so chains nest recursively and the nesting is the tree.

`logpdf` sums the per-step log-densities over the matching slices of the value
vector:

```math
\log f(v_1, \dots, v_k) = \sum_{i=1}^{k} \log f_{D_i}(v_i).
```

This is the plain generic composition; censoring and per-record marginalisation
are not part of this type. Cumulative event times, if wanted, are the running
sum of the step values.

# Fields
- `components`: tuple of the step distributions (each univariate or a nested
  composer).
- `names`: tuple of the step names (`Symbol`s), one per component; the `compose`
  front-ends thread the user's names through, positional construction assigns
  `:step_1, :step_2, ...`.

# See also
- [`Parallel`](@ref): independent branches
- [`Resolve`](@ref): exactly one of several outcomes
"
struct Sequential{C <: Tuple, N <: Tuple} <:
       AbstractMultiChild{Continuous}
    "Tuple of the step distributions ``D_1, \\dots, D_k`` (each univariate or a
    nested composer)."
    components::C
    "Tuple of the step names (`Symbol`s), one per component. The `compose`
    NamedTuple front-end uses the user's keys; positional construction assigns
    `:step_1, :step_2, ...`."
    names::N

    function Sequential(components::C, names::N) where {C <: Tuple, N <: Tuple}
        length(components) >= 1 ||
            throw(ArgumentError("Sequential needs at least one component"))
        all(_is_composable, components) ||
            throw(ArgumentError(
                "every Sequential component must be a UnivariateDistribution " *
                "or a nested composer"))
        length(names) == length(components) ||
            throw(ArgumentError(
                "Sequential names must match the number of components"))
        all(n -> n isa Symbol, names) ||
            throw(ArgumentError("every Sequential name must be a Symbol"))
        allunique(names) ||
            throw(ArgumentError("Sequential step names must be unique"))
        new{C, N}(components, names)
    end
end

# Positional construction assigns default `:step_i` names.
function Sequential(components::C) where {C <: Tuple}
    return Sequential(components, _default_names(:step, length(components)))
end

# A zero-arg call has no method through the variadic `Sequential(c1, cs...)`
# front-end below, so it would otherwise raise a bare `MethodError` rather
# than the inner constructor's friendly "needs at least one component".
Sequential() = throw(ArgumentError("Sequential needs at least one component"))

@doc "

Compose univariate distributions into a [`Sequential`](@ref) chain.

Each argument is a step distribution; the realisation is the vector of
cumulative event times. Pass components as positional arguments or a single
vector/tuple. Any [`Parallel`](@ref), [`Resolve`](@ref), [`Compete`](@ref) or
[`Choose`](@ref) child nests.

# Examples
```@example
using ComposedDistributions, Distributions

d = Sequential(Gamma(2.0, 1.0), LogNormal(0.5, 0.4))
rand(d)
```

# See also
- [`Sequential`](@ref): the composer type
"
Sequential(c1, cs...) = Sequential((c1, cs...))
Sequential(components::AbstractVector) = Sequential(Tuple(components))

@doc "

Compose univariate distributions into a [`Sequential`](@ref) chain.

Lowercase verb mirroring [`parallel`](@ref) / [`resolve`](@ref): the public
constructor for a [`Sequential`](@ref) chain. Pass step distributions
positionally (default names `:step_1, :step_2, ...`) or `name => dist` pairs to
name the steps; a step may itself be a [`Parallel`](@ref), [`Resolve`](@ref),
[`Compete`](@ref), [`Choose`](@ref) or nested chain. Prefer this verb over the
bare struct constructor.

# Arguments
- `steps`: the step distributions, either as positional distributions or as
  `name => dist` pairs naming each step.

# Examples
```@example
using ComposedDistributions, Distributions

d = sequential(:onset_admit => Gamma(2.0, 1.0),
    :admit_death => LogNormal(0.5, 0.4))
event_names(d)
```

# See also
- [`Sequential`](@ref): the composer type
- [`parallel`](@ref), [`resolve`](@ref), [`compete`](@ref): the sibling constructors
- [`compose`](@ref): the NamedTuple/table/matrix front-end
"
function sequential(steps::Pair...)
    length(steps) >= 1 ||
        throw(ArgumentError("sequential needs at least one step"))
    names = Tuple(s.first for s in steps)
    all(n -> n isa Symbol, names) ||
        throw(ArgumentError("each sequential step name must be a Symbol"))
    dists = Tuple(s.second for s in steps)
    return Sequential(dists, names)
end

sequential(s1, ss...) = Sequential((s1, ss...))
sequential(steps::AbstractVector) = Sequential(Tuple(steps))

# Lower a positional NamedTuple to the `name => value` Pairs the verb
# constructors take, so a verb accepts both spellings from one Pairs path. The
# NamedTuple is ordered, so the lowered Pairs keep field order; it is positional
# (not kwargs), so it never clashes with a config keyword such as `choose`'s
# `selector`. Shared by `resolve` / `compete` / `choose`.
_nt_pairs(nt::NamedTuple) = map(=>, keys(nt), values(nt))

# Total number of leaf values in a realisation (sum over nested children).
Base.length(d::Sequential) = _nleaves(d.components)

function Base.eltype(::Type{<:Sequential{C}}) where {C <: Tuple}
    return mapreduce(eltype, promote_type, fieldtypes(C))
end

@doc "

The child names of a composed distribution.

Returns the tuple of names for a composer's direct children: the step names of a
[`Sequential`](@ref) chain, the branch names of a [`Parallel`](@ref) set, or the
outcome names of a [`Resolve`](@ref) node. These EDGE names key the parameter
inventory, distinct from the flat EVENT names of `_flat_event_names`.

# Examples
```@example
using ComposedDistributions, Distributions

oa = LogNormal(1.5, 0.4)
ad = Gamma(2.0, 1.0)
tree = compose((onset_admit = [oa, ad],))
ComposedDistributions.component_names(tree)
```

# See also
- [`event_names`](@ref): the public EDGE-name accessor
- `_flat_event_names`: the flat EVENT names
"
component_names(d::Sequential) = d.names

@doc "

Nested, name-keyed parameters of the chain.

Returns a `NamedTuple` keyed by the step names, each value the `params` of that
step (recursing into nested composers; a leaf delegates to its standard/extended
`Distributions.params`). This nested form is for prior introspection via
[`params_table`](@ref); a composed distribution reconstructs through
[`compose`](@ref), not through `Distribution(params...)`.

See also: [`params_table`](@ref), [`event_names`](@ref), [`event`](@ref)
"
params(d::Sequential) = _composed_params(d)

@doc "

Log probability density of a chain's step-value vector.

See also: [`Sequential`](@ref)
"
function logpdf(d::Sequential, x::AbstractVector)
    length(x) == length(d) || throw(DimensionMismatch(
        "expected $(length(d)) step values, got $(length(x))"))
    return _composite_logpdf(d.components, x)
end

@doc "

Probability density of a chain's step-value vector.

See also: [`logpdf`](@ref)
"
pdf(d::Sequential, x::AbstractVector) = exp(logpdf(d, x))

@doc "

Sample a chain realisation as a `NamedTuple` keyed by the per-step value
names: one entry per leaf step, a nested [`Sequential`](@ref)/[`Parallel`](@ref)
step contributing its own sub-values under dotted-joined names, and a
[`Resolve`](@ref) step contributing its own collapsed scalar.

See also: [`Sequential`](@ref)
"
Base.rand(rng::AbstractRNG, d::Sequential) = _named_composer_rand(rng, d)

Base.rand(d::Sequential) = rand(default_rng(), d)
sampler(d::Sequential) = d

@doc "

Print a [`Sequential`](@ref) chain as a recursive indented tree, descending
into any nested composer children so the whole structure is shown at once.

See also: [`Sequential`](@ref)
"
function Base.show(io::IO, ::MIME"text/plain", d::Sequential)
    _show_composer_tree(io, d)
    return nothing
end

function Base.show(io::IO, d::Sequential)
    print(io, "Sequential(", join(string.(d.components), " -> "), ")")
    return nothing
end
