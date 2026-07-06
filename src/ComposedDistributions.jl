"""
    ComposedDistributions

The verb grammar for n-ary composition over any `Distributions.jl`
`UnivariateDistribution`. Compose delays into chains ([`Sequential`](@ref)),
independent branches ([`Parallel`](@ref)), fixed-probability or racing one_of
outcomes ([`Resolve`](@ref) / [`Compete`](@ref)) and data-selected disjunctions
([`Choose`](@ref)); the [`compose`](@ref) front-end lowers a NamedTuple, a
Tables.jl table, or a nested matrix to the same stack. Read the structure with
[`params_table`](@ref) / [`event_names`](@ref) / [`event`](@ref), build priors
with [`build_priors`](@ref), and edit the tree with [`update`](@ref) /
[`prune`](@ref) / [`splice`](@ref).

Hard-deps and re-exports `ConvolvedDistributions` (a chain collapses to a
convolved total via [`observed_distribution`](@ref)), so its convolution and
quadrature surface — [`convolve_distributions`](@ref), `integrate`/`gl_integrate`
and the solver-method types — is reachable through this package alone. No
censoring: this is the generic composition layer.

# Examples
```@example
using ComposedDistributions, Distributions

# A two-step delay chain, then its parameter table.
tree = compose((onset_admit = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],))
params_table(tree)
```
"""
module ComposedDistributions

using Random: AbstractRNG, default_rng

# Functions extended with new methods.
import Distributions: params, insupport, pdf, logpdf, cdf, logcdf,
                      ccdf, logccdf, quantile, mean, var, std, sampler, probs
import Base: minimum, maximum

# Types, constructors, and helpers used without method extension.
using Distributions: Distributions, UnivariateDistribution, Distribution,
                     Continuous, Multivariate, ValueSupport, MixtureModel,
                     Truncated, truncated

using LogExpFunctions: log1mexp

using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF,
                           TYPEDFIELDS, TYPEDSIGNATURES

import Tables

# The convolution + quadrature substrate. Re-exported below so downstream
# packages sit on ComposedDistributions alone. Every name is imported explicitly
# (the exported surface plus the public-but-unexported quadrature helpers).
using ConvolvedDistributions: ConvolvedDistributions, convolve_distributions,
                              Difference, difference, AnalyticalSolver,
                              NumericSolver, gl_integrate, GaussLegendre,
                              integrate, AbstractSolverMethod, Convolved
# AD-safe survival helper. Called by the racing-hazard node and extended for it
# (the `Compete` methods are defined fully-qualified in hazard_one_of.jl). This
# is an upstream internal, so it is listed in `ei_ignore` in the QA config.
using ConvolvedDistributions: _logccdf_ad_safe

# Register the standard EpiAware docstring conventions before any
# docstrings are defined (see src/docstrings.jl).
include("docstrings.jl")

# --- exports ---------------------------------------------------------------

# Generic composers and front-end constructors. `resolve(...)` builds the fixed-
# probability mixture `Resolve`; `compete(...)` builds the racing-hazard
# `Compete`. `NoEvent` marks an absorbing no-event branch; `Distributions.probs`
# (extended, not re-exported) reads the per-outcome split of either node and
# `occurrence_probability` its sum (the any-event probability).
export Sequential, Parallel, Resolve, Compete, NoEvent,
       sequential, parallel, compete, resolve,
       compose, as_mixture, occurrence_probability

# Data-selected disjunction (case selector over independent alternatives).
export Choose, choose

# Shared-parameter tie: tie a leaf across branches by name so the prior/params
# interface treats its occurrences as one free parameter.
export Shared, shared, tie

# Introspection: the flat prior table and name introspection. `event_names` is
# the flat per-event name tuple; `event_tree` the nested tree of event names;
# `event` fetches a child or descends a path.
export params_table, event_names, event_tree, event, update, build_priors,
       default_prior

# Structural edits on a composed tree. `update` (the `path => new_node` method)
# replaces a named node keeping the shape; `prune` drops a branch and `splice`
# inserts a step (topology edits). `intervene` / `swap_child` / `cut_branch` are
# deprecated aliases.
export prune, splice, intervene, swap_child, cut_branch

export observed_distribution

# Re-exported ConvolvedDistributions surface, so downstream packages reach
# convolution + quadrature through ComposedDistributions alone.
export convolve_distributions, Difference, difference,
       AnalyticalSolver, NumericSolver, Convolved, AbstractSolverMethod,
       GaussLegendre, integrate, gl_integrate

# --- includes --------------------------------------------------------------

include("composers/Sequential.jl")
include("composers/Parallel.jl")
include("composers/Resolve.jl")
# Racing-hazard one_of node (the `min`-of-delays dual of convolve). After
# Resolve since it builds on `AbstractOneOf` / the `_n_branches` / `_is_no_event`
# helpers.
include("composers/hazard_one_of.jl")
include("composers/Choose.jl")
# Shared nesting machinery, defined once all composer types exist.
include("composers/nesting.jl")
include("composers/equality.jl")
include("composers/compose.jl")
include("composers/introspection.jl")
# Structural edits (`update` node replace / `prune` / `splice`): after
# introspection so it reuses `_rebuild`, `component_names`, `_split_edge`.
include("composers/intervene.jl")
# Shared (name-tagged tied leaf): after introspection (extends `free_leaf` /
# `rewrap_leaf`) and intervene (reuses `_edit_at`).
include("composers/Shared.jl")
include("composers/tree_events.jl")
# Collapse a chain to its observed convolved total. After the composers.
include("composers/observed.jl")
# Per-edge delay moments: after the composers it walks and observed.jl.
include("composers/composed_moments.jl")
# Labelled NamedTuple outputs + the generic realisation seam. Last: wraps the
# composers' vector-valued draws by name.
include("composers/named_outputs.jl")

# Public API - functions that are part of the public interface but not exported
# (Julia 1.11+).
@static if VERSION >= v"1.11"
    include("public.jl")
end

end # module ComposedDistributions
