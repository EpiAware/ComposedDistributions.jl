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
[`prune`](@ref) / [`splice`](@ref). Attach parameter uncertainty with
[`uncertain`](@ref) (parameters that are themselves distributions, nestable):
`rand` draws the marginal, and [`update`](@ref) collapses an uncertain leaf to
its concrete template.

Hard-deps `ConvolvedDistributions` (a chain collapses to a convolved total via
[`observed_distribution`](@ref)) and extends its `convolve_series`/`difference`
generics for composed tree types; its own convolution/quadrature surface
(`convolved`, `integrate`/`gl_integrate`, the solver-method types) is reached
with a separate `using ConvolvedDistributions`, not re-exported here. No
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
                     Continuous, Multivariate, Univariate, VariateForm,
                     ValueSupport, MixtureModel, Truncated, truncated, censored

using LogExpFunctions: log1mexp

import Tables

# The convolution + quadrature substrate CD's own interop methods build on
# (see composers/convolved_interop.jl and hazard_one_of.jl) — not re-exported
# (#228): a caller reaches ConvolvedDistributions' own surface (`convolved`,
# `product`, `discretise_pmf`, the solver types, ...) with its own
# `using ConvolvedDistributions`. Only the names CD extends or constructs
# internally are imported here.
using ConvolvedDistributions: ConvolvedDistributions, convolved,
                              convolve_series, Difference, difference,
                              GaussLegendre, integrate, Convolved
# AD-safe survival helpers, now owned by EpiAwareADTools (ConvolvedDistributions
# 0.2 moved the `*_ad_safe` family out under underscore-free names, #137).
# `logccdf_ad_safe` is called by the racing-hazard node and both it and
# `ccdf_ad_safe` are extended for `Compete` (the methods are defined
# fully-qualified in Compete.jl, so the module name is imported too).
using EpiAwareADTools: EpiAwareADTools, logccdf_ad_safe

# Docstring-template helpers, imported here (centralised) and used by the
# `@template` blocks in src/docstrings.jl.
using DocStringExtensions: @template, DOCSTRING, EXPORTS, IMPORTS, TYPEDEF,
                           TYPEDFIELDS, TYPEDSIGNATURES

# Register the standard EpiAware docstring conventions before any
# docstrings are defined (see src/docstrings.jl).
include("docstrings.jl")

# The composer abstract-type hierarchy (`AbstractComposedDistribution` /
# `AbstractMultiChild`). Before the composers, which subtype it.
include("interface.jl")

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

# Parameter uncertainty: a leaf whose parameters are themselves distributions
# (nestable). `rand` draws the marginal; `update(tree, params)` collapses an
# uncertain leaf to its concrete template. `has_uncertain` flags a tree that
# still holds one, for a scoring/fitting loop to guard against a forgotten
# `update` (the rest of the surface silently reports the template's values).
export Uncertain, uncertain, has_uncertain, @uncertain

# Event-skeleton topology: `@events` declares an event tree's STRUCTURE (named
# holes joined by → / | / & operators) with no distributions attached;
# `update(skeleton; name = dist, ...)` fills the holes and builds the concrete
# composed tree through the existing verbs (a `|` node becomes a `Resolve` or
# `Compete` decided by the fill value type).
export EventSkeleton, @events

# Partial pooling: a parameter drawn, across the leaves of a group, from one
# shared population distribution whose free parameters are the estimated
# hyperparameters. `pool(:group, population)` is a spec inside `uncertain`; a
# location-scale population lowers non-centred (hyperparameters + one
# `Normal(0, 1)` latent per member), a general population is scored centred. The
# middle of the pooling spectrum between `shared`/`tie` (complete) and
# independent `uncertain` (none).
export Pool, pool

# Context-indexed (non-stationary) leaves: a `Varying` leaf varies with a
# covariate (time, strata, ...); `instantiate(tree, Context(...))` resolves a tree
# against a context to a concrete stationary tree.
export Varying, varying, Context, AbstractContext, instantiate, with_covariates,
       has_varying

# Introspection: the flat prior table and name introspection. `event_names` is
# the flat per-event name tuple; `event_tree` the nested tree of event names;
# `event` fetches a child or descends a path. `param_priors` is the tree-level
# front-door over `build_priors`; `inspect` is the opt-in detailed tree print.
export params_table, event_names, event_tree, event, update, build_priors,
       default_prior, param_priors, inspect

# Structural edits on a composed tree. `update` (the `path => new_node` method)
# replaces a named node keeping the shape; `prune` drops a branch and `splice`
# inserts a step (topology edits).
export prune, splice

# Inference-readback verbs: read a fitted chain's parameters back onto a
# composed-distribution template. `chain_to_params` reduces a chain to the
# nested NamedTuple `update` consumes; `param_draws` keeps every draw;
# `strip_prefix` drops the outer submodel prefix from a chain's parameter
# names. No method until both `DynamicPPL` and `FlexiChains` are loaded; the
# methods live in `ext/ComposedDistributionsFlexiChainsExt.jl`.
export chain_to_params, param_draws, strip_prefix

export observed_distribution

# --- includes --------------------------------------------------------------

include("composers/Sequential.jl")
include("composers/Parallel.jl")
include("composers/Resolve.jl")
# Racing-hazard one_of node (the `min`-of-delays dual of convolve). After
# Resolve since it builds on `AbstractOneOf` / the `_n_branches` / `_is_no_event`
# helpers.
include("composers/Compete.jl")
include("composers/Choose.jl")
# Shared nesting machinery, defined once all composer types exist.
include("composers/nesting.jl")
include("composers/equality.jl")
include("composers/compose.jl")
include("composers/introspection.jl")
# Inference-readback verb stubs (`chain_to_params` / `param_draws` /
# `strip_prefix`): declared here with no method (see
# `ext/ComposedDistributionsFlexiChainsExt.jl`), so this package stays
# Turing-free until that extension is triggered.
include("composers/readback.jl")
# Structural edits (`update` node replace / `prune` / `splice`): after
# introspection so it reuses `_rebuild`, `component_names`, `_split_edge` and
# the `update` value method.
include("composers/structural_edits.jl")
# Event-skeleton topology + the `@events` macro. After introspection (the fill
# adds an `update(::EventSkeleton; ...)` method to the `update` generic) and the
# composer verbs (`sequential` / `parallel` / `resolve` / `compete`) it lowers
# to. The macro file loads after the spec types it references.
include("composers/events.jl")
include("composers/events_macro.jl")
# Shared (name-tagged tied leaf): after introspection so it can extend
# `free_leaf`/`rewrap_leaf`, and after the structural edits (reuses `_edit_at`).
include("composers/Shared.jl")
# Uncertain (distribution-valued parameters): after introspection (extends
# `free_leaf` / `rewrap_leaf` / `_uncertain_specs`, reuses `_update_leaf` /
# `_rebuild`) and Shared (forwards `_shared_tag` / `_uncertain_specs` through
# the tag wrapper).
include("composers/Uncertain.jl")
# The `@uncertain` syntax front-end over the positional `uncertain` family
# form: pure `Expr` rewriting, so it loads right after the constructor it emits.
include("composers/uncertain_macro.jl")
# Partial pooling (a parameter pooled, across a group of leaves, through one
# estimated population). After Uncertain (a `Pool` rides an uncertain leaf's
# specs) and introspection (it reuses `_node_children`/`_split_edge`/`_join_path`
# /`_update_leaf` and the `_walk_rows!` / `_update` leaf hooks).
include("composers/Pool.jl")
# Context-indexed leaves + the `instantiate` resolution seam. After every
# composer type exists (it rebuilds Sequential/Parallel/Choose/Resolve/Compete/
# Shared against a context) and after introspection (it extends free_leaf/
# rewrap_leaf/_shared_tag for the Varying leaf).
include("composers/varying.jl")
# `Censored` (Distributions.jl's `censored(...)` wrapper) leaf-protocol parity
# with `Truncated`, plus the tree-level truncated/censored guard. After
# introspection.jl (extends free_leaf/rewrap_leaf/uncertain_specs/
# extra_leaf_params/set_extra_leaf_params), Shared.jl (shared_tag), Uncertain.jl
# (the `Uncertain` type, and the censored-pushes-inside method set) and
# varying.jl (has_varying), which it extends.
include("composers/wrapped_leaves.jl")
# The generated type-domain flat <-> nested codec (`unflatten`/`flatten`/
# `flat_dimension`/`reconstruct`, #178 PR 2). After every composer/wrapper
# type (Sequential/Parallel/Choose/Resolve/Compete/Uncertain/Shared/Pool) and
# varying.jl (`has_varying`, for the `_reject_varying` guard).
include("composers/codec_gen.jl")
# The Turing-free `ComposedLogDensity`/`as_logdensity`/`logdensity` core (no
# LogDensityProblems/DynamicPPL dependency; DistributionsInference.jl hosts the
# PPL-facing extensions on top of this via the fit protocol). After
# introspection (`params_table`/`build_priors`/`update`), Uncertain (an
# uncertain leaf's row is inventoried like any other) and codec_gen.jl (the
# flat <-> nested codec it evaluates against).
include("composers/logdensity.jl")
include("composers/tree_events.jl")
# Collapse a chain to its observed convolved total. After the composers.
include("composers/observed.jl")
# ConvolvedDistributions interop (vector convolution, difference, composite
# leaves): after observed.jl (the collapse), introspection.jl (the params_table
# / update walks it extends to see through a composite leaf) and Uncertain.jl
# (it extends `has_uncertain` for a composite carrying an uncertain component).
include("composers/convolved_interop.jl")
# Per-edge delay moments: after the composers it walks and observed.jl.
include("composers/composed_moments.jl")
# Labelled NamedTuple outputs + the generic realisation seam. Last: wraps the
# composers' vector-valued draws by name.
include("composers/named_outputs.jl")

# The reusable interface-conformance harness (`TestUtils.test_interface` and
# friends). Last: it uses the whole public surface defined above.
include("TestUtils.jl")

# Public API - functions that are part of the public interface but not exported
# (Julia 1.11+).
@static if VERSION >= v"1.11"
    include("public.jl")
end

end # module ComposedDistributions
