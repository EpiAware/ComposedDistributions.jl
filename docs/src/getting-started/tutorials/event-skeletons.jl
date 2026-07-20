# # [Event skeletons: reusable topologies with `@events`](@id event-skeletons)
#
# ## Introduction
#
# The composers in [Composing distributions](@ref composing-distributions) build
# one tree in one call: name each branch, wire it, and its distributions are
# baked in on the spot.
# That is fine for a single fit, but a natural-history topology is often shared:
# the same onset-admission-outcome pathway describes several pathogens, or the
# same reporting structure repeats across regions, each time with different
# delay parameters.
# Rebuilding the nested `sequential`/`resolve` calls for every variant
# duplicates the topology and buries it inside the distribution literals.
#
# [`@events`](@ref) separates the two: write the topology once as a readable
# operator diagram, an [`EventSkeleton`](@ref) with named holes and no
# distributions, then fill the holes with [`update`](@ref) as many times as
# there are settings.
# This tutorial builds a hospital pathway skeleton, fills it for two pathogens,
# adds a parallel branch and a literature-uncertain delay, and shows what
# `update` validates along the way.
# It builds on [Composing distributions](@ref composing-distributions) and
# [Competing outcomes](@ref competing-outcomes).

using ComposedDistributions
using Distributions
using Random

# ## The operator diagram
#
# A hospital pathway: onset, then an admission delay, then a death-or-discharge
# split.
# `→` (`\to`) chains events, and `|` marks the outcome split; the whole diagram
# is the topology, no distributions yet.

skeleton = @events begin
    onset → admission → (death | discharge)
end

# `@events` lowers this straight to an [`EventSkeleton`](@ref): a bare
# identifier becomes a named hole (the fill key), and `show` renders it back as
# the same operator diagram.

skeleton

# ## Filling the skeleton
#
# [`update`](@ref)`(skeleton; name = fill, ...)` substitutes every hole and
# builds the concrete tree through the ordinary composer verbs: the `→` chain
# becomes a [`Sequential`](@ref) and the `|` group a [`Resolve`](@ref) or
# [`Compete`](@ref), decided by the fill value type (see
# [Competing outcomes](@ref competing-outcomes)).
# Filling `death` with a `(dist, prob)` tuple and leaving `discharge` bare — the
# residual form used in [Composing distributions](@ref composing-distributions)
# — makes this a fixed-probability `Resolve`, the death probability the
# case-fatality ratio.

cfr = 0.12

pathogen_a = update(skeleton;
    onset = Gamma(2.0, 1.0),
    admission = LogNormal(0.5, 0.4),
    death = (Gamma(1.5, 1.0), cfr),
    discharge = Gamma(2.0, 1.5))

# The tree prints as the sequential-of-resolve it built; the nested one_of step
# is auto-named from its branches (`death_or_discharge`), so a fill names only
# the branch holes, never the group.

pathogen_a

# It is a distribution like any composed tree: `rand` simulates a record, and
# `logpdf` scores one straight back.

record = rand(Xoshiro(1), pathogen_a)

logpdf(pathogen_a, record)

# [`event`](@ref) reaches the resolve node by its auto-name to read the outcome
# split directly.

outcome = event(pathogen_a, :death_or_discharge)

probs(outcome)

# ## Reusing the skeleton across settings
#
# The point of separating topology from fill: the same `skeleton` builds a
# second pathogen's tree with no change to the diagram, only the numbers.

pathogen_b = update(skeleton;
    onset = Gamma(2.3, 0.9),
    admission = LogNormal(0.7, 0.3),
    death = (Gamma(1.7, 1.1), 0.28),
    discharge = Gamma(2.2, 1.3))

(pathogen_a = mean(pathogen_a), pathogen_b = mean(pathogen_b))

# Every fill still goes through the same validation: an unfilled hole is
# rejected and named in the error, so a copy-pasted fill missing a branch fails
# loudly rather than silently building a half-specified tree.

incomplete = try
    update(skeleton; onset = Gamma(2.0, 1.0), admission = LogNormal(0.5, 0.4))
    nothing
catch err
    err
end

sprint(showerror, incomplete)

# ## A racing-hazard fill
#
# The same `|` group becomes a [`Compete`](@ref) instead when every branch is
# filled with a bare distribution, no probabilities: the split then follows
# from which cause-specific delay fires first, not a fixed rate.

racing = update(skeleton;
    onset = Gamma(2.0, 1.0),
    admission = LogNormal(0.5, 0.4),
    death = Gamma(1.5, 1.0),
    discharge = Gamma(2.0, 1.5))

event(racing, :death_or_discharge)

# ## Parallel branches
#
# `&` runs branches off one shared origin instead of chaining them; a
# notification delay alongside the admission delay is one more hole, not a
# nested tree written by hand.

parallel_skeleton = @events begin
    onset → (admission & notification)
end

parallel_tree = update(parallel_skeleton;
    onset = Gamma(2.0, 1.0),
    admission = LogNormal(0.5, 0.4),
    notification = Gamma(1.0, 1.0))

parallel_tree

# The parallel group is auto-named the same way, joining its branch names with
# `_and_`, so `event` reaches into it by that name.

event(parallel_tree, :admission_and_notification, :notification)

# ## An uncertain fill
#
# A fill value is any valid leaf, not only a plain distribution: an
# [`@uncertain`](@ref) leaf carries literature uncertainty on a parameter, and
# it fills a hole exactly like a bare distribution.
# Here the onset delay's shape is uncertain rather than fixed at `2.0`.

uncertain_tree = update(skeleton;
    onset = (@uncertain Gamma(Normal(2.0, 0.3), 1.0)),
    admission = LogNormal(0.5, 0.4),
    death = (Gamma(1.5, 1.0), cfr),
    discharge = Gamma(2.0, 1.5))

has_uncertain(uncertain_tree)

# [`params_table`](@ref) carries the attached prior on its `prior` column, same
# as any other uncertain leaf; nothing about `@events` changes how the
# estimation surface reads it.

params_table(uncertain_tree)

# ## Summary
#
# - [`@events`](@ref) lowers a `→`/`|`/`&` operator diagram to an
#   [`EventSkeleton`](@ref): names and structure, no distributions.
# - [`update`](@ref)`(skeleton; name = fill, ...)` fills every hole and builds
#   the concrete tree through the ordinary composer verbs, validating that every
#   hole is filled and every fill key names a hole.
# - The `|` group becomes a [`Resolve`](@ref) or a [`Compete`](@ref) depending
#   on whether the fill values carry probabilities.
# - A fill value is any valid leaf — a plain distribution, an
#   [`@uncertain`](@ref) leaf, or a pre-built subtree — so one skeleton reuses
#   across pathogens, regions or scenarios by filling it differently each time.
# - A group's auto-name (`_or_` for one_of, `_and_` for parallel) is
#   structural; a fill names only the branch holes, never the group.
#
# ## Where next
#
# - [Composing distributions](@ref composing-distributions) is the full verb
#   walkthrough behind the tree `update` builds.
# - [Competing outcomes](@ref competing-outcomes) works through `resolve` versus
#   `compete` in depth.
# - [Concepts](@ref concepts) maps every verb, including `update`, to the layer
#   it belongs to.
