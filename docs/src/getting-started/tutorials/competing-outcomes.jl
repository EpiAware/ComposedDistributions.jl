# # [Competing outcomes: resolve versus compete](@id competing-outcomes)
#
# ## Introduction
#
# A natural history often ends in one of several mutually exclusive outcomes: a
# case recovers or dies, an infection is detected or missed.
# ComposedDistributions expresses this as a one_of node and offers two flavours
# that differ in where the outcome split comes from.
# [`resolve`](@ref) sets the split as fixed probabilities; [`compete`](@ref)
# derives it from racing hazards, so which outcome occurs is coupled to when.
#
# This tutorial builds both over the same two outcomes and shows when to reach
# for each.
# It builds on [Composing distributions](@ref composing-distributions).
#
# | Modelling concept | Composed primitive |
# |---|---|
# | one outcome by a fixed probability | a [`resolve`](@ref) node |
# | rival risks, which-and-when hazard-driven | a [`compete`](@ref) node |
# | an event that only sometimes occurs | a no-event [`resolve`](@ref) branch |
# | an outcome that continues into a further chain | a resolve outcome holding a subtree |

using ComposedDistributions
using Distributions
using Random
import ComposedDistributions: rand_outcome

# ## Fixed-probability resolution
#
# [`resolve`](@ref) sets the outcome split directly.
# A death-versus-recovery split makes the death probability the case-fatality
# ratio; the last outcome's probability may be omitted as the residual.

cfr = 0.3

outcome = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :recover => Gamma(2.0, 1.5))

# Its marginal is the time to resolution, whichever outcome occurs.

mean(outcome)

# A bare [`rand`](@ref) draws the full named event record of the outcome that
# fired (the fired outcome's time present, the rest `missing`), which feeds
# straight back into `logpdf`.

rand(Xoshiro(1), outcome)

# [`rand_outcome`](@ref) is the compact `(outcome, time)` pair view of the same
# draw, so a standalone draw tells you which outcome won.

rand_outcome(Xoshiro(1), outcome)

# Over many draws the outcome frequencies match the fixed split.

rng = Xoshiro(42)
draws = [first(rand_outcome(rng, outcome)) for _ in 1:5000]
count(==(:death), draws) / length(draws)     # ≈ cfr

# ## An outcome that only sometimes occurs
#
# A [`NoEvent`](@ref) branch carries the mass of cases that never resolve, so its
# probability is the residual and a draw of that branch has no event time.

with_survivors = resolve(:death => (Gamma(1.5, 1.0), 0.2),
    :recover => (Gamma(2.0, 1.5), 0.5), :survive => NoEvent())

# A no-event draw returns a `missing` time.

rand_outcome(Xoshiro(4), with_survivors)

# ## Racing hazards
#
# [`compete`](@ref) takes bare outcomes with no probabilities: the cause-specific
# delays race, the first to fire wins, and the split is derived from the hazards.

racing = compete(:death => Gamma(1.5, 1.0), :recover => Gamma(2.0, 1.5))

# The marginal any-event time is the `min` of the racing delays, so its survival
# is the product of the per-cause survivals.

t = 3.0
(racing_ccdf = ccdf(racing, t),
    product_ccdf = ccdf(Gamma(1.5, 1.0), t) * ccdf(Gamma(2.0, 1.5), t))

# Because the split follows from the delays, the death frequency is a consequence
# of the hazards rather than a set parameter.

race_rng = Xoshiro(2024)
races = [first(rand_outcome(race_rng, racing))
         for _ in 1:5000]
count(==(:death), races) / length(races)

# ## Which to use
#
# Reach for `resolve` when the outcome split is a fixed probability independent
# of timing, such as a known case-fatality ratio.
# Reach for `compete` when rival risks act on a shared clock, so the split
# follows from the delays.

# ## Nesting a resolution in a natural history
#
# A one_of node is a valid step in a larger tree, so a chain can end in a
# resolution: onset, then admission, then a death-versus-recovery outcome.

history = compose((
    path = sequential(:onset_admit => LogNormal(1.5, 0.4),
    :admit_resolve => outcome),))

event_names(history)

# A draw fills the origin, the admission, and the resolution.

rand(Xoshiro(1), history)

# ## Summary
#
# - [`resolve`](@ref) sets the outcome split as fixed probabilities; a
#   [`NoEvent`](@ref) branch carries cases that never resolve.
# - [`compete`](@ref) derives the split from racing hazards, coupling which
#   outcome occurs to when.
# - [`rand_outcome`](@ref) reads the sampled outcome and time; the marginal
#   `logpdf` and `mean` treat the node as one time-to-resolution distribution.
# - A one_of node nests as a step in a larger composed tree.
#
# ## Where next
#
# - [Composing distributions](@ref composing-distributions) covers all five
#   composers and the structural edits.
# - [Delay chains and the linear chain trick](@ref linear-chain) shows the
#   conjunctive `Sequential` chain in depth.
