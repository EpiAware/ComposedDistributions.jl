# # [Delay chains and the linear chain trick](@id linear-chain)
#
# ## Introduction
#
# A multi-step delay is a chain: each step adds an independent delay onto the
# previous event.
# [`Sequential`](@ref) composes the steps, and the chain is one distribution over
# the whole origin-to-final gap.
# When every step is an `Exponential` with the same rate, the total is an Erlang
# (a `Gamma` with integer shape).
# This is the linear chain trick, the identity behind representing a
# Gamma-distributed delay as a series of exponential compartments.
#
# This tutorial builds a chain, reads its structure and moments, and collapses it
# to its total.
# It builds on [Composing distributions](@ref composing-distributions).

using ComposedDistributions
using Distributions

# ## A chain of exponential steps
#
# We build a four-step chain, each step an `Exponential` with mean `theta`.
# `sequential` names the steps; a bare `Vector` passed to `compose` is the same
# chain with default step names.

theta = 1.5

steps = [Symbol("step_", i) => Exponential(theta) for i in 1:4]

chain = sequential(steps...)

# The flat event layout is the origin plus one event per step.

event_names(chain)

# ## Additive moments
#
# The overall moments of a chain sum over the steps, so a four-step exponential
# chain has mean `4 * theta` and variance `4 * theta^2`.

mean(chain), var(chain)

# ## Collapsing to the total
#
# [`observed_distribution`](@ref) collapses the chain to the single distribution
# of its origin-to-final gap, integrating the intermediate events out.
# For the exponential chain this total is an Erlang, so its moments match
# `Gamma(4, theta)`.

total = observed_distribution(chain)

mean(total) ≈ mean(Gamma(4, theta))

# The variance matches too.

var(total) ≈ var(Gamma(4, theta))

# The linear chain trick is exactly this identity: a chain of `k` exponential
# steps of mean `theta` is a `Gamma(k, theta)` delay, so a smooth delay can be
# represented as a series of memoryless compartments and back again.

# ## Reusing the chain
#
# A chain is a composer, so it drops into a larger tree as a branch and the same
# steps can be reused across models.

tree = compose((incubation = chain, reporting = Exponential(2.0)))

event_names(tree)

# ## Summary
#
# - A [`Sequential`](@ref) chain composes per-step delays into one distribution
#   over the whole gap.
# - The overall `mean` and `var` are additive over the steps.
# - [`observed_distribution`](@ref) collapses the chain to its convolved total.
# - A chain of `k` equal exponential steps is a `Gamma(k, theta)`, the linear
#   chain trick, and the chain nests inside a larger tree.
#
# ## Where next
#
# - [Competing outcomes](@ref competing-outcomes) shows the disjunctive one_of
#   nodes.
# - [Composing distributions](@ref composing-distributions) is the full
#   walkthrough of every verb.
