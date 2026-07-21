# Distribution-level elapsed-distance accessors over a named chain.
#
# `elapsed_between` returns the LAW of the elapsed distance between two named
# events of a `Sequential` chain as an ordinary univariate distribution (the
# convolution of the intervening steps), the distribution-level counterpart to
# the sample-level difference of two event positions. It is NOT `difference`:
# two events on one chain descend from a SHARED origin, so their absolute
# positions are not independent and a plain `difference` of the two totals would
# double-count the shared leading steps and give the wrong law (#274). Instead
# the elapsed distance is exactly the convolution of the steps strictly between
# the two events.
#
# This reads the public `convolved` verb and the chain's `event_names`/observed
# leaves; it does not touch the convolution-layer internals. The event ->
# leaf-prefix mapping mirrors the cumulative-prefix logic in
# `convolved_interop.jl` (`_event_prefix_delay`) rather than calling it, so the
# elapsed accessor does not couple to that file; if the two later diverge, that
# is convolution-layer migration cleanup.

@doc "

Distribution of the elapsed distance between two named events of a chain.

`elapsed_between(chain, from, to)` returns the LAW of the elapsed distance from
event `from` to event `to` on a [`Sequential`](@ref) chain, as an ordinary
univariate distribution: the convolution of the steps strictly between the two
events. `elapsed_between(chain, to)` is the common origin-to-`to` form (the
convolution of the chain prefix up to `to`).

This is the distribution-level counterpart to the sample-level difference of two
event positions, and is deliberately NOT [`difference`](@ref): two events on one
chain descend from a shared origin, so their absolute positions are not
independent and `difference(chain, chain)` would double-count the shared leading
steps. `elapsed_between` convolves only the intervening steps, so it is the
correct same-chain law. The result discretises and composes like any other
univariate component, and recomputes when an upstream step's parameters change.

Only a plain continuous chain (every step a delay leaf or a nested plain chain,
no branching) has a single elapsed-distance law between two of its events; a
chain with a branching step, or a pair whose ordering has no single scalar law,
is rejected with a clear error. An uncertain-leaf chain must be pinned with
[`update`](@ref) first, as the template density is not the marginal.

# Arguments
- `chain`: a [`Sequential`](@ref) chain.
- `from`: the earlier event name (origin-to-`to` form omits it).
- `to`: the later event name.

# Examples
```@example
using ComposedDistributions, Distributions

chain = sequential(:origin_onset => Gamma(2.0, 1.0),
    :onset_admit => LogNormal(0.5, 0.4),
    :admit_exit => Gamma(1.5, 1.0))

# Between two named events: the single intervening step.
law = elapsed_between(chain, :onset, :admit)
rand(law)

# Origin to a named intermediate event: the convolution of the prefix.
law0 = elapsed_between(chain, :admit)
mean(law0)
```

# See also
- [`convolved`](@ref): the chain-step convolution this returns.
- [`event_names`](@ref): the chain's event names, the valid selectors.
- [`difference`](@ref): the difference of two INDEPENDENT observed totals.
"
function elapsed_between end

# origin -> `to`.
function elapsed_between(chain::Sequential, to::Symbol)
    return _elapsed_between(chain, nothing, to)
end

# `from` -> `to`.
function elapsed_between(chain::Sequential, from::Symbol, to::Symbol)
    return _elapsed_between(chain, from, to)
end

# A non-chain operand has no per-event elapsed layout; name the way forward.
function elapsed_between(d::Parallel, args::Symbol...)
    throw(ArgumentError(
        "elapsed_between needs a Sequential chain; a Parallel has several " *
        "independent endpoints and no single elapsed-distance law between two " *
        "events. Take the branch chain first, e.g. " *
        "`elapsed_between(event(d, name), from, to)`"))
end

function elapsed_between(d::Choose, args::Symbol...)
    throw(ArgumentError(
        "elapsed_between needs a Sequential chain; a Choose's active " *
        "alternative is data-selected, so two events on different " *
        "alternatives have no single scalar elapsed law. Take the chosen " *
        "alternative first, e.g. `elapsed_between(event(d, :index), from, to)`"))
end

function elapsed_between(d, ::Symbol, args::Symbol...)
    throw(ArgumentError(
        "elapsed_between needs a Sequential chain of named steps; got a " *
        "$(nameof(typeof(d)))"))
end

function _elapsed_between(chain::Sequential, from, to::Symbol)
    has_uncertain(chain) && throw(ArgumentError(
        "cannot read an elapsed-distance law on a chain with uncertain " *
        "leaves; pin the parameters with `update(tree, params)` first"))
    leaves = _observed_leaves(chain.components)
    enames = event_names(chain)
    # The flat events are `(origin, target_1, ..., target_k)`, one target per
    # observed delay leaf, so a plain chain has exactly one more event than
    # leaves; a branching step breaks that one-to-one line-up.
    length(enames) == length(leaves) + 1 || throw(ArgumentError(
        "elapsed_between needs a plain continuous chain (every step a delay " *
        "leaf, no branching); this chain's events $(collect(enames)) do not " *
        "line up one-to-one with its delay steps"))
    to_idx = _elapsed_event_index(enames, to)
    from_idx = from === nothing ? 1 : _elapsed_event_index(enames, from)
    from_idx < to_idx || throw(ArgumentError(
        "elapsed_between: `from` ($(repr(from === nothing ? enames[1] : from)))" *
        " must come strictly before `to` ($(repr(to))) in the chain; the " *
        "events in order are $(collect(enames))"))
    slice = leaves[from_idx:(to_idx - 1)]
    return length(slice) == 1 ? only(slice) : convolved(slice)
end

# The flat position of a named event, erroring clearly (with the valid events)
# when the name is not one of the chain's events.
function _elapsed_event_index(enames::Tuple, name::Symbol)
    idx = findfirst(==(name), enames)
    idx === nothing && throw(ArgumentError(
        "elapsed_between: $(repr(name)) is not an event of this chain; the " *
        "events in order are $(collect(enames))"))
    return idx
end
