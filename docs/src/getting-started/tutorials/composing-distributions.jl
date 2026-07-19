# # [Composing distributions](@id composing-distributions)
#
# ## Introduction
#
# ComposedDistributions.jl composes per-event delay distributions into one
# object that describes a whole record: named events linked by delays, wired
# into a tree.
#
# ### What we do here
#
# Each section is a small runnable example rather than a full analysis.
# We:
#
# 1. Compose a record from per-event delays with [`compose`](@ref), starting
#    from plain [Distributions.jl](https://juliastats.org/Distributions.jl)
#    leaves.
# 2. Build the five composers directly ([`Sequential`](@ref), [`Parallel`](@ref),
#    [`Resolve`](@ref), [`Compete`](@ref), [`Choose`](@ref)) and see how they
#    nest.
# 3. Score and simulate from one composed object.
# 4. Attach parameters and priors with [`params_table`](@ref) and
#    [`build_priors`](@ref).
# 5. Edit an assembled tree with [`update`](@ref), [`prune`](@ref) and
#    [`splice`](@ref).
#
# ### The operator map
#
# Every operator here falls into one of three families: structural composers
# wire branches into a tree, combinators add or difference whole delays, and
# the introspection verbs read or edit an assembled object.
# See the [Concepts](@ref concepts) verb map for the full list, grouped the
# same way.

# ## Packages used
#
# We use Distributions for the delay distributions and Random for reproducibility.

using ComposedDistributions
using ConvolvedDistributions
using Distributions
using Random

# ## Composing a record
#
# A record is a set of events linked by delays.
# [`compose`](@ref) is the front-end: it takes a friendly description and lowers
# it to a nested stack of the composers, without introducing a new tree type.
# A NamedTuple of bare distributions names each branch off one shared origin (a
# [`Parallel`](@ref)); a `Vector` value is a chain of steps (a
# [`Sequential`](@ref)).

onset_admit = LogNormal(1.5, 0.4);

admit_death = Gamma(2.0, 1.0);

# Two branches off one onset: an onset-to-admission delay and an
# onset-to-notification delay.
parallel_stack = compose((onset_admit = onset_admit,
    onset_notif = Gamma(1.5, 1.0)));

event_names(parallel_stack)

# A composed stack is a distribution like any leaf.
# It simulates a named event record with `rand`,

rand(Xoshiro(1), parallel_stack)

# and scores one with `logpdf`.
# The whole toolkit below is built on these two operations.

logpdf(parallel_stack, rand(Xoshiro(1), parallel_stack))

# A chain step is a `Vector`: onset to admission, then admission to death.
chain = compose((path = [onset_admit, admit_death],));

event_names(chain)

# The same stack also lowers from an explicit Tables.jl table, so a
# column-oriented data source builds a composer without a hand-written
# NamedTuple.
# A table has `name` and `dist` columns, one row per branch.
# A `chain` column folds rows sharing a non-zero id into a [`Sequential`](@ref),
# and a `compete`/`prob` column pair folds rows into a [`Resolve`](@ref) node
# whose `prob` entries are the branch probabilities.

table = [
    (name = :death, dist = Gamma(1.5, 1.0), compete = 1, prob = 0.3),
    (name = :discharge, dist = Gamma(2.0, 1.5), compete = 1, prob = 0.7),
    (name = :onset_notif, dist = Gamma(1.5, 1.0), compete = 0,
        prob = missing)];

table_stack = compose(table);

event_names(table_stack)

# ## The five composers
#
# Each front-end lowers to these composers, which can also be built directly.
# They differ in how the branches relate.
#
# [`Sequential`](@ref) is a conjunctive chain: each step adds an independent
# delay onto the previous event.
# The lowercase [`sequential`](@ref) verb is the public constructor; name the
# steps with `name => dist` pairs.

seq = sequential(:onset_admit => onset_admit, :admit_death => admit_death);

# [`Parallel`](@ref) places independent branches off one shared origin, built
# with the [`parallel`](@ref) verb.

par = parallel(:onset_admit => onset_admit, :onset_notif => admit_death);

# [`Resolve`](@ref) is a disjunction where exactly one outcome occurs, governed
# by fixed branch probabilities that sum to one.
# A death-versus-discharge split makes the death probability the case-fatality
# ratio.

cfr = 0.3;

resolution = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :discharge => (Gamma(2.0, 1.5), 1 - cfr));

# The last outcome's probability may be omitted (a bare `name => delay`).
# It then takes the residual `1 - sum(of the others)`, so the discharge
# probability `1 - cfr` need not be written out.

resolution_residual = resolve(:death => (Gamma(1.5, 1.0), cfr),
    :discharge => Gamma(2.0, 1.5));

# Its marginal is the time to resolution regardless of which outcome occurs.

mean(resolution)

# A `Resolve` node carries its own time-to-resolution event slot alongside the
# named per-outcome slots, so its flat event layout pairs that resolution slot
# (defaulting to `:event_1`) with the outcome names.

event_names(resolution)

# [`Compete`](@ref) is the racing-hazard sibling of `Resolve`, built with the
# [`compete`](@ref) verb from bare `name => delay` outcomes (no probabilities).
# The cause-specific delays race, the first to fire wins, and which outcome wins
# is coupled to when it fires.
# Where `resolve` takes the winning probability as a fixed parameter, `compete`
# derives it from the hazards: the marginal any-event time is `min` of the racing
# delays, with survival the product of the per-cause survivals.

racing = compete(:death => Gamma(1.5, 1.0), :discharge => Gamma(2.0, 1.5));

# [Competing outcomes](@ref competing-outcomes) works through `resolve` versus
# `compete` in full, including when to reach for each.

# [`Choose`](@ref) is a data-selected disjunction: the alternatives are
# independent sub-models with different origins, and a data field picks which one
# applies to a record.
# Neither `Parallel` nor `Resolve` (both shared origin) expresses this.

selector = choose(:index => onset_admit, :sourced => admit_death);

# Scoring names the active alternative through the `kind` keyword.

logpdf(selector, 3.0; kind = :index)

# ## Nesting
#
# The composers nest, so trees of arbitrary depth are built by composing on
# composers.
# A `compose` result drops into another `compose` as a branch.

early = compose((onset_admit = onset_admit, onset_notif = admit_death));

nested = compose((early = early, late = chain));

event_names(nested)

# A pre-built composer is a valid `Sequential` step, so a chain can carry a
# `Resolve` resolution as its terminal step.
# Naming the chain steps gives the simulated record readable event names.

tree = compose((
    path = sequential(:onset_admit => onset_admit,
        :admit_resolve => resolution),
    onset_notif = admit_death));

# The flat event layout of a tree is derived from the edge names.

event_names(tree)

# ## Scoring and simulation from one object
#
# The composer is dual-purpose: it scores observed records and simulates new
# ones.
# A `rand` of a tree returns a full named event record, and `logpdf` reads one
# straight back.

record = rand(Xoshiro(7), tree)

# The labelled draw round-trips through `logpdf`, either as the record or as its
# flat vector of values.

(from_record = logpdf(tree, record),
    from_vector = logpdf(tree, collect(values(record))))

# A `Resolve` node scores and samples its marginal time to resolution directly.

logpdf(resolution, 3.0)

# [`rand`](@ref)`(node; outcome = true)` draws which outcome occurs and its time
# as a compact `(outcome, time)` pair, so a standalone draw tells you which
# outcome won.

rand(Xoshiro(7), resolution; outcome = true)

# ## Combining whole delays
#
# Where the composers wire named branches into a tree, the combinators join two
# whole delays algebraically.
# `convolved` forms the sum `X + Y` (the total of two independent delays),
# and `difference` forms the dual `X - Y`. Both are ConvolvedDistributions'
# own verbs, extended here for composed tree operands (a bare distribution
# works unchanged too).

total = convolved(Gamma(2.0, 1.0), LogNormal(0.5, 0.4));

(convolved_mean = mean(total),
    summed_mean = mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4)))

# ## Reading the composed marginal
#
# A composed chain has a marginal delay from its origin to its final event.
# The moments are additive over the steps, so `mean` and `var` of a
# [`Sequential`](@ref) sum the per-step moments.

chain_moments = sequential(:onset_admit => Gamma(2.0, 1.0),
    :admit_death => LogNormal(0.5, 0.4));

mean(chain_moments), var(chain_moments)

# [`observed_distribution`](@ref) collapses that chain to the single convolved
# distribution of its origin-to-final gap, integrating the intermediate event
# out.
# Its mean matches the chain's overall mean.

collapsed = observed_distribution(chain_moments);

(collapsed_mean = mean(collapsed), chain_mean = mean(chain_moments))

# ## Parameters and priors
#
# A composed distribution carries a flat inventory of its free parameters.
# [`params_table`](@ref) lists one row per scalar parameter, keyed by the edge
# path and the parameter name, with the support a prior must respect.
# It prints as a table and is a Tables.jl source, so `tbl.edge` / `tbl.param`
# read its columns.

template = compose((onset_admit = Gamma(2.0, 1.0),
    admit_death = LogNormal(0.5, 0.4)));

tbl = params_table(template)

# Its columns are accessed by name.

tbl.edge, tbl.param

# [`build_priors`](@ref) takes that table (any Tables.jl source with `edge`,
# `param`, `value`, `support` columns) and derives a default prior per row from
# that leaf's support: a positive scale parameter gets a positive-truncated
# prior, a location parameter an unbounded one, a `[0, 1]` probability a
# `Uniform(0, 1)`.
# So `build_priors(tbl)` alone yields a complete set, defined against the table
# rather than by hand-matching the tree.

priors = build_priors(tbl);

priors.onset_admit.shape

# ## Editing a composed tree
#
# [`update`](@ref) applies a set of parameter values back to a composed object,
# returning a distribution of the same structure.

updated = update(template, (onset_admit = (shape = 3.0, scale = 1.5),
    admit_death = (mu = 0.7, sigma = 0.5)));

mean(updated)

# [`update`](@ref) also replaces whole nodes, not just their values.
# Passing `path => new_node` swaps the node at an address for a new distribution,
# keeping the tree shape.
# The address is the same one [`event`](@ref) reads: a bare name, a dotted
# `Symbol`, or a tuple of edge names.

replaced = update(template, :admit_death => Gamma(3.0, 1.5));

event(replaced, :admit_death)

# Two edits that change the tree shape are kept separate.
# [`prune`](@ref) drops a branch from a node (renormalising a [`Resolve`](@ref)
# arm's remaining probabilities), and [`splice`](@ref) inserts a step around a
# node.

three_way = resolve(:death => (Gamma(1.5, 1.0), 0.3),
    :discharge => (Gamma(2.0, 1.5), 0.4),
    :transfer => (Gamma(1.0, 1.0), 0.3));

resolution_tree = compose((resolution = three_way, onset = Gamma(1.0, 1.0)));

pruned = prune(resolution_tree, :resolution, :transfer);

# The `:transfer` outcome is gone from the pruned node's event layout.

event_names(event(pruned, :resolution))

# `splice` wraps a node in a chain, here adding a reporting delay after the
# death branch.

spliced = splice(template, :admit_death;
    after = :death_report => Gamma(1.0, 2.0));

event_names(event(spliced, :admit_death))

# [`tie`](@ref) links leaves at several paths into one free parameter group, so
# [`params_table`](@ref) inventories the tied occurrences once under the shared
# tag rather than as separate parameters.

shared_rate = compose((a = Gamma(2.0, 1.0), b = Gamma(2.0, 1.0)));

tied = tie(shared_rate, :a, :b; name = :rate);

unique(params_table(tied).edge)

# ## Syntax reference
#
# Every public composition form on one object, with whether it preserves the
# tree shape.
#
# | Syntax | What it does | Shape |
# |---|---|---|
# | `compose((a = d1, b = d2))` | NamedTuple front-end; a `Vector` value is a chain | builds |
# | `compose(table)` | Tables.jl `name`/`dist` source; a `chain` column folds rows into a `Sequential`, a `compete`/`prob` pair into a `Resolve` | builds |
# | `sequential(:a => d1, :b => d2)` | a [`Sequential`](@ref) chain (steps add up) | builds |
# | `parallel(:a => d1, :b => d2)` | a [`Parallel`](@ref) branch set (shared origin) | builds |
# | `resolve(:a => (d1, p1), :b => (d2, p2))` | a [`Resolve`](@ref) node; the last prob may be omitted as the residual | builds |
# | `compete(:a => d1, :b => d2)` | a [`Compete`](@ref) racing-hazard node (the winning probability derived from the hazards) | builds |
# | `choose(:a => d1, :b => d2)` | a [`Choose`](@ref) disjunction (data picks the branch) | builds |
# | `convolved(d1, d2)` | a `Convolved` sum `X + Y` | builds |
# | `difference(d1, d2)` | a `Difference` `X - Y` | builds |
# | `shared(:tag, d)` | tag a leaf as a tied parameter group | leaf wrap |
# | `tie(d, paths...; name)` | tie leaves at `paths` into one group | yes |
# | `update(d, (a = (shape = 3,),))` | replace free parameter values | yes |
# | `update(d, path => new_node)` | replace a whole node | yes |
# | `prune(d, path...)` | drop a branch (renormalise a `Resolve` arm) | no (topology) |
# | `splice(d, path; before, after)` | insert a step at a node | no (topology) |
# | `event(d, path...)` | fetch a child or descend a name path | read |
# | `event_tree(d)` | the nested tree of event names | read |
# | `event_names(d)` | the per-record key names | read |
# | `observed_distribution(d)` | collapse a chain to its convolved total | read |
# | `params_table(d)` | the flat free-parameter inventory | read |
#
# The address `path` in `event` / `update` / `prune` / `splice` / `tie` is the
# same in all: a bare `Symbol`, a dotted `Symbol` (`:a.b`), or a tuple of edge
# names.
# `shared(:tag, d)` and `tie(d, paths...; name = :tag)` are two spellings of the
# same tie: `shared` tags a leaf where it is built, `tie` walks the tree to the
# named leaves and wraps each in the same tie.
# Both make the tagged occurrences one free parameter.

# ## Summary
#
# - [`compose`](@ref) lowers a NamedTuple, table, or matrix to the same composer
#   stack.
# - [`Sequential`](@ref), [`Parallel`](@ref), [`Resolve`](@ref), [`Compete`](@ref)
#   and [`Choose`](@ref) are conjunctive chains, shared-origin branches,
#   fixed-probability disjunctions, racing-hazard disjunctions and data-selected
#   disjunctions.
# - The composers nest, including a composer as a chain step.
# - One object scores records and simulates them: `logpdf` reads a record,
#   `rand` generates one.
# - `convolved` and `difference` combine two whole
#   delays algebraically.
# - `mean` and `var` read the composed marginal moments, and
#   [`observed_distribution`](@ref) collapses a chain to its convolved total.
# - [`params_table`](@ref) and [`build_priors`](@ref) attach parameters and
#   support-derived priors to the same object.
# - [`update`](@ref) edits the tree: `path => new_node` replaces nodes keeping
#   the shape, [`prune`](@ref) and [`splice`](@ref) are the two topology edits,
#   and [`tie`](@ref) links leaves into one parameter group.
#
# ## Where next
#
# - The [Public API](@ref public-api) lists every composer, combinator and
#   introspection verb with its full docstring.
