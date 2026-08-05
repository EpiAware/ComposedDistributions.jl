# Codec/params_table ordering consistency (#192, the #190 review follow-up).
#
# Three independent hand-maintained implementations of the same walk-order +
# dedup rule exist: `_walk_rows!` (params_table, a runtime walk that COLLECTS
# rows of DATA — edges, values, supports, priors — for a specific instance)
# and `_unflatten_expr`/`_flatten_reads!` (the generated codec, a compile-time
# walk that EMITS EXPRESSIONS over a TYPE). They agree today, but nothing
# catches divergence, and DistributionsInference.jl's generic `as_turing`
# (via its ComposedDistributions fit-protocol extension) zips `params_table`'s
# estimated rows against `flatten`'s flat vector index-for-index, assuming the
# orderings coincide.
#
# Full unification (rewriting both as one shared, pluggable walker — a
# "collect runtime rows" instantiation and a "generate an expression" one) is
# a genuine architectural change to the hottest, most heavily tested part of
# this package, at a moment when #202 was independently rewriting
# introspection.jl (merged since). This test is the fallback the issue offers
# instead: it builds every composer/leaf alpha the walk must handle and
# asserts, for each, that feeding params_table's own current values (in table
# order) through the codec (`reconstruct`) reproduces the SAME values — which
# can only hold if table order and codec order are the same bijection onto
# the tree's estimated parameters. A silent divergence (a reordering, a skip
# that disagrees between the two walks) breaks this immediately. Pool's
# ordering is deliberately not re-covered here: DistributionsInference.jl's
# `test/composed_ext.jl` non-centred pooled round-trip (ported from this
# package's own former `turing_ext.jl`, #221/#233) already exercises it end
# to end through real NUTS sampling and readback.
#
# `param_names` (introspection.jl) used to have a duplicated type-level
# mirror, `_param_names_of` (codec_gen.jl), that this file guarded against
# drifting apart from the instance-level original. Stage S3 of the
# codec-generation plan removed the whole type-level leaf-protocol-companion
# table `_param_names_of` belonged to (`_leaf_free_type`, `_extra_names_of`,
# `_params_arity_of`, the load-order-independent registry, and their
# `_leaf_type_param_names` combiner): the generated codec's leaf case now
# calls the SAME instance-level hooks (`param_names`, `leaf_param_names`,
# `leaf_param_values`) at runtime, from the returned generated code rather
# than the generator (see `_leaf_entry`/`_leaf_flatten_values`,
# introspection.jl, and the comments on `_leaf_unflatten_expr`/
# `_leaf_flatten_reads!`, codec_gen.jl). With only one name table left, there
# is nothing left for this file to compare it against.

@testsnippet CodecConsistencyHelpers begin
    using Distributions
    using ComposedDistributions: flat_dimension, reconstruct

    # The shared check: table order and codec order name the SAME parameters.
    # Feeding params_table's own current values (already domain-valid, since
    # they came from a live instance) through `reconstruct`, in table order,
    # must reproduce every row's value exactly — fixed rows untouched,
    # estimated rows round-tripped. A table/codec order mismatch reassigns at
    # least one parameter's value to the wrong leaf, which a tree built with
    # distinct per-leaf values always surfaces as a changed row somewhere.
    function _assert_codec_matches_table(d)
        table = params_table(d)
        est = findall(!isnothing, table.prior)
        n = flat_dimension(d)
        n == length(est) || return (:length_mismatch, n, length(est))
        n == 0 && return :ok
        x = Float64.(collect(table.value[est]))
        rebuilt = reconstruct(d, x)
        table2 = params_table(rebuilt)
        all(isapprox.(collect(table2.value), collect(table.value); atol = 1e-10)) ||
            return (:value_mismatch, table.edge, table.value, table2.value)
        return :ok
    end
end

@testitem "codec/params_table order: Sequential and Parallel" setup=[
    CodecConsistencyHelpers] begin
    seq = sequential(uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(seq) == :ok

    par = parallel(uncertain(Gamma(3.0, 1.5); theta = LogNormal(0.0, 0.2)),
        LogNormal(0.7, 0.3))
    @test _assert_codec_matches_table(par) == :ok
end

@testitem "codec/params_table order: Resolve (fixed probs)" setup=[
    CodecConsistencyHelpers] begin
    # Dirichlet-uncertain branch_probs is deliberately NOT covered by this
    # generic check: reconstruct collapses it from K-1 estimated stick rows to
    # K fixed, no-prior outcome-probability rows, a genuine structural change
    # in what the table's rows MEAN (not just their values), which the
    # same-row-count comparison this helper relies on cannot express. That
    # ordering is already covered end to end, through real NUTS sampling and
    # readback, by DistributionsInference.jl's "as_turing round-trip:
    # Dirichlet branch_probs stick coordinate" (test/composed_ext.jl).
    fixed = resolve(:a => (uncertain(Gamma(2.0, 1.0);
                alpha = LogNormal(0.0, 0.3)), 0.4),
        :b => (uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)), 0.6))
    @test _assert_codec_matches_table(fixed) == :ok

    # A NoEvent branch is skipped by both walks alike.
    withno = resolve(
        :event => (uncertain(Gamma(2.0, 1.0);
                alpha = LogNormal(0.0, 0.3)), 0.4),
        :none => (NoEvent(), 0.6))
    @test _assert_codec_matches_table(withno) == :ok
end

@testitem "codec/params_table order: Compete and Choose" setup=[
    CodecConsistencyHelpers] begin
    cmp = compete(:a => uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        :b => uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(cmp) == :ok

    ch = choose(:fast => uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        :slow => uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(ch) == :ok
end

@testitem "codec/params_table order: Shared tie across branches" setup=[
    CodecConsistencyHelpers] begin
    tied = shared(:g, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)))
    tree = compose((a = tied, b = tied,
        c = uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2))))
    @test _assert_codec_matches_table(tree) == :ok
end

@testitem "codec/params_table order: Convolved/Difference leaves" setup=[
    CodecConsistencyHelpers] begin
    using ConvolvedDistributions: ConvolvedDistributions, convolved, convolve_series,
                                  Difference, difference, product, Product,
                                  Convolved, AnalyticalSolver, NumericSolver,
                                  GaussLegendre, integrate, gl_integrate,
                                  AbstractSolverMethod

    conv_leaf = convolved(uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        Gamma(1.0, 1.0))
    seq = sequential(:total => conv_leaf,
        :report => uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(seq) == :ok

    diff_leaf = difference(uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)),
        Normal(1.0, 0.5))
    seq2 = sequential(:total => diff_leaf,
        :report => uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(seq2) == :ok
end

@testitem "codec/params_table order: Truncated and Censored wrapper leaves" setup=[
    CodecConsistencyHelpers] begin
    # Wrapper peeling (free_leaf/rewrap_leaf) is a distinct code path both
    # walks must agree on independently of plain leaves and composers — the
    # exact kind of alpha this file exists to guard, and one this file didn't
    # cover until now despite Truncated pre-dating this PR and Censored
    # landing alongside it (#215).
    trunc_leaf = truncated(
        uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)); upper = 10.0)
    cens_leaf = censored(
        uncertain(Gamma(3.0, 1.5); theta = LogNormal(0.0, 0.2)); upper = 10.0)
    tree = sequential(:trunc => trunc_leaf, :cens => cens_leaf,
        :plain => uncertain(LogNormal(0.5, 0.4); mu = Normal(0.5, 0.2)))
    @test _assert_codec_matches_table(tree) == :ok
end

@testitem "codec/params_table order: deeply nested mixed tree" setup=[
    CodecConsistencyHelpers] begin
    # A fixed-probs Resolve, not a Dirichlet-uncertain one, for the reason
    # given in the "Resolve (fixed probs)" testitem above.
    tied = shared(:g, uncertain(Gamma(2.0, 1.0); alpha = LogNormal(0.0, 0.3)))
    nested = compose((
        chain = sequential(uncertain(LogNormal(0.5, 0.4);
                mu = Normal(0.5, 0.2)),
            compete(:a => tied, :b => Exponential(1.0))),
        branch = resolve(
            :x => (uncertain(LogNormal(0.3, 0.2);
                    mu = Normal(0.3, 0.1)), 0.5),
            :y => (Gamma(1.5, 1.0), 0.5)),
        also_tied = tied))
    @test _assert_codec_matches_table(nested) == :ok
end
