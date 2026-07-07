# Uncertain-distribution tests: construction, template delegation, marginal
# sampling, collapse via `update`, truncation push-inside, composition, and the
# prior/params integration.

@testitem "uncertain: three constructor forms and validation" begin
    using Distributions

    # Keyword form on a concrete template (partial: one param uncertain).
    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    @test u isa Uncertain
    @test u.template == Gamma(2.0, 1.0)
    @test u.specs == (; shape = LogNormal(log(2.0), 0.2))

    # A Real keyword re-pins the template's fixed value.
    u2 = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2),
        scale = 2.5)
    @test u2.template == Gamma(2.0, 2.5)

    # Positional family form: both parameters uncertain (the family's default
    # instance is a valid placeholder; the specs drive the draws).
    pf = uncertain(Gamma, LogNormal(log(2.0), 0.2), Exponential(1.0))
    @test pf isa Uncertain
    @test keys(pf.specs) == (:shape, :scale)
    @test pf.template == Gamma()

    # Positional family form: shape uncertain, scale fixed at 1.0.
    pf2 = uncertain(Gamma, LogNormal(log(2.0), 0.2), 1.0)
    @test keys(pf2.specs) == (:shape,)
    @test params(pf2.template)[2] == 1.0

    # Type keyword form: every parameter given explicitly.
    tf = uncertain(Gamma; shape = LogNormal(log(2.0), 0.2), scale = 1.0)
    @test tf isa Uncertain
    @test tf.specs == u.specs
    @test params(tf.template)[2] == 1.0

    # Validation.
    @test_throws ArgumentError uncertain(Gamma(2.0, 1.0);
        rate = LogNormal(0.0, 1.0))
    @test_throws ArgumentError uncertain(Gamma(2.0, 1.0); shape = "no")
    @test_throws ArgumentError uncertain(Gamma(2.0, 1.0))
    @test_throws ArgumentError uncertain(Gamma;
        shape = LogNormal(log(2.0), 0.2))
    # Positional form needs one argument per parameter.
    @test_throws ArgumentError uncertain(Gamma, LogNormal(log(2.0), 0.2))
    # Nesting goes in the specs, not the template.
    @test_throws ArgumentError Uncertain(u, (; shape = LogNormal(0.0, 1.0)))
end

@testitem "uncertain: equality, hash, show" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    v = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    w = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.5))
    @test u == v
    @test hash(u) == hash(v)
    @test u != w

    @test occursin("uncertain(", string(u))
    @test occursin("shape", string(u))
end

@testitem "uncertain: standard interface delegates to the template" begin
    using Distributions

    tmpl = Gamma(2.0, 1.0)
    u = uncertain(tmpl; shape = LogNormal(log(2.0), 0.2))

    @test params(u) == params(tmpl)
    @test minimum(u) == minimum(tmpl)
    @test maximum(u) == maximum(tmpl)
    @test insupport(u, 1.0)
    @test !insupport(u, -1.0)
    @test eltype(typeof(u)) == eltype(typeof(tmpl))

    # Scalar density/cdf/quantile are at the template's central values (NOT the
    # marginal), so they equal the template's exactly.
    @test logpdf(u, 1.5) == logpdf(tmpl, 1.5)
    @test pdf(u, 1.5) == pdf(tmpl, 1.5)
    @test cdf(u, 1.5) == cdf(tmpl, 1.5)
    @test logcdf(u, 1.5) == logcdf(tmpl, 1.5)
    @test ccdf(u, 1.5) == ccdf(tmpl, 1.5)
    @test logccdf(u, 1.5) == logccdf(tmpl, 1.5)
    @test quantile(u, 0.5) == quantile(tmpl, 0.5)

    # Template-value moments (uncertainty not propagated; documented).
    @test mean(u) == mean(tmpl)
    @test var(u) == var(tmpl)
    @test std(u) == std(tmpl)

    # has_uncertain is the loud guard for the silent delegation above.
    @test has_uncertain(u)
    @test !has_uncertain(tmpl)
end

@testitem "rand: marginal draw matches a hand-written two-stage draw" begin
    using Distributions, Random

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))

    # The marginal draw is: draw the spec, then draw the value from the concrete
    # leaf, on the SAME rng stream.
    rng = Xoshiro(42)
    shape = rand(rng, LogNormal(log(2.0), 0.2))
    expected = rand(rng, Gamma(shape, 1.0))
    @test rand(Xoshiro(42), u) == expected

    # A fresh parameter set each call (marginal iid) with the right mean.
    stream = Xoshiro(3)
    draws = [rand(stream, u) for _ in 1:20000]
    @test all(>=(0), draws)
    @test mean(draws) ≈ mean(LogNormal(log(2.0), 0.2)) * 1.0 atol = 0.05
    @test length(unique(draws)) == length(draws)
end

@testitem "rand: a wrapped template re-applies its fixed structure" begin
    using Distributions, Random

    tu = uncertain(truncated(Gamma(2.0, 1.0); upper = 10.0);
        shape = LogNormal(log(2.0), 0.2))
    draws = [rand(Xoshiro(i), tu) for i in 1:200]
    @test all(x -> 0.0 <= x <= 10.0, draws)
end

@testitem "rand: nested uncertainty draws the whole stack" begin
    using Distributions, Random

    nested = uncertain(Gamma(2.0, 1.0);
        shape = uncertain(LogNormal(log(2.0), 0.2);
            mu = Normal(log(2.0), 0.1)))
    @test rand(Xoshiro(2), nested) isa Real
    @test rand(Xoshiro(2), nested) == rand(Xoshiro(2), nested)
    @test all(>=(0), [rand(Xoshiro(i), nested) for i in 1:50])
end

@testitem "truncated pushes inside an uncertain leaf" begin
    using Distributions, Random

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    t = truncated(u; upper = 5.0)
    @test t isa Uncertain
    @test t.template isa Truncated
    @test t.specs == u.specs

    t2 = truncated(u, 1.0, 5.0)
    @test t2 isa Uncertain
    @test t2.template.lower == 1.0
    @test t2.template.upper == 5.0

    @test truncated(u) === u

    rng = Xoshiro(4)
    @test all(x -> 1.0 <= x <= 5.0, [rand(rng, t2) for _ in 1:200])
end

@testitem "update collapses an uncertain leaf to its concrete template" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    tree = compose((onset_admit = u, admit_death = LogNormal(0.5, 0.4)))

    tree2 = update(tree, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))
    @test event(tree2, :onset_admit) == Gamma(3.0, 1.5)
    @test !has_uncertain(tree2)
    # The collapsed tree scores with the ordinary surface.
    @test isfinite(logpdf(tree2, [1.5, 0.8]))

    # A wrapped uncertain leaf keeps its fixed structure through the update.
    tu = uncertain(truncated(Gamma(2.0, 1.0); upper = 10.0);
        shape = LogNormal(log(2.0), 0.2))
    ttree = compose((onset_admit = tu,))
    ttree2 = update(ttree, (onset_admit = (shape = 3.0, scale = 1.5),))
    leaf = event(ttree2, :onset_admit)
    @test leaf isa Truncated
    @test leaf.upper == 10.0
    @test ComposedDistributions.free_leaf(leaf) == Gamma(3.0, 1.5)
end

@testitem "update node-replace makes a leaf uncertain" begin
    using Distributions, Random

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))

    utree = update(tree, :onset_admit => u)
    @test event(utree, :onset_admit) === u
    @test has_uncertain(utree)
    # The tree still samples (the uncertain leaf draws its marginal).
    draw = rand(Xoshiro(1), utree)
    @test draw isa NamedTuple
    @test all(isfinite, values(draw))
end

@testitem "a shared-tied uncertain leaf samples through the tag" begin
    using Distributions, Random

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    su = shared(:inc, u)

    # The tag is transparent to sampling: the marginal draws through it once.
    @test rand(Xoshiro(1), su) == rand(Xoshiro(1), u)

    # The specs stay visible through the tag (routing + the prior column).
    @test ComposedDistributions._uncertain_specs(su) == u.specs
    @test has_uncertain(su)

    # It composes and samples as a leaf, and is inventoried once under its tag.
    p = parallel(:a => su, :b => LogNormal(0.5, 0.4))
    @test rand(Xoshiro(2), p) isa NamedTuple
    tbl = params_table(p)
    idx = findfirst(==(:shape), tbl.param)
    @test tbl.edge[idx] == :inc
    @test tbl.prior[idx] == LogNormal(log(2.0), 0.2)
end

@testitem "params_table prior column and build_priors precedence" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    tree = compose((onset_admit = u, admit_death = LogNormal(0.5, 0.4)))
    tbl = params_table(tree)
    @test :prior in propertynames(tbl)

    # The uncertain shape row carries its spec; fixed rows carry nothing.
    idx = findfirst(i -> tbl.edge[i] == :onset_admit &&
                         tbl.param[i] == :shape,
        eachindex(tbl.edge))
    @test tbl.prior[idx] == LogNormal(log(2.0), 0.2)
    fixed = findfirst(i -> tbl.edge[i] == :onset_admit &&
                           tbl.param[i] == :scale,
        eachindex(tbl.edge))
    @test tbl.prior[fixed] === nothing

    # Precedence: attached spec beats the default, override beats both.
    nested = build_priors(tbl)
    @test nested.onset_admit.shape == LogNormal(log(2.0), 0.2)
    @test nested.onset_admit.scale isa Distribution
    ovr = build_priors(tbl;
        priors = (onset_admit = (shape = Exponential(1.0),),))
    @test ovr.onset_admit.shape == Exponential(1.0)

    # A four-column table (no prior column) still works.
    legacy = (edge = collect(tbl.edge), param = collect(tbl.param),
        value = collect(tbl.value), support = collect(tbl.support))
    @test build_priors(legacy).onset_admit.shape isa Distribution

    # The table prints, with blank prior cells for fixed rows.
    @test occursin("prior", sprint(show, MIME("text/plain"), tbl))
end

@testitem "observed_distribution rejects an uncertain chain" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    seq = sequential(:onset_admit => u, :admit_death => LogNormal(0.5, 0.4))
    # The reworded guidance points at `update(tree, params)`, not `realise`.
    @test_throws "update(tree, params)" observed_distribution(seq)

    # After collapsing with `update`, the chain lowers to its convolved total.
    collapsed = update(seq, (onset_admit = (shape = 2.0, scale = 1.0),
        admit_death = (mu = 0.5, sigma = 0.4)))
    @test observed_distribution(collapsed) isa UnivariateDistribution
end

@testitem "uncertain composes across the verb surface" begin
    using Distributions, Random

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))

    # Sequential: samples, and its tree moment reads the template's free delay.
    s = sequential(:onset_admit => u, :admit_death => LogNormal(0.5, 0.4))
    @test isfinite(sum(values(rand(Xoshiro(1), s))))
    @test mean(s) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))

    # Parallel and Compete sample the uncertain leaf directly; a one_of node's
    # `rand` is its named event record, which scores finitely through `logpdf`.
    p = parallel(:admit => u, :notif => LogNormal(1.0, 0.5))
    @test rand(Xoshiro(1), p) isa NamedTuple

    c = compete(:death => u, :recover => Gamma(3.0, 2.0))
    @test isfinite(logpdf(c, rand(Xoshiro(1), c)))

    # Resolve: it composes and collapses via `update` to a concrete node. The
    # uncollapsed node draws the resolved outcome's uncertain delay directly and
    # its marginal (`as_mixture`) needs `Uncertain` to report a concrete
    # `ValueSupport` (not the abstract type) to dispatch correctly when a branch
    # sits alongside a differently-typed sibling.
    r = resolve(:death => (u, 0.3), :disch => Gamma(2.0, 1.5))
    @test isfinite(logpdf(r, rand(Xoshiro(1), r)))
    @test isfinite(rand(Xoshiro(1), as_mixture(r)))
    rc = update(r, (death = (shape = 2.0, scale = 1.0),
        disch = (shape = 2.0, scale = 1.5)))
    @test !has_uncertain(rc)
    @test isfinite(logpdf(rc, rand(Xoshiro(1), rc)))

    # Choose: it composes, and its prior column carries the spec.
    ch = choose(:index => u, :sourced => Gamma(2.0, 1.0))
    tbl = params_table(ch)
    idx = findfirst(i -> tbl.edge[i] == :index && tbl.param[i] == :shape,
        eachindex(tbl.edge))
    @test tbl.prior[idx] == LogNormal(log(2.0), 0.2)

    # event / show / event_names see the uncertain leaf like any leaf.
    @test event(s, :onset_admit) === u
    @test event_names(s) == (:onset, :admit, :death)
    @test occursin("uncertain(", sprint(show, MIME("text/plain"), s))
end

@testitem "has_uncertain: flags a tree across composer shapes" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))

    @test has_uncertain(u)
    @test !has_uncertain(Gamma(2.0, 1.0))

    s = sequential(:onset_admit => u, :admit_death => LogNormal(0.5, 0.4))
    @test has_uncertain(s)
    @test !has_uncertain(sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4)))

    p = parallel(:admit => u, :notif => LogNormal(1.0, 0.5))
    @test has_uncertain(p)

    r = resolve(:death => (u, 0.3), :disch => Gamma(2.0, 1.5))
    @test has_uncertain(r)

    c = compete(:death => u, :recover => Gamma(3.0, 2.0))
    @test has_uncertain(c)

    ch = choose(:index => u, :sourced => Gamma(2.0, 1.0))
    @test has_uncertain(ch)

    # A shared-tagged uncertain leaf is still visible through the tag.
    @test has_uncertain(shared(:inc, u))

    # Collapsing with `update` removes the uncertainty from the tree.
    rc = update(r, (death = (shape = 2.0, scale = 1.0),
        disch = (shape = 2.0, scale = 1.5)))
    @test !has_uncertain(rc)
end

@testitem "update introduces uncertainty on a plain leaf (targeted)" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))

    # A distribution in a parameter slot makes just that parameter uncertain;
    # a partial NamedTuple leaves the rest of the tree fixed.
    est = update(tree, (onset_admit = (shape = LogNormal(log(2.0), 0.2),),))
    @test has_uncertain(est)
    leaf = event(est, :onset_admit)
    @test leaf isa Uncertain
    @test leaf.specs == (; shape = LogNormal(log(2.0), 0.2))
    @test leaf.template == Gamma(2.0, 1.0)   # scale stays fixed at 1.0
    # admit_death is untouched and stays a fixed leaf.
    @test event(est, :admit_death) == LogNormal(0.5, 0.4)
    @test !has_uncertain(event(est, :admit_death))
    # The estimation boundary follows: exactly one estimated parameter.
    @test flat_dimension(est) == 1
end

@testitem "update merges, extends and collapses uncertain specs" begin
    using Distributions

    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    tree = compose((onset_admit = u,))

    # Extending: add a spec on scale while keeping the shape spec (merge).
    ext = update(tree, (onset_admit = (scale = Exponential(1.0),),))
    leaf = event(ext, :onset_admit)
    @test leaf isa Uncertain
    @test keys(leaf.specs) == (:shape, :scale)
    @test leaf.specs.shape == LogNormal(log(2.0), 0.2)
    @test leaf.specs.scale == Exponential(1.0)

    # A Real on a spec'd parameter drops that spec and pins the value, while a
    # distribution on another parameter (re)specs it in the same call.
    col = update(tree, (onset_admit = (shape = 3.0, scale = Exponential(1.0)),))
    cleaf = event(col, :onset_admit)
    @test keys(cleaf.specs) == (:scale,)
    @test cleaf.template == Gamma(3.0, 1.0)
end

@testitem "promote: update(tree, param_priors(tree)) specs every parameter" begin
    using Distributions
    using ComposedDistributions: flat_dimension

    tree = compose((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    promoted = update(tree, param_priors(tree))

    @test has_uncertain(promoted)
    # Every free parameter is now estimated (the escape hatch to estimate-all).
    @test flat_dimension(promoted) == length(params_table(tree).edge)
    oa = event(promoted, :onset_admit)
    @test oa isa Uncertain
    @test keys(oa.specs) == (:shape, :scale)
    # The specs are the support-derived defaults.
    @test oa.specs.scale == param_priors(tree).onset_admit.scale
end

@testitem "promote keeps an existing spec and defaults the rest" begin
    using Distributions

    tree = compose((onset_admit = uncertain(Gamma(2.0, 1.0);
        shape = LogNormal(log(2.0), 0.2)),))
    promoted = update(tree, param_priors(tree))
    oa = event(promoted, :onset_admit)
    @test keys(oa.specs) == (:shape, :scale)
    # The user's shape spec is kept; scale gets the support-derived default.
    @test oa.specs.shape == LogNormal(log(2.0), 0.2)
    @test oa.specs.scale == param_priors(tree).onset_admit.scale
end

@testitem "promote through a shared tag ties the estimated parameter" begin
    using Distributions
    using ComposedDistributions: flat_dimension, _shared_tag

    d = compose((a = shared(:g, Gamma(2.0, 1.0)),
        b = shared(:g, Gamma(2.0, 1.0))))
    promoted = update(d, param_priors(d))

    @test has_uncertain(promoted)
    # The tied leaf is inventoried once, so shape + scale = two estimated rows.
    @test flat_dimension(promoted) == 2
    la = event(promoted, :a)
    @test _shared_tag(la) == :g
    @test has_uncertain(la)
    # Collapsing the promoted tree from a flat draw keeps the tie: both
    # occurrences are the same shared leaf at the drawn values.
    collapsed = update(promoted, ComposedDistributions.unflatten(promoted,
        [2.5, 1.2]))
    @test !has_uncertain(collapsed)
    @test event(collapsed, :a) == event(collapsed, :b)
    @test ComposedDistributions.free_leaf(event(collapsed, :a)) ==
          Gamma(2.5, 1.2)
end

@testitem "update merge rejects an unknown or ill-typed parameter" begin
    using Distributions

    tree = compose((onset_admit = Gamma(2.0, 1.0),))
    # An unknown parameter name in a merge update errors.
    @test_throws ArgumentError update(tree,
        (onset_admit = (rate = LogNormal(0.0, 1.0),),))
    # A non-Real, non-distribution value errors.
    @test_throws ArgumentError update(tree,
        (onset_admit = (shape = LogNormal(0.0, 1.0), scale = "no"),))
end

@testitem "rand/logpdf round trip through a nested uncertain leaf" begin
    using Distributions, Random

    # A Sequential containing a Parallel branch, one of whose endpoints is an
    # uncertain leaf: exercises the composed `rand`/`logpdf` surfaces (the
    # package's stack of per-leaf draws / densities) on a genuinely nested
    # tree shape, not just a flat sibling list.
    u = uncertain(Gamma(2.0, 1.0); shape = LogNormal(log(2.0), 0.2))
    inner = parallel(:admit => u, :notif => LogNormal(1.0, 0.5))
    tree = sequential(:onset => Gamma(1.5, 1.0), :branch => inner)

    draw = rand(Xoshiro(7), tree)
    @test draw isa NamedTuple
    @test all(isfinite, values(draw))

    # Scoring the draw is finite and, since `logpdf` on an uncertain leaf
    # reports the template's density (not the marginal, by design), matches a
    # hand-built tree with the SAME template values substituted for `u`.
    flat = collect(values(draw))
    @test isfinite(logpdf(tree, flat))

    tmpl_tree = sequential(:onset => Gamma(1.5, 1.0),
        :branch => parallel(:admit => Gamma(2.0, 1.0),
            :notif => LogNormal(1.0, 0.5)))
    @test logpdf(tree, flat) == logpdf(tmpl_tree, flat)
end
