# End-to-end scenario tests: realistic CONTINUOUS delay stacks driven through
# the whole composition-verb surface at once. Each testitem is one named,
# epi-flavoured stack (the comment names the modelling story) and asserts the
# verbs work TOGETHER and CORRECTLY on it — construction (both spellings), rand
# / logpdf round-trip, a seeded large-N Monte-Carlo moment check against the
# analytic / quadrature values, the introspection + edit + prior surface, the
# convolution interop, and one ForwardDiff gradient. Public verbs only, so this
# doubles as usage documentation. Tolerances are chosen at several Monte-Carlo
# standard errors: loose enough for sampling noise, tight enough that a real
# bug (a wrong distribution, an unnormalised weight, a dropped step) fails.

@testitem "Scenario: onset→admission→death continuous chain" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: the line-list delay from symptom onset to hospital admission
    # (a Gamma incubation-like step) then admission to death (LogNormal), the
    # canonical two-step continuous reporting chain.
    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))

    # Construction via both spellings gives equal trees.
    @test chain == sequential((onset_admit = Gamma(2.0, 1.0),
        admit_death = LogNormal(0.5, 0.4)))
    @test event_names(chain) == (:onset, :admit, :death)

    # rand returns the labelled event record; logpdf round-trips (both the
    # NamedTuple and the equivalent value vector).
    rng = MersenneTwister(20240101)
    draw = rand(rng, chain)
    @test draw isa NamedTuple
    @test keys(draw) == (:onset_admit, :admit_death)
    @test isfinite(logpdf(chain, draw))
    @test logpdf(chain, draw) ≈ logpdf(chain, collect(values(draw)))

    # Large-N Monte Carlo of the per-step draws and the observed total matches
    # the analytic moments (mean/var additive over the independent steps).
    N = 40000
    onset = Vector{Float64}(undef, N)
    total = Vector{Float64}(undef, N)
    for i in 1:N
        d = rand(rng, chain)
        onset[i] = d.onset_admit
        total[i] = d.onset_admit + d.admit_death
    end
    @test mean(onset) ≈ mean(Gamma(2.0, 1.0)) atol = 0.05
    @test mean(total) ≈ mean(chain) atol = 0.08     # mean(chain) ≈ 3.786
    @test var(total) ≈ var(chain) rtol = 0.05       # var(chain) ≈ 2.553

    # observed_distribution collapses the chain to its convolved total; its
    # moments and a cdf point match the simulated step-sums.
    od = observed_distribution(chain)
    @test od isa Convolved
    @test mean(od) ≈ mean(chain)
    @test var(od) ≈ var(chain)
    @test cdf(od, 5.0) ≈ count(<=(5.0), total) / N atol = 0.01

    # params_table row count and labels.
    tbl = params_table(chain)
    @test tbl.edge == [:onset_admit, :onset_admit, :admit_death, :admit_death]
    @test tbl.param == [:shape, :scale, :mu, :sigma]

    # build_priors produces a prior per free parameter (four params, four
    # priors; no Resolve simplex to fold here).
    priors = build_priors(tbl)
    @test priors.onset_admit.shape isa Distribution
    @test priors.onset_admit.scale isa Distribution
    @test priors.admit_death.mu isa Distribution
    @test priors.admit_death.sigma isa Distribution
    @test param_priors(chain) == priors

    # update replaces the values; rand/logpdf reflect the change.
    tuned = update(chain, (onset_admit = (shape = 3.0, scale = 1.5),
        admit_death = (mu = 0.7, sigma = 0.5)))
    @test event(tuned, :onset_admit) == Gamma(3.0, 1.5)
    @test mean(tuned) != mean(chain)
    @test logpdf(tuned, draw) != logpdf(chain, draw)

    # One ForwardDiff gradient of logpdf w.r.t. the observation, finite.
    g = ForwardDiff.gradient(v -> logpdf(chain, v), collect(values(draw)))
    @test length(g) == 2 && all(isfinite, g)
end

@testitem "Scenario: independent parallel reporting branches" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: from a shared symptom onset, two independent observation branches
    # run in parallel — a hospitalisation pathway (onset→admit→discharge chain)
    # and a separate notification delay (a single leaf).
    par = parallel(
        :hosp => sequential(:onset_admit => Gamma(2.0, 1.0),
            :admit_disch => LogNormal(0.6, 0.3)),
        :notify => Gamma(1.5, 1.0))

    # Both spellings build the same tree.
    @test par == parallel((
        hosp = sequential(:onset_admit => Gamma(2.0, 1.0),
            :admit_disch => LogNormal(0.6, 0.3)),
        notify = Gamma(1.5, 1.0)))

    # rand returns the labelled record (nested branch names joined); logpdf
    # round-trips.
    rng = MersenneTwister(20240202)
    draw = rand(rng, par)
    @test keys(draw) == (:hosp_onset_admit, :hosp_admit_disch, :notify)
    @test isfinite(logpdf(par, draw))
    @test logpdf(par, draw) ≈ logpdf(par, collect(values(draw)))

    # A Parallel is genuinely multivariate: mean/var are the per-endpoint
    # NamedTuples. Monte-Carlo the two independent endpoint totals against them.
    m = mean(par)
    v = var(par)
    @test m isa NamedTuple && keys(m) == (:hosp, :notify)
    N = 40000
    hosp = Vector{Float64}(undef, N)
    notify = Vector{Float64}(undef, N)
    for i in 1:N
        d = rand(rng, par)
        hosp[i] = d.hosp_onset_admit + d.hosp_admit_disch
        notify[i] = d.notify
    end
    @test mean(hosp) ≈ m.hosp atol = 0.08      # m.hosp ≈ 3.906
    @test mean(notify) ≈ m.notify atol = 0.05  # m.notify == 1.5
    @test var(hosp) ≈ v.hosp rtol = 0.06

    # params_table names each branch with a dotted edge path; build_priors gives
    # a prior per free parameter.
    tbl = params_table(par)
    @test tbl.edge == [Symbol("hosp.onset_admit"), Symbol("hosp.onset_admit"),
        Symbol("hosp.admit_disch"), Symbol("hosp.admit_disch"),
        :notify, :notify]
    priors = param_priors(par)
    @test priors.hosp.onset_admit.shape isa Distribution
    @test priors.notify.shape isa Distribution

    # One ForwardDiff gradient over the three-value observation, finite.
    g = ForwardDiff.gradient(v -> logpdf(par, v), collect(values(draw)))
    @test length(g) == 3 && all(isfinite, g)
end

@testitem "Scenario: death-vs-discharge case-fatality Resolve" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: an admitted case resolves to exactly one outcome — death with the
    # case-fatality probability, discharge otherwise — cause independent of
    # timing (the fixed-probability mixture).
    cfr = 0.3
    res = resolve(:death => (Gamma(1.5, 1.0), cfr),
        :disch => (Gamma(2.0, 1.5), 1 - cfr))

    # Both spellings; probs reads back the declared branch split.
    @test res == resolve((death = (Gamma(1.5, 1.0), cfr),
        disch = (Gamma(2.0, 1.5), 1 - cfr)))
    @test probs(res) == (death = 0.3, disch = 0.7)
    @test occurrence_probability(res) ≈ 1.0

    # The marginal time-to-resolution equals the MixtureModel lowering.
    mix = MixtureModel([Gamma(1.5, 1.0), Gamma(2.0, 1.5)], [cfr, 1 - cfr])
    @test mean(res) ≈ mean(mix)    # == 2.55
    @test var(res) ≈ var(mix)      # == 4.0725

    # rand returns the named event record; logpdf round-trips it.
    rng = MersenneTwister(20240303)
    rec = rand(rng, res)
    @test keys(rec) == (:event_1, :death, :disch)
    @test isfinite(logpdf(res, rec))
    # The count form draws a vector of records scoring as the row sum.
    recs = rand(rng, res, 8)
    @test recs isa Vector && length(recs) == 8
    @test logpdf(res, recs) ≈ sum(logpdf(res, x) for x in recs)

    # Monte-Carlo winning frequency matches probs; the marginal-time moments
    # match the mixture.
    N = 40000
    outcomes = [rand(rng, res; outcome = true) for _ in 1:N]
    times = [o[2] for o in outcomes]
    @test count(o -> o[1] == :death, outcomes) / N ≈
          probs(res).death atol = 0.02   # 0.30
    @test mean(times) ≈ mean(res) atol = 0.05
    @test var(times) ≈ var(res) rtol = 0.06

    # prune drops an arm and renormalises the survivors proportionally.
    three = resolve(:death => (Gamma(1.5, 1.0), 0.3),
        :disch => (Gamma(2.0, 1.5), 0.5),
        :transfer => (Gamma(1.0, 1.0), 0.2))
    pruned = prune(three, :transfer)
    @test sum(values(probs(pruned))) ≈ 1.0
    @test probs(pruned).death ≈ 0.3 / 0.8   # proportional to the survivors
    @test probs(pruned).disch ≈ 0.5 / 0.8

    # update: node-replace swaps a delay (probs unchanged), a value-update moves
    # the branch split.
    swapped = update(res, :death => Gamma(3.0, 2.0))
    @test event(swapped, :death) == Gamma(3.0, 2.0)
    @test probs(swapped) == probs(res)
    reweighted = update(res,
        (death = (shape = 1.5, scale = 1.0),
            disch = (shape = 2.0, scale = 1.5),
            branch_probs = (death = 0.6, disch = 0.4)))
    @test probs(reweighted) == (death = 0.6, disch = 0.4)

    # param_priors: a prior per delay parameter and a Dirichlet over the simplex.
    priors = param_priors(res)
    @test priors.death.shape isa Distribution
    @test priors.branch_probs isa Dirichlet

    # One ForwardDiff derivative of the marginal logpdf w.r.t. the time, finite.
    g = ForwardDiff.derivative(t -> logpdf(res, t), 2.0)
    @test isfinite(g)
end

@testitem "Scenario: competing causes racing hazard (Compete)" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: two causes race for the first event — death vs recovery — the
    # winner and time coupled through the hazards (min of the cause delays); the
    # winning probability is DERIVED, not declared.
    comp = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    @test comp == compete((death = Gamma(2.0, 3.0), recover = Gamma(3.0, 2.0)))

    # The marginal any-event survival is the product of the cause survivals.
    @test logccdf(comp, 5.0) ≈ logccdf(Gamma(2.0, 3.0), 5.0) +
                               logccdf(Gamma(3.0, 2.0), 5.0)

    # Derived winning probabilities sum to one for proper causes.
    wp = probs(comp)
    @test sum(values(wp)) ≈ 1.0 atol = 1e-3        # (0.525, 0.475)

    # rand returns the winning cause's record; logpdf round-trips it.
    rng = MersenneTwister(20240404)
    rec = rand(rng, comp)
    @test keys(rec) == (:event_1, :death, :recover)
    @test isfinite(logpdf(comp, rec))

    # Monte-Carlo the race: the empirical winner frequency matches the DERIVED
    # probs, and the marginal min-time moments match the quadrature values.
    N = 40000
    outcomes = [rand(rng, comp; outcome = true) for _ in 1:N]
    mins = [o[2] for o in outcomes]
    @test count(o -> o[1] == :death, outcomes) / N ≈
          wp.death atol = 0.02                     # 0.525
    @test mean(mins) ≈ mean(comp) atol = 0.1       # mean(comp) ≈ 3.926
    @test var(mins) ≈ var(comp) rtol = 0.06        # var(comp) ≈ 5.458

    # params_table lists only the cause-delay parameters (no free simplex).
    tbl = params_table(comp)
    @test tbl.edge == [:death, :death, :recover, :recover]

    # One ForwardDiff derivative of the marginal logpdf, finite.
    g = ForwardDiff.derivative(t -> logpdf(comp, t), 3.0)
    @test isfinite(g)
end

@testitem "Scenario: chain resolving to a death-or-discharge outcome" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: onset→admission (a delay leaf), then the admission RESOLVES to
    # death or discharge (a nested Resolve as the terminal step) — a continuous
    # chain feeding a one_of outcome.
    stack = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_out => resolve(:death => (Gamma(1.5, 1.0), 0.4),
            :disch => Gamma(2.0, 1.5)))

    @test event(stack, :admit_out) isa Resolve

    # rand/logpdf round-trip through the nested one_of (a scalar value slot).
    rng = MersenneTwister(20240505)
    draw = rand(rng, stack)
    @test keys(draw) == (:onset_admit, :admit_out)
    @test draw.admit_out isa Real
    @test isfinite(logpdf(stack, draw))
    @test logpdf(stack, draw) ≈ logpdf(stack, collect(values(draw)))

    # Monte-Carlo the observed total (onset gap plus resolution time) against the
    # additive analytic mean (leaf mean plus the mixture marginal mean).
    N = 40000
    total = Vector{Float64}(undef, N)
    for i in 1:N
        d = rand(rng, stack)
        total[i] = d.onset_admit + d.admit_out
    end
    @test mean(total) ≈ mean(stack) atol = 0.08
    @test var(total) ≈ var(stack) rtol = 0.06

    # params_table descends into the nested Resolve (delay params plus the
    # branch-probability rows).
    tbl = params_table(stack)
    @test :onset_admit in tbl.edge
    @test Symbol("admit_out.death") in tbl.edge
    @test Symbol("admit_out.branch_probs") in tbl.edge

    # update descends by path into the nested Resolve to replace a cause's
    # delay (prune of a Resolve arm is covered in the case-fatality scenario).
    edited = update(stack, (:admit_out, :death) => Gamma(3.0, 2.0))
    @test event(edited, :admit_out, :death) == Gamma(3.0, 2.0)

    # One ForwardDiff gradient over the observation, finite.
    g = ForwardDiff.gradient(v -> logpdf(stack, v), collect(values(draw)))
    @test length(g) == 2 && all(isfinite, g)
end

@testitem "Scenario: competing-risks arm beside a reporting arm" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: two independent arms in parallel — a clinical fate arm where death
    # and recovery race (Compete) and a separate reporting chain (onset→notify→
    # confirm). Mixes a racing one_of and a continuous chain under one Parallel.
    mix = parallel(
        :fate => compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0)),
        :report => sequential(:onset_notify => Gamma(1.5, 1.0),
            :notify_confirm => Gamma(1.0, 1.5)))

    @test event(mix, :fate) isa Compete
    @test event(mix, :report) isa Sequential

    # rand/logpdf round-trip (the Compete arm is one scalar value slot).
    rng = MersenneTwister(20240606)
    draw = rand(rng, mix)
    @test keys(draw) == (:fate, :report_onset_notify, :report_notify_confirm)
    @test isfinite(logpdf(mix, draw))
    @test logpdf(mix, draw) ≈ logpdf(mix, collect(values(draw)))

    # Per-endpoint moments: the fate endpoint is the racing marginal, the report
    # endpoint the reporting chain's total. Monte-Carlo both.
    m = mean(mix)
    @test keys(m) == (:fate, :report)
    @test m.fate ≈ mean(event(mix, :fate))
    @test m.report ≈ mean(event(mix, :report))
    N = 40000
    fate = Vector{Float64}(undef, N)
    report = Vector{Float64}(undef, N)
    for i in 1:N
        d = rand(rng, mix)
        fate[i] = d.fate
        report[i] = d.report_onset_notify + d.report_notify_confirm
    end
    @test mean(fate) ≈ m.fate atol = 0.1
    @test mean(report) ≈ m.report atol = 0.08

    # The derived winning split of the nested Compete is reachable via event.
    @test sum(values(probs(event(mix, :fate)))) ≈ 1.0 atol = 1e-3

    # One ForwardDiff gradient over the observation, finite.
    g = ForwardDiff.gradient(v -> logpdf(mix, v), collect(values(draw)))
    @test length(g) == 3 && all(isfinite, g)
end

@testitem "Scenario: deep compose of chain, parallel and Resolve" tags = [:scenarios] begin
    using Distributions, Random, Statistics
    using ForwardDiff

    # Story: a shared incubation delay, then the case splits into a parallel of
    # a hospitalisation chain and a fatal-vs-discharge Resolve — the compose
    # front-end lowering a NamedTuple that mixes all three multi-child verbs.
    deep = compose((
        incubation = Gamma(2.0, 1.0),
        branches = parallel(
            :hosp => sequential(:admit_disch => LogNormal(0.6, 0.3)),
            :fatal => resolve(:death => (Gamma(1.5, 1.0), 0.4),
                :disch => Gamma(2.0, 1.5)))))
    @test deep isa Parallel

    # rand/logpdf round-trip through the whole nest.
    rng = MersenneTwister(20240707)
    draw = rand(rng, deep)
    @test keys(draw) ==
          (:incubation, :branches_hosp_admit_disch, :branches_fatal)
    @test isfinite(logpdf(deep, draw))
    @test logpdf(deep, draw) ≈ logpdf(deep, collect(values(draw)))

    # params_table walks the full tree with dotted edge paths.
    edges = params_table(deep).edge
    @test :incubation in edges
    @test Symbol("branches.hosp.admit_disch") in edges
    @test Symbol("branches.fatal.death") in edges
    @test Symbol("branches.fatal.branch_probs") in edges

    # update node-replace keeps the shape and swaps the incubation leaf.
    edited = update(deep, :incubation => Gamma(3.0, 0.7))
    @test event(edited, :incubation) == Gamma(3.0, 0.7)

    # One ForwardDiff gradient over the observation, finite.
    g = ForwardDiff.gradient(v -> logpdf(deep, v), collect(values(draw)))
    @test length(g) == 3 && all(isfinite, g)
end

@testitem "Scenario: renewal convolution through a delay chain" tags = [:scenarios] begin
    using Distributions

    # Story: an infection incidence series pushed through the onset→admission→
    # death delay chain to expected downstream counts — the EpiNow2-style latent
    # observation layer, driven by a composed chain rather than a bare delay.
    chain = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    infections = [0.0, 1.0, 3.0, 6.0, 8.0, 5.0, 2.0, 1.0, 0.0, 0.0]

    # Convolving through the chain is identical to collapsing it to its observed
    # total, discretising that (ConvolvedDistributions 0.2 is discrete-only), and
    # convolving the PMF — the pre-0.2 continuous output unchanged.
    counts = convolve_series(chain, infections)
    @test length(counts) == length(infections)
    @test counts ≈ convolve_series(
        discretise_pmf(observed_distribution(chain), length(infections) - 1),
        infections)

    # Selecting the chain's interim events gives the count series at each event;
    # the terminal event reproduces the whole-chain result, and the first event
    # is just the first step's convolution.
    by_event = convolve_series(chain, infections; events = (:admit, :death))
    @test keys(by_event) == (:admit, :death)
    @test by_event.death ≈ counts
    @test by_event.admit ≈ convolve_series(
        discretise_pmf(Gamma(2.0, 1.0), length(infections) - 1), infections)
end

@testitem "Scenario: shared incubation tied across two branches" tags = [:scenarios] begin
    using Distributions, Random

    # Story: two reporting branches that share ONE incubation period — tie makes
    # the two leaves a single free parameter, so the prior/params interface and
    # update treat them as one.
    d = compose((primary = Gamma(2.0, 1.0), secondary = Gamma(2.0, 1.0)))
    tied = tie(d, :primary, :secondary; name = :incubation)

    # params_table dedups the tied leaves to one row-group under the tag.
    @test unique(params_table(tied).edge) == [:incubation]

    # build_priors produces a single prior for the tied group.
    priors = param_priors(tied)
    @test keys(priors) == (:incubation,)
    @test priors.incubation.shape isa Distribution

    # update propagates the shared value to both leaves.
    updated = update(tied, (incubation = (shape = 4.0, scale = 0.5),))
    @test event(updated, :primary) == shared(:incubation, Gamma(4.0, 0.5))
    @test event(updated, :secondary) == shared(:incubation, Gamma(4.0, 0.5))

    # The tied tree still samples and scores (the tag is transparent).
    draw = rand(MersenneTwister(20240808), tied)
    @test isfinite(logpdf(tied, draw))
end

@testitem "Scenario: difference of two observed reporting totals" tags = [:scenarios] begin
    using Distributions, Random

    # Story: the gap between two observed reporting totals — an onset→admit→
    # death chain and an onset→report→confirm chain — as a Difference of the two
    # convolved totals.
    onset = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    report = sequential(:onset_report => Gamma(1.5, 1.0),
        :report_confirm => Gamma(1.0, 2.0))

    gap = difference(onset, report)
    @test gap isa Difference
    # Collapsing each chain first and differencing the totals is identical.
    @test gap == difference(observed_distribution(onset),
        observed_distribution(report))
    # The difference mean is the difference of the observed-total means.
    @test mean(gap) ≈ mean(onset) - mean(report)

    # A cdf point on the Difference of two composite (Convolved) totals. This is
    # the path ConvolvedDistributions #45 fixed: cdf on a Difference whose
    # members are themselves composites previously threw a _primal(::Tuple)
    # MethodError, so #122 asserted an additive-variance relation as a stand-in.
    # With 0.2 pinned the real cdf point holds; cross-check it against the
    # Monte-Carlo empirical cdf of the sampled gap.
    rng = MersenneTwister(20240201)
    N = 40000
    gaps = rand(rng, observed_distribution(onset), N) .-
           rand(rng, observed_distribution(report), N)
    @test cdf(gap, 0.5) ≈ count(<=(0.5), gaps) / N atol = 0.01
end
