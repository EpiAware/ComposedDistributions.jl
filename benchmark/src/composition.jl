# Core composition-algebra benchmarks: construction, `rand`, and `logpdf`
# (or the node's own marginal, e.g. `logccdf` for a `Compete`) over one
# representative tree per composer, plus the `compose` front-end lowering a
# NamedTuple into a small nested (Parallel-of-Sequential/Resolve) tree.

SUITE["Composition"] = BenchmarkGroup()

# --- Sequential: a two-step delay chain ---

let
    s = sequential(:onset_admit => Gamma(2.0, 1.0),
        :admit_death => LogNormal(0.5, 0.4))
    x = [1.5, 0.8]
    SUITE["Composition"]["Sequential"] = BenchmarkGroup()
    SUITE["Composition"]["Sequential"]["construct"] = @benchmarkable sequential(
        :onset_admit => Gamma(2.0, 1.0), :admit_death => LogNormal(0.5, 0.4))
    SUITE["Composition"]["Sequential"]["logpdf"] = @benchmarkable logpdf($s, $x)
    SUITE["Composition"]["Sequential"]["rand"] = @benchmarkable rand($s)
end

# --- Parallel: independent branches sharing no origin ---

let
    p = parallel(:admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    x = [1.2, 2.3]
    SUITE["Composition"]["Parallel"] = BenchmarkGroup()
    SUITE["Composition"]["Parallel"]["construct"] = @benchmarkable parallel(
        :admit => Gamma(2.0, 1.0), :notif => LogNormal(1.0, 0.5))
    SUITE["Composition"]["Parallel"]["logpdf"] = @benchmarkable logpdf($p, $x)
    SUITE["Composition"]["Parallel"]["rand"] = @benchmarkable rand($p)
end

# --- Resolve: fixed-probability mixture marginal ---

let
    r = resolve(:death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
    SUITE["Composition"]["Resolve"] = BenchmarkGroup()
    SUITE["Composition"]["Resolve"]["construct"] = @benchmarkable resolve(
        :death => (Gamma(1.5, 1.0), 0.3), :disch => (Gamma(2.0, 1.5), 0.7))
    SUITE["Composition"]["Resolve"]["logpdf"] = @benchmarkable logpdf($r, 2.0)
    SUITE["Composition"]["Resolve"]["rand"] = @benchmarkable rand($r)
end

# --- Compete: racing-hazard marginal (survival product) ---

let
    c = compete(:death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    SUITE["Composition"]["Compete"] = BenchmarkGroup()
    SUITE["Composition"]["Compete"]["construct"] = @benchmarkable compete(
        :death => Gamma(2.0, 3.0), :recover => Gamma(3.0, 2.0))
    SUITE["Composition"]["Compete"]["logccdf"] = @benchmarkable logccdf($c, 5.0)
    SUITE["Composition"]["Compete"]["rand"] = @benchmarkable rand($c)
end

# --- Choose: data-selected disjunction ---

let
    d = choose(:short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    SUITE["Composition"]["Choose"] = BenchmarkGroup()
    SUITE["Composition"]["Choose"]["construct"] = @benchmarkable choose(
        :short => Gamma(2.0, 1.0), :long => Gamma(5.0, 1.0))
    SUITE["Composition"]["Choose"]["logpdf"] = @benchmarkable logpdf(
        $d, 3.0; kind = :short)
end

# --- compose: front-end lowering + scoring a representative nested tree ---
# A two-branch Parallel: one branch a two-step Sequential chain, the other a
# pre-built Resolve mixture leaf — the shape of a typical onset -> admission ->
# outcome model with a probabilistic (not racing-hazard) outcome split.

let
    nt = (path = [Gamma(2.0, 1.0), LogNormal(0.5, 0.4)],
        outcome = resolve(:death => (Gamma(3.0, 1.0), 0.3),
            :disch => (Gamma(2.0, 1.5), 0.7)))
    tree = compose(nt)
    draw = rand(tree)
    SUITE["Composition"]["Nested"] = BenchmarkGroup()
    SUITE["Composition"]["Nested"]["compose"] = @benchmarkable compose($nt)
    SUITE["Composition"]["Nested"]["rand"] = @benchmarkable rand($tree)
    SUITE["Composition"]["Nested"]["logpdf"] = @benchmarkable logpdf(
        $tree, $draw)
end
