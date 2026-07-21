# The reserved per-record field registry (`reserved_record_fields`) is public,
# and a row that misspells a reserved field raises a helpful error naming the
# reserved field rather than a bare "not an event". Additive: valid records
# (including reserved fields) score exactly as before. See #262.

@testitem "reserved_record_fields: publishes the reserved row fields" begin
    using ComposedDistributions: reserved_record_fields

    r = reserved_record_fields()
    @test r isa Tuple
    @test :weight in r && :count in r
    @test :obs_time in r && :obs_window in r
    @test :branch_probs in r && :branch_prob in r
end

@testitem "reserved_record_fields: a valid record still scores unchanged" begin
    using ComposedDistributions: resolve
    using Distributions, Random

    c = resolve(:left => (Gamma(1.5, 1.0), 0.3), :right => Gamma(2.0, 1.5))
    rec = rand(Xoshiro(1), c)
    base = logpdf(c, rec)
    # a reserved field rides the record without being treated as an event
    @test logpdf(c, merge(rec, (obs_time = 20.0,))) isa Real
    @test logpdf(c, rec) == base            # unchanged behaviour
end

@testitem "reserved_record_fields: a misspelled reserved field is loud" begin
    using ComposedDistributions: resolve
    using Distributions, Random

    c = resolve(:left => (Gamma(1.5, 1.0), 0.3), :right => Gamma(2.0, 1.5))
    rec = rand(Xoshiro(1), c)
    # `obs_tim` is a near-miss of the reserved `obs_time`
    err = try
        logpdf(c, merge(rec, (obs_tim = 5.0,)))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("obs_time", msg)
    @test occursin("Reserved fields", msg)
    # a genuine unknown (not near a reserved field) gets no reserved hint
    err2 = try
        logpdf(c, merge(rec, (zzzzzz = 5.0,)))
        nothing
    catch e
        e
    end
    @test !occursin("Did you mean", sprint(showerror, err2))
end

@testitem "reserved_record_fields: near-miss detection" begin
    using ComposedDistributions: _reserved_near_miss, _edit_distance

    @test _edit_distance("obs_time", "obs_tim") == 1
    @test _edit_distance("weight", "wieght") == 2       # transposition -> 2
    @test _reserved_near_miss(:obs_tim) === :obs_time
    @test _reserved_near_miss(:branch_prb) === :branch_prob
    @test _reserved_near_miss(:zzzzzz) === nothing
    @test _reserved_near_miss(:onset) === nothing       # a plausible event name
end
