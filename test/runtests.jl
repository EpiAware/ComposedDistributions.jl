# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Main test entry. Discovers `@testitem`s (the managed QA testset under
# `test/package/` plus the package's own unit tests) with TestItemRunner. The
# `:ad`-tagged items live under `test/ad/` with their own environment and run in
# dedicated per-backend CI, so they are excluded here (see test/ad/runtests.jl).
#
# Filters:
#   skip_quality  — skip the QA testset (fast local iteration)
#   quality_only  — run only the QA testset
#   readme_only   — run only `:readme`-tagged items (README/tutorial tests)
#
# The suite runs one test FILE at a time, announcing each (and the items it
# holds) before it starts and reporting its wall clock after. TestItemRunner
# prints nothing at all until its own summary, so a testitem that never
# returns produces a silent job that dies at the CI timeout with no clue what
# was running — that is how #356 stayed unlocated across several runs. The
# banners cost one extra discovery pass per file (a fraction of a second each)
# and buy a log that names where a hang is.

using TestItemRunner

# Restrict discovery to THIS package's test tree so a nested worktree's items
# are not globbed in. Trailing separator guards against sibling dirs sharing a
# string prefix.
const TEST_ROOT = normpath(@__DIR__) * Base.Filesystem.path_separator
const PACKAGE_ROOT = normpath(joinpath(@__DIR__, ".."))
in_this_package(ti) = startswith(normpath(ti.filename), TEST_ROOT)

const SELECT = if "skip_quality" in ARGS
    ti -> in_this_package(ti) && !(:quality in ti.tags) && !(:ad in ti.tags)
elseif "quality_only" in ARGS
    ti -> in_this_package(ti) && :quality in ti.tags
elseif "readme_only" in ARGS
    ti -> in_this_package(ti) && :readme in ti.tags
else
    ti -> in_this_package(ti) && !(:ad in ti.tags)
end

# Explicit flushes: a banner still sitting in a buffer when the process wedges
# tells nobody anything, which is the whole point of printing it.
function announce(msg)
    println(stdout, Libc.strftime("%H:%M:%S", time()), "  ", msg)
    flush(stdout)
end

# What would run, per file, without running any of it: the filter records and
# then rejects everything, so this only walks and parses the test tree. Its
# empty summary is swallowed.
function discover()
    plan = Dict{String, Vector{String}}()
    function record_only(ti)
        SELECT(ti) || return false
        push!(get!(plan, ti.filename, String[]), ti.name)
        return false
    end
    redirect_stdout(devnull) do
        TestItemRunner.run_tests(PACKAGE_ROOT; filter = record_only)
    end
    return plan
end

function main()
    plan = discover()
    files = sort!(collect(keys(plan)))
    total = sum(length, values(plan); init = 0)
    announce("running $total testitems in $(length(files)) files")

    failed = String[]
    first_failure = nothing
    for file in files
        rel = relpath(file, PACKAGE_ROOT)
        items = plan[file]
        announce("▶ $rel ($(length(items)) items)")
        for name in items
            announce("    · $name")
        end
        started = time()
        mark = "✓"
        try
            TestItemRunner.run_tests(PACKAGE_ROOT;
                filter = ti -> SELECT(ti) && ti.filename == file)
        catch err
            err isa InterruptException && rethrow()
            mark = "✗"
            push!(failed, rel)
            first_failure === nothing && (first_failure = err)
        end
        announce("$mark $rel ($(round(time() - started; digits = 1))s)")
    end

    isempty(failed) && return nothing
    announce("failing files: " * join(failed, ", "))
    throw(first_failure)
end

main()
