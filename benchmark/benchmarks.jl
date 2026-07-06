# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Benchmark suite definition. Build a BenchmarkTools `BenchmarkGroup` named
# `SUITE`; the managed `run.jl` / `compare.jl` consume it. Put AD-gradient
# benchmarks under the `"AD gradients"` group so the comparison comment folds
# them into a compact per-(scenario x backend) matrix. Edit freely.

using BenchmarkTools
using ComposedDistributions
using Distributions

const SUITE = BenchmarkGroup()

# Include benchmark definitions
include("src/composition.jl")
include("src/ad_gradients.jl")
