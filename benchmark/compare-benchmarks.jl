using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

include(joinpath(@__DIR__, "utils", "benchmark-utils.jl"))
include(joinpath(@__DIR__, "utils", "compare-utils.jl"))

current_dir = joinpath("artifacts", "current")
reference_dir = joinpath("artifacts", "reference")
ipopt_dir = joinpath("artifacts", "ipopt")

# Step 1: Load benchmark stats
stats = Dict{Symbol,DataFrame}()

@info "Loading current benchmark results"
load_stats(current_dir, stats, "_current")

@info "Loading reference benchmark results"
load_stats(reference_dir, stats, "_reference")

@info "Loading ipopt benchmark results"
load_stats(ipopt_dir, stats, "")

mkpath("benchmark/result")

p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_reference, :l2penalty_exact_current],
    compare_n_fact = true,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_reference, :l2penalty_lbfgs_current],
    compare_n_fact = true,
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison.svg")

# Plot IPOPT
p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_current, :ipopt_exact],
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_current, :ipopt_lbfgs],
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison_ipopt.svg")
