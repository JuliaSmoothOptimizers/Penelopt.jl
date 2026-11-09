using DataFrames
using JLD2
using Measures
using Printf
using SolverBenchmark
using Plots

include(joinpath(@__DIR__, "utils", "benchmark-utils.jl"))
include(joinpath(@__DIR__, "utils", "infeasibility-checker.jl"))
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

# Step 2: Certify Infeasibility results
@info "Infeasibility results\n"

reference_certification =
  CERTIFY_INFEASIBILITY ? load_precomputed_certification(reference_dir) : nothing
ipopt_certification =
  CERTIFY_INFEASIBILITY ? load_precomputed_certification(ipopt_dir) : nothing

certified_infeasible = Dict{Symbol,Set{String}}()

# reference isn't paired against ipopt below, so register it directly.
register_certified!(
  certified_infeasible,
  reference_certification,
  :l2penalty_exact_reference,
)
register_certified!(
  certified_infeasible,
  reference_certification,
  :l2penalty_lbfgs_reference,
)

register_certified!(certified_infeasible, ipopt_certification, :ipopt_exact)
register_certified!(certified_infeasible, ipopt_certification, :ipopt_lbfgs)

reports = DataFrame[]
if CERTIFY_INFEASIBILITY
  for key in [:l2penalty_exact_current, :l2penalty_lbfgs_current]
    push!(reports, certify_local_infeasibility(stats, key))
  end
end

exact_certification =
  isempty(reports) ?
  DataFrame(
    name = String[],
    hessian = Symbol[],
    status = Symbol[],
    certified_locally_infeasible = Union{Bool,Missing}[],
  ) : vcat(reports...)

register_certified!(certified_infeasible, exact_certification, :l2penalty_exact_current)
register_certified!(certified_infeasible, exact_certification, :l2penalty_lbfgs_current)

# Save the certification of the current run so that it can be downloaded as an artifact.
if CERTIFY_INFEASIBILITY
  mkpath("benchmark/result")
  infeasibility_certification = exact_certification
  @save "benchmark/result/infeasibility_certification.jld2" infeasibility_certification
end

for key in [:l2penalty_exact_current, :l2penalty_lbfgs_current]
  @info "Infeasibility certification results:\n" * sprint(
    io -> show(
      io,
      get(certified_infeasible, key, Set{String}());
      allrows = true,
      allcols = true,
    ),
  )
end

mkpath("benchmark/result")

p = plot(
  pairwise_plot(
    stats,
    [:l2penalty_exact_reference, :l2penalty_exact_current],
    compare_n_fact = true,
    certified_infeasible = certified_infeasible,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_reference, :l2penalty_lbfgs_current],
    compare_n_fact = true,
    certified_infeasible = certified_infeasible,
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
    certified_infeasible = certified_infeasible,
  ),
  pairwise_plot(
    stats,
    [:l2penalty_lbfgs_current, :ipopt_lbfgs],
    certified_infeasible = certified_infeasible,
  ),
  layout = (2, 1),
  size = (1920, 1080),
)

savefig(p, "benchmark/result/benchmark_comparison_ipopt.svg")
