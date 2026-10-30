using DataFrames
using JLD2
using Penelopt
using NLPModelsIpopt

const METHODS = (:exact, :lbfgs)

"""
    load_stats(dir, stats, suffix = "")

Load every stats split under `dir`, concatenate, merge into `stats` under
keys named `<key><suffix>`.
"""
function load_stats(dir::AbstractString, stats, suffix = "")

  for method in METHODS

    @info "Loading $(method) benchmark results"

    file_splits = String[]

    for (root, _, files) in walkdir(dir)
      for file in files
        if (
          startswith(file, "stats_$(method)") ||
          (startswith(file, "stats_ipopt_$(method)") && suffix == "")
        ) && occursin(r"\d+\.jld2$", file)
          push!(file_splits, joinpath(root, file))
        end
      end
    end

    sort!(file_splits)

    n_splits = length(file_splits)

    n_splits == 0 && continue

    # Load the first split and initialize the dictionary
    file = file_splits[1]
    @info "Loading $file"
    dict = load(file)["stats"]

    # Load the remaining splits and concatenate the data
    for split = 2:n_splits
      file = file_splits[split]
      @info "Loading $file"
      dict_split = load(file)["stats"]
      for key in keys(dict)
        append!(dict[key], dict_split[key])
      end
    end

    for key in keys(dict)
      new_key = Symbol("$(key)$suffix")
      stats[new_key] = dict[key]
    end
  end

  return stats
end

# Reproduces the exact kwargs of the corresponding benchmark script, so the
# candidate point x̄ is the actual point produced by the benchmarked run.
const BENCHMARK_TOL = 1e-6
const BENCHMARK_MAX_TIME = 300.0

const BENCHMARK_SOLVERS = Dict(
  (:l2penalty, :exact) =>
    nlp -> L2Penalty(
      nlp,
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      linear_solver = "mumps",
    ),
  (:l2penalty, :lbfgs) =>
    nlp -> L2Penalty(
      nlp,
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      qn_hessian_approximation = "bfgs",
      linear_solver = "mumps",
    ),
  (:ipopt, :exact) =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      tol = BENCHMARK_TOL,
      dual_inf_tol = BENCHMARK_TOL,
      constr_viol_tol = BENCHMARK_TOL,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = floatmax(Float64),
      nlp_scaling_method = "none",
      max_cpu_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int32),
    ),
  (:ipopt, :lbfgs) =>
    nlp -> ipopt(
      nlp,
      print_level = 0,
      tol = BENCHMARK_TOL,
      dual_inf_tol = BENCHMARK_TOL,
      constr_viol_tol = BENCHMARK_TOL,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = floatmax(Float64),
      hessian_approximation = "limited-memory",
      nlp_scaling_method = "none",
      max_cpu_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int32),
    ),
)
