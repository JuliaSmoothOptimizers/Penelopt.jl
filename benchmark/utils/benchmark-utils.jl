using DataFrames
using JLD2
using Penelopt
using NLPModelsIpopt
using NLPModelsModifiers
using CUTEst

"""
    benchmark_branch()

Branch the benchmark runs for. On a pull request, GitHub Actions sets
`GITHUB_HEAD_REF`/`GITHUB_BASE_REF`; otherwise `GITHUB_REF_NAME`. Falls back to
the local git branch.
"""
function benchmark_branch()
  refs = filter(!isempty, [get(ENV, v, "") for v in ("GITHUB_HEAD_REF", "GITHUB_BASE_REF", "GITHUB_REF_NAME")])
  isempty(refs) || return join(refs, " ")
  return try
    readchomp(`git rev-parse --abbrev-ref HEAD`)
  catch
    ""
  end
end

"""
    bounds_problem_set()

Whether to benchmark on all constrained problems (bounds-related branches) instead
of the equality-constrained, free-variable set. Override with
`BENCHMARK_PROBLEM_SET=bounds` or `BENCHMARK_PROBLEM_SET=equality`.
"""
function bounds_problem_set()
  set = lowercase(get(ENV, "BENCHMARK_PROBLEM_SET", ""))
  set == "bounds" && return true
  set == "equality" && return false
  return occursin("bounds", lowercase(benchmark_branch()))
end

function select_benchmark_problems()
  if bounds_problem_set()
    @info "Bounds-related branch ($(benchmark_branch())): all problems with at least one constraint"
    names = CUTEst.select_sif_problems(min_con = 1)
  else
    names = CUTEst.select_sif_problems(
      min_con = 1,
      only_equ_con = true,
      custom_filter = meta -> (
        meta["variables"]["number"] >= meta["constraints"]["number"] &&
        meta["variables"]["free"] + meta["variables"]["fixed"] == meta["variables"]["number"]
      ),
    )
  end
  return collect(names)
end

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
      SlackModel(nlp), # no-op without inequalities
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      linear_solver = "mumps",
    ),
  (:l2penalty, :lbfgs) =>
    nlp -> L2Penalty(
      SlackModel(nlp), # no-op without inequalities
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
      max_cpu_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int32),
    ),
)
