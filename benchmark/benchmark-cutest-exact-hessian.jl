using JLD2

using CUTEst, Penelopt, SolverBenchmark, SolverCore

include(joinpath(@__DIR__, "utils", "infeasibility-checker.jl"))

const BENCHMARK_TOL = 1e-6
const BENCHMARK_MAX_TIME = 300.0

problem_names = CUTEst.select_sif_problems(
  min_con = 1,
  only_equ_con = true,
  custom_filter = meta -> (
    meta["variables"]["number"] >= meta["constraints"]["number"] &&
    meta["variables"]["free"] + meta["variables"]["fixed"] == meta["variables"]["number"]
  ),
)

# Speedup benchmark time for exact Hessian.
# Split problems across 2 runners.
split = parse(Int, get(ENV, "CUTEST_SPLIT", "1"))
n_splits = 2

@assert 1 <= split <= n_splits

problem_names = collect(problem_names)

n = length(problem_names)
first = fld((split - 1) * n, n_splits) + 1
last = fld(split * n, n_splits)

problem_names = problem_names[first:last]

@info "Running CUTEst split $split/$n_splits: problems $first:$last ($(length(problem_names)) problems)"

problem_list = (CUTEstModel(name) for name in problem_names)

solvers = Dict(
  :l2penalty_exact => nlp -> begin
    # Solve the problem
    stats = L2Penalty(
      nlp,
      print_level = 0,
      atol = BENCHMARK_TOL,
      rtol = 0.0,
      max_time = BENCHMARK_MAX_TIME,
      max_iter = typemax(Int),
      linear_solver = "mumps",)

    set_solver_specific!(stats, :certified_infeasible, false)
    if stats.status == :infeasible
      set_solver_specific!(stats, :certified_infeasible, certify_local_infeasibility(nlp, stats.solution))
    end
    return stats
  end
)

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_exact_$(split).jld2" stats
