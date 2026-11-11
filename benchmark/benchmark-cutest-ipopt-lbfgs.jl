using JLD2

using CUTEst, NLPModelsIpopt, SolverBenchmark

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
const CERTIFY_LOCAL_INFEASIBILITY = parse(Bool, get(ENV, "CERTIFY_INFEASIBILITY", "false"))

filter!(x -> x != "FLOSP2TL", problem_names) # This problem of the form min 0 st c(x) = 0 causes the benchmarks to timeout...

# Speedup benchmark time for BFGS
# Split problems across 4 runners.
split = parse(Int, get(ENV, "CUTEST_SPLIT", "1"))
n_splits = 4

@assert 1 <= split <= n_splits

problem_names = collect(problem_names)

n = length(problem_names)
first = fld((split - 1) * n, n_splits) + 1
last = fld(split * n, n_splits)

problem_names = problem_names[first:last]

@info "Running CUTEst split $split/$n_splits: problems $first:$last ($(length(problem_names)) problems)"

problem_list = (CUTEstModel(name) for name in problem_names)

solvers = Dict(
  :ipopt_lbfgs => nlp -> begin
    # Solve the problem
    stats = ipopt(
      nlp,
      print_level = 0,
      tol = BENCHMARK_TOL,
      dual_inf_tol = BENCHMARK_TOL,
      constr_viol_tol = BENCHMARK_TOL,
      compl_inf_tol = Inf,
      acceptable_iter = 0,
      s_max = floatmax(Float64),
      max_cpu_time = BENCHMARK_MAX_TIME,
      hessian_approximation = "limited-memory",
      max_iter = typemax(Int32)
    )

    set_solver_specific!(stats, :certified_infeasible, false)
    # Set success
    if CERTIFY_LOCAL_INFEASIBILITY && stats.status == :infeasible
      certify_local_infeasibility(nlp, stats.solution) && set_solver_specific!(stats, :certified_infeasible, true)
    end
  end
)

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_ipopt_lbfgs_$(split).jld2" stats
