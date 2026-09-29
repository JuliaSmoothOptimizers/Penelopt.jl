using JLD2

using CUTEst, NLPModelsIpopt, SolverBenchmark

include(joinpath(@__DIR__, "utils", "benchmark-utils.jl"))

problem_names = select_benchmark_problems()

# Speedup benchmark time for exact Hessian.
# Split problems across 2 runners.
split = parse(Int, get(ENV, "CUTEST_SPLIT", "1"))
n_splits = 2

@assert 1 <= split <= n_splits

n = length(problem_names)
first = fld((split - 1) * n, n_splits) + 1
last = fld(split * n, n_splits)

problem_names = problem_names[first:last]

@info "Running CUTEst split $split/$n_splits: problems $first:$last ($(length(problem_names)) problems)"

problem_list = (CUTEstModel(name) for name in problem_names)

solvers = Dict(:ipopt_exact => BENCHMARK_SOLVERS[(:ipopt, :exact)])

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_ipopt_exact_$(split).jld2" stats
