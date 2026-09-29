using JLD2

using CUTEst, NLPModelsIpopt, SolverBenchmark

include(joinpath(@__DIR__, "utils", "benchmark-utils.jl"))

problem_names = select_benchmark_problems()

filter!(x -> x != "FLOSP2TL", problem_names) # This problem of the form min 0 st c(x) = 0 causes the benchmarks to timeout...

# Speedup benchmark time for BFGS
# Split problems across 4 runners.
split = parse(Int, get(ENV, "CUTEST_SPLIT", "1"))
n_splits = 4

@assert 1 <= split <= n_splits

n = length(problem_names)
first = fld((split - 1) * n, n_splits) + 1
last = fld(split * n, n_splits)

problem_names = problem_names[first:last]

@info "Running CUTEst split $split/$n_splits: problems $first:$last ($(length(problem_names)) problems)"

problem_list = (CUTEstModel(name) for name in problem_names)

solvers = Dict(:ipopt_lbfgs => BENCHMARK_SOLVERS[(:ipopt, :lbfgs)])

stats = bmark_solvers(solvers, problem_list)
@save "benchmark/result/stats_ipopt_lbfgs_$(split).jld2" stats
