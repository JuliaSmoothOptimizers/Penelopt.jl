using DataFrames
using JLD2
using SolverBenchmark
using Plots

function pairwise_plot(
  stats,
  keys;
  compare_n_fact = false,
)
  df_1 = stats[keys[1]]
  df_2 = stats[keys[2]]

  solved(df) = (df.status .== :first_order .|| df.status .== :certified_infeasible)
  costs = [
    df -> .!solved(df) * Inf + df.elapsed_time,
    df -> .!solved(df) * Inf + df.neval_obj,
    df -> .!solved(df) * Inf + df.neval_grad,
  ]
  costnames = ["CPU Time", "# Objective Evals", "# Gradient Evals"]

  if compare_n_fact &&
     hasproperty(stats[keys[1]], :n_fact) &&
     hasproperty(stats[keys[2]], :n_fact)
    costnames[1] = "# Factorizations"
    costs[1] = df -> .!solved(df) * Inf .+ coalesce.(df.n_fact, Inf)
  end

  stats_subset = filter(kv -> kv[1] in keys, stats)

  # split the stat names. the first part is always l2penalty, then the method used, then the tolerance and finally the branch.
  parts_1 = Symbol.(split(string(keys[1]), "_"))
  parts_2 = Symbol.(split(string(keys[2]), "_"))

  models = Dict(:exact => "∇²L(x, y)", :lbfgs => "BFGS")
  precision = Dict(:imprecise => "1e-3", :precise => "1e-9")

  suptitle = "\nHessian model: Bₖ(x) = " * models[parts_1[2]]
  p = profile_solvers(
    stats_subset,
    costs,
    costnames;
    suptitle = suptitle,
    xlabel = "",
    ylabel = "",
  )
  p.subplots[2][:legend_position] = :bottomright
  p.subplots[3][:legend_position] = :bottomright

  compare_with = (parts_1[end] == :reference || parts_2[end] == :reference) ? :main : :ipopt
  p.series_list[1][:label] =
    Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[2][:label] =
    Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[3][:label] =
    Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[4][:label] =
    Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[5][:label] =
    Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :current :
    compare_with
  p.series_list[6][:label] =
    Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :current :
    compare_with

  p.series_list[1][:linecolor] =
    Symbol.(split(string(p.series_list[1][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[2][:linecolor] =
    Symbol.(split(string(p.series_list[2][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[3][:linecolor] =
    Symbol.(split(string(p.series_list[3][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[4][:linecolor] =
    Symbol.(split(string(p.series_list[4][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[5][:linecolor] =
    Symbol.(split(string(p.series_list[5][:label]), "_"))[end] == :current ? :blue : :red
  p.series_list[6][:linecolor] =
    Symbol.(split(string(p.series_list[6][:label]), "_"))[end] == :current ? :blue : :red

  return p
end