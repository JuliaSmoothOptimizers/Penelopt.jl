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
