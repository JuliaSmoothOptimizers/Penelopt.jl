export PeneloptExecutionStats

"""
    stats = PeneloptExecutionStats(nlp)

Allocate a `GenericExecutionStats` object (see
[SolverCore.jl](https://github.com/JuliaSmoothOptimizers/SolverCore.jl)) for `nlp`
with the `solver_specific` entries used by Penelopt.jl.

Pass it to [`solve!`](@ref) together with an [`L2PenaltySolver`](@ref); it is filled in
place. See [Outputs](@ref) for a description of its fields.
"""
function PeneloptExecutionStats(nlp::AbstractNLPModel{T,V}) where {T,V}
  stats = GenericExecutionStats(nlp, solver_specific = Dict{Symbol,Int}())
  set_solver_specific!(stats, :n_fact, T(0))
  return stats
end

"""
    stats = PenaltyR2NExecutionStats(penalty_nlp)

Allocate the execution statistics of a `PenaltyR2NSolver` for `penalty_nlp`.
This wraps a `GenericExecutionStats` whose `solver_specific` entries used by the outer loop
are guaranteed to exist. All `GenericExecutionStats` fields and setters are forwarded.
"""
struct PenaltyR2NExecutionStats{T,V} <: AbstractExecutionStats
  stats::GenericExecutionStats{T,V,V,T}
end

function PenaltyR2NExecutionStats(penalty_nlp::AbstractNLPModel{T,V}) where {T,V}
  stats = GenericExecutionStats(penalty_nlp, solver_specific = Dict{Symbol,T}())
  for key in
      (:primal_ktol, :dual_ktol, :n_fact, :tau, :sigma, :rho, :smooth_obj, :nonsmooth_obj)
    set_solver_specific!(stats, key, T(0))
  end
  return PenaltyR2NExecutionStats{T,V}(stats)
end

@inline Base.getproperty(s::PenaltyR2NExecutionStats, name::Symbol) =
  name === :stats ? getfield(s, :stats) : getproperty(getfield(s, :stats), name)

@inline Base.setproperty!(s::PenaltyR2NExecutionStats, name::Symbol, value) =
  setproperty!(getfield(s, :stats), name, value)

Base.propertynames(s::PenaltyR2NExecutionStats) = propertynames(getfield(s, :stats))

for f in (
  :set_status!,
  :set_solution!,
  :set_objective!,
  :set_residuals!,
  :set_primal_residual!,
  :set_dual_residual!,
  :set_multipliers!,
  :set_constraint_multipliers!,
  :set_bounds_multipliers!,
  :set_iter!,
  :set_step_status!,
  :set_time!,
  :set_solver_specific!,
  :reset!,
)
  @eval function SolverCore.$f(s::PenaltyR2NExecutionStats, args...)
    SolverCore.$f(getfield(s, :stats), args...)
    return s
  end
end

"""
    get_smooth_obj(stats::PenaltyR2NExecutionStats)

Return f(x) at the last solution of the subsolver.
"""
get_smooth_obj(stats::PenaltyR2NExecutionStats) = stats.solver_specific[:smooth_obj]

"""
    get_nonsmooth_obj(stats::PenaltyR2NExecutionStats)

Return τ‖c(x)‖₂ at the last solution of the subsolver.
"""
get_nonsmooth_obj(stats::PenaltyR2NExecutionStats) = stats.solver_specific[:nonsmooth_obj]

"""
    get_sigma(stats::PenaltyR2NExecutionStats)

Return the regularization parameter at the last solution of the subsolver.
It is used to warm-start the next subsolve.
"""
get_sigma(stats::PenaltyR2NExecutionStats) = stats.solver_specific[:sigma]

"""
    get_n_fact(stats::PenaltyR2NExecutionStats)

Return the number of factorizations performed by the subsolver.
"""
get_n_fact(stats::PenaltyR2NExecutionStats) = stats.solver_specific[:n_fact]
