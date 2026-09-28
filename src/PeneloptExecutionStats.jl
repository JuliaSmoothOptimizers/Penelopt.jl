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
