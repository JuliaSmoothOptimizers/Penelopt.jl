export EnveloptL2PenaltySolver

# Workspace of the exact ℓ₂-penalty method with a nonsmooth regularizer, whose penalized
# subproblems are solved by Envelopt. Constructors and `solve!` are defined in the
# PeneloptEnveloptExt extension.
mutable struct EnveloptL2PenaltySolver{T<:Real,V<:AbstractVector{T},E,S} <:
               AbstractL2PenaltySolver{T,V}
  x::V          # current outer iterate
  c::V          # c(x)
  y_env::V      # multiplier estimate of c(x) - u = 0
  env_model::E  # f(x) + e_{μ τ‖⋅‖₂}(c(x) + μy);
  subsolver::S
end

function EnveloptL2PenaltySolver(::AbstractNLPModel; kwargs...)
  error(
    "EnveloptL2PenaltySolver requires loading Envelopt.jl and RegularizedOptimization.jl.",
  )
end
