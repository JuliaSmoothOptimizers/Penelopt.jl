module PeneloptEnveloptExt

using LinearAlgebra, Printf
using NLPModels, SolverCore
using Penelopt
using Penelopt: NormL2
using Envelopt:
  Envelopt,
  EnveloptNLPModel,
  EqualityFeasibilityModel,
  EqualityLessModel,
  R2NEnveloptSubSolver,
  envelopt
using RegularizedProblems: AbstractRegularizedNLPModel, RegularizedNLPModel

using ProximalCore

Penelopt.AbstractPenalizedProblem{T, V} <: AbstractRegularizedNLPModel{T, V} where {T, V}

include("Envelopt/EnveloptFModel.jl")
include("Envelopt/EnveloptSolver.jl")

end
