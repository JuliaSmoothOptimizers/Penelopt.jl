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

include("Envelopt/EnveloptSolver.jl")

end
