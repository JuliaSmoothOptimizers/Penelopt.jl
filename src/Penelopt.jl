@doc """
Penelopt.jl: A Large-Scale Equality-Constrained Optimization Solver.

* 📖 Documentation: [https://jso.dev/Penelopt.jl/stable](https://jso.dev/Penelopt.jl/stable)
* 🗂️ Repository: [github.com/JuliaSmoothOptimizers/Penelopt.jl](https://github.com/JuliaSmoothOptimizers/Penelopt.jl)
* 💬 Discussions: [github.com/JuliaSmoothOptimizers/Penelopt.jl/discussions](https://github.com/JuliaSmoothOptimizers/Penelopt.jl/discussions)
* 🎯 Issues: [github.com/JuliaSmoothOptimizers/Penelopt.jl/issues](https://github.com/JuliaSmoothOptimizers/Penelopt.jl/issues)
"""
module Penelopt

using LinearAlgebra, Printf, SparseArrays
using NLPModels, NLPModelsModifiers
using LinearOperators, QuadraticModels, SolverCore, SparseMatricesCOO
using MUMPS

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import NLPModelsModifiers: get_model, get_op
import SolverCore: get_status, reset!

abstract type AbstractPenalizedProblemSolver <: AbstractOptimizationSolver end

include("PeneloptExecutionStats.jl")

include("types/quasi-newton/NullHessian.jl")
include("types/quasi-newton/CompactBFGS.jl")

include("types/norm/NormL2.jl")
include("types/norm/CompositeNormL2.jl")
include("types/norm/ShiftedCompositeNormL2.jl")

include("types/pre-processing/utils.jl")
include("types/pre-processing/FixedVariable.jl")
include("types/pre-processing/Scaling.jl")
include("types/pre-processing/ShiftedConstraint.jl")

include("linear_algebra/K2.jl")
include("linear_algebra/construct_workspace.jl")
include("linear_algebra/mumps.jl")
include("linear_algebra/lapack.jl")

include("types/PenalizedProblem.jl")
include("types/ShiftedPenalizedProblem.jl")
include("types/Watchdog.jl")

include("subsolvers/more-sorensen.jl")

include("ir2n.jl")

include("algorithm.jl")

include("extrapolate.jl")
include("feas_computer.jl")
end
