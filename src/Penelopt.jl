@doc """
Penelopt.jl: A Large-Scale Equality-Constrained Optimization Solver.

* 📖 Documentation: [https://jso.dev/Penelopt.jl/stable](https://jso.dev/Penelopt.jl/stable)
* 🗂️ Repository: [github.com/JuliaSmoothOptimizers/Penelopt.jl](https://github.com/JuliaSmoothOptimizers/Penelopt.jl)
* 💬 Discussions: [github.com/JuliaSmoothOptimizers/Penelopt.jl/discussions](https://github.com/JuliaSmoothOptimizers/Penelopt.jl/discussions)
* 🎯 Issues: [github.com/JuliaSmoothOptimizers/Penelopt.jl/issues](https://github.com/JuliaSmoothOptimizers/Penelopt.jl/issues)
"""
module Penelopt

# Prefer the sequential, MPI-free build of MUMPS (MUMPS_seq_jll) by default,
# unless the user already requested a custom installation or explicitly set
# MUMPS_SEQ themselves. MUMPS.jl reads these environment variables at module
# *load* (i.e. precompile) time, so this only has an effect the first time
# MUMPS.jl gets precompiled in a given Julia depot/environment.
if !haskey(ENV, "JULIA_MUMPS_LIBRARY_PATH") && !haskey(ENV, "MUMPS_SEQ")
  ENV["MUMPS_SEQ"] = "true"
end

using LinearAlgebra, Printf, SparseArrays
using NLPModels, NLPModelsModifiers
using LinearOperators, QuadraticModels, SolverCore, SparseMatricesCOO
using MPI, MUMPS

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import NLPModelsModifiers: get_model, get_op
import SolverCore: get_status, reset!

function __init__()
  MPI.Init()
  if isdefined(MUMPS, :MUMPS_jll)
    @warn "Penelopt.jl: MUMPS.jl was loaded with its default parallel build, which requires a working MPI installation. Penelopt.jl will still work, but for the best experience we recommend using the sequential MUMPS_seq_jll build instead: set the MUMPS_SEQ environment variable before MUMPS.jl is (re)precompiled."
  end
end

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
