using Penelopt
using ADNLPModels,
  CUTEst,
  LinearOperators,
  NLPModels,
  NLPModelsModifiers,
  NLPModelsTest,
  QuadraticModels,
  SolverCore,
  SparseMatricesCOO

using LinearAlgebra, Random, SparseArrays, Test

import Penelopt:
  solve!, ShiftedCompositeNormL2, ShiftedL2PenalizedProblem, MoreSorensenSolver

Random.seed!(0)

include("allocations-macro.jl")

include("instances/instance-reader.jl")
include("instances/instance-generator.jl")

@testset "pre-processing" begin
  include("test-no-fixed-vars.jl")
  include("test-no-shifted-constraints.jl")
  include("test-preprocessing-order.jl")
  include("test-scaling.jl")
end

@testset "quasi-Newton" begin
  include("test-quasi-newton.jl")
end

@testset "CUTEst-default" begin
  @test isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt)) # Check that the extension is not loaded.
  include("test-cutest.jl")
end

@testset "Errors and warnings" begin
  include("test-errors.jl")
end

# The Subsolvers allocation tests specifically exercise the zero-allocation
# LDLFactorizations.jl path, so the extension needs to be loaded from here on.
using LDLFactorizations

@testset "Subsolvers" begin
  include("test-subsolvers.jl")
end

@testset "CUTEst-LDLFactorizations" begin
  @test !isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt))
  include("test-cutest.jl")
end

# HSL.jl is always installed for the tests, but MA57 is only functional when a
# licensed HSL_jll.jl is available (see .github/workflows/TestHSL.yml). On the
# GitHub-hosted runners, only the fallback path is exercised.
using HSL

@testset "HSL" begin
  @test !isnothing(Base.get_extension(Penelopt, :PeneloptHSLExt))
  include("test-hsl.jl")
end
