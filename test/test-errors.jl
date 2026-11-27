# Test warnings, errors, and exceptions thrown by Penelopt.


@testset "Extensions not loaded" begin

  nlp = CUTEstModel("BT1")

  @test_warn "Penelopt.jl: LDLFactorizations extension is not loaded. Please install LDLFactorizations.jl. Switching to MUMPS..." begin
    L2Penalty(nlp, linear_solver = :ldlt)
  end
  @test_warn "Penelopt.jl: HSL extension is not loaded. Please install HSL.jl. Switching to MUMPS..." begin
    L2Penalty(nlp, linear_solver = :ma57)
  end

  finalize(nlp)
end

@testset "Unknown options" begin

  nlp = CUTEstModel("BT1")

  @test_warn "Penelopt.jl: unknown linear solver `:foo`. Supported values are :mumps, :ldlt and :ma57. Switching to MUMPS..." begin
    L2Penalty(nlp, linear_solver = :foo)
  end
  @test_warn "Penelopt.jl: unknown Hessian approximation `:foo`. Supported values are :exact, :bfgs and :null. Switching to :exact..." begin
    L2Penalty(nlp, qn_hessian_approximation = :foo)
  end

  # Options are Symbols, not Strings.
  @test_throws TypeError L2Penalty(nlp, linear_solver = "mumps")
  @test_throws TypeError L2Penalty(nlp, qn_hessian_approximation = "bfgs")

  finalize(nlp)
end

@testset "Wrong problem type" begin
  # Problem with fixed variables
  nlp = CUTEstModel("AIRCRFTA")
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: The problem has fixed variables. Refer to the documentation for information on how to preprocess the problem." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)
end
