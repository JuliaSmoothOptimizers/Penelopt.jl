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

  # Unconstrained problem
  nlp = CUTEstModel("3PK")
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." L2Penalty(
    nlp,
  )
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)

  # Problem with bounds 
  nlp = CUTEstModel("LIN")
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." L2Penalty(
    nlp,
  )
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)

  # Problem with one-sided bounds: must be rejected before any preprocessing
  for (lvar, uvar) in (([0.0, -Inf], [Inf, Inf]), ([-Inf, -Inf], [Inf, 1.0]))
    nlp = ADNLPModel(
      x -> (x[1] - 1)^2 + x[2]^2,
      [2.0, 2.0],
      lvar,
      uvar,
      x -> [x[1] + x[2] - 1],
      [0.0],
      [0.0],
    )
    @test_throws "L2Penalty: This algorithm only works for equality constrained problems." L2Penalty(
      nlp,
    )
  end

  # Problem with inequalities
  nlp = CUTEstModel("AVGASA")
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." L2Penalty(
    nlp,
  )
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality constrained problems." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)

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
