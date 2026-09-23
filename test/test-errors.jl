# Test warnings, errors, and exceptions thrown by Penelopt.


@testset "Extensions not loaded" begin

  nlp = CUTEstModel("BT1")

  @test_warn "Penelopt.jl: LDLFactorizations extension is not loaded. Please install LDLFactorizations.jl. Switching to MUMPS..." begin
    L2Penalty(nlp, linear_solver = "ldlt")
  end
  @test_warn "Penelopt.jl: HSL extension is not loaded. Please install HSL.jl. Switching to MUMPS..." begin
    L2Penalty(nlp, linear_solver = "ma57")
  end

  finalize(nlp)
end

@testset "Wrong problem type" begin

  # Unconstrained problem
  nlp = CUTEstModel("3PK")
  @test_throws "L2Penalty: This algorithm only works for equality contrained problems." L2Penalty(
    nlp,
  )
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality contrained problems." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)

  # Problem with bounds and an initial point that is not strictly interior
  nlp = CUTEstModel("LIN")
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality contrained problems." solve!(
    solver,
    nlp,
    stats,
  )
  finalize(nlp)

  # Problem with inequalities
  nlp = CUTEstModel("AVGASA")
  @test_throws "L2Penalty: This algorithm only works for equality contrained problems." L2Penalty(
    nlp,
  )
  solver = L2PenaltySolver(nlp)
  stats = PeneloptExecutionStats(nlp)
  @test_throws "L2Penalty: This algorithm only works for equality contrained problems." solve!(
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
