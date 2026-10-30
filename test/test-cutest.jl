problem_names = ["BT1", "MSS1", "SSINE", "VANDANIUMS"]
expected_status = [:first_order, :first_order, :infeasible, :infeasible]

tol = 1e-3

function test_problem(
  name,
  primal_solution,
  dual_solution,
  expected_status;
  linear_solver::String = "mumps",
  ignore_null_hessian::Bool = false,
  ignore_bfgs::Bool = false,
  ignore_exact::Bool = false,
)
  nlp = CUTEstModel(name)

  # Test with R2

  !ignore_null_hessian && @testset "NullHessian" begin
    stats = L2Penalty(
      nlp,
      atol = tol,
      rtol = tol,
      qn_hessian_approximation = "null",
      linear_solver = linear_solver,
      τ0 = 1.0,
    )

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test stats.primal_feas == norm(cons(nlp, stats.solution), Inf)

    # Test stability and allocations
    preprocessed_nlp = nlp
    if length(nlp.meta.ifix) > 0
      preprocessed_nlp = remove_fixed_variables(nlp)
    end
    null_model = NullHessianModel(preprocessed_nlp)

    solver = L2PenaltySolver(null_model, linear_solver = linear_solver)
    stats_optimized = PeneloptExecutionStats(null_model)
    if linear_solver == "ldlt"
      @test @wrappedallocs(
        solve!(solver, null_model, stats_optimized, atol = 1e-3, rtol = 1e-3, τ0 = 1.0)
      ) == 0
    else
      solve!(solver, null_model, stats_optimized, atol = 1e-3, rtol = 1e-3, τ0 = 1.0)
    end

    stats_optimized.solution = recover_full_solution(null_model, stats_optimized.solution)

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
  end

  # Test with BFGS
  !ignore_bfgs && @testset "BFGS" begin
    stats = L2Penalty(
      nlp,
      atol = tol,
      rtol = tol,
      qn_hessian_approximation = "bfgs",
      linear_solver = linear_solver,
      τ0 = 1.0,
    )

    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test abs(stats.primal_feas - norm(cons(nlp, stats.solution), Inf)) ≤ 1000*tol
    @test stats.solver_specific[:n_fact] > 0

    # Test stability and allocations
    preprocessed_nlp = nlp
    if length(nlp.meta.ifix) > 0
      preprocessed_nlp = remove_fixed_variables(nlp)
    end
    LBFGS_model = CompactBFGSModel(preprocessed_nlp)

    solver = L2PenaltySolver(LBFGS_model, linear_solver = linear_solver)
    stats_optimized = PeneloptExecutionStats(LBFGS_model)
    if linear_solver != "mumps"
      @test @wrappedallocs(
        solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3, τ0 = 1.0)
      ) == 0
    else
      solve!(solver, LBFGS_model, stats_optimized, atol = 1e-3, rtol = 1e-3, τ0 = 1.0)
    end

    stats_optimized.solution =
      recover_full_solution(LBFGS_model, stats_optimized.solution)

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
    @test stats_optimized.solver_specific[:n_fact] == stats.solver_specific[:n_fact]
  end

  !ignore_exact && @testset "Exact" begin

    stats = L2Penalty(nlp, atol = tol, rtol = tol; linear_solver = linear_solver, τ0 = 1.0)

    # Test whether the outputs are well defined
    @test stats.status == expected_status
    if expected_status == :first_order
      @test norm(primal_solution - stats.solution) ≤ 100*tol
      @test abs(stats.objective - obj(nlp, primal_solution)) ≤ 100*tol
      @test norm(stats.multipliers - dual_solution) ≤ 100*tol
      @test norm(
        jtprod(nlp, stats.solution, stats.multipliers) + grad(nlp, stats.solution),
        Inf,
      ) ≤ 100*tol
    end
    @test abs(stats.primal_feas - norm(cons(nlp, stats.solution), Inf)) ≤ 1000*tol
    @test stats.solver_specific[:n_fact] > 0

    # Test stability and allocations
    preprocessed_nlp = nlp
    if length(nlp.meta.ifix) > 0
      preprocessed_nlp = remove_fixed_variables(nlp)
    end

    solver = L2PenaltySolver(preprocessed_nlp, linear_solver = linear_solver)
    stats_optimized = PeneloptExecutionStats(preprocessed_nlp)
    if linear_solver == "ldlt"
      @test @wrappedallocs(
        solve!(
          solver,
          preprocessed_nlp,
          stats_optimized,
          atol = 1e-3,
          rtol = 1e-3,
          τ0 = 1.0,
        )
      ) == 0
    else
      solve!(solver, preprocessed_nlp, stats_optimized, atol = 1e-3, rtol = 1e-3, τ0 = 1.0)
    end

    stats_optimized.solution =
      recover_full_solution(preprocessed_nlp, stats_optimized.solution)

    # Test that the second calling form gives the same output
    @test stats_optimized.status == stats.status
    @test stats_optimized.objective == stats.objective
    @test stats_optimized.primal_feas == stats.primal_feas
    @test stats_optimized.dual_feas == stats.dual_feas
    @test all(stats_optimized.multipliers .== stats.multipliers)
    @test all(stats_optimized.solution .== stats.solution)
    @test stats_optimized.iter == stats.iter
    @test stats_optimized.solver_specific[:n_fact] == stats.solver_specific[:n_fact]
  end

  finalize(nlp)
end
# Test a simple problem
@testset "BT1" begin
  primal_solution = [1, 0]
  dual_solution = [-99.5]
  linear_solver =
    !isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt)) ? "ldlt" :
    "mumps"
  test_problem(
    "BT1",
    primal_solution,
    dual_solution,
    :first_order;
    linear_solver = linear_solver,
  )
end

# Test a problem where the function f is unbounded from below
@testset "MARATOS" begin
  primal_solution = [1, 0]
  dual_solution = [0.499999]
  linear_solver =
    !isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt)) ? "ldlt" :
    "mumps"
  test_problem(
    "MARATOS",
    primal_solution,
    dual_solution,
    :first_order;
    linear_solver = linear_solver,
  )
end

# Test an infeasible problem
@testset "VANDANIUMS" begin
  if isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt))
    test_problem("VANDANIUMS", Float64[], Float64[], :infeasible)
  end
end

# Test a problem that requires the watchdog technique
@testset "SSINE" begin
  # The problem is infeasible but the primal feas is arbitrarily small for some ||x|| -> inf.
  # Only test that we converge to first order
  nlp = CUTEstModel("SSINE")
  stats = L2Penalty(nlp, atol = 1e-3, rtol = 0.0)
  if !Sys.iswindows()
    @test stats.status == :first_order
  end

  # stats = L2Penalty(nlp, atol = 1e-3, rtol = 0.0, qn_hessian_approximation = "bfgs")
  # @test stats.status == :first_order
  finalize(nlp)
end

# Test a problem with fixed variables
@testset "AIRCRFTA" begin
  primal_solution = [
    0.005652720539657081,
    -0.00653774373397042,
    -0.0006212466680082355,
    -0.12333555374458006,
    -0.0003874221934374081,
    0.1,
    0.0,
    0.0,
  ]
  dual_solution = zeros(5)
  linear_solver =
    !isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt)) ? "ldlt" :
    "mumps"
  test_problem(
    "AIRCRFTA",
    primal_solution,
    dual_solution,
    :first_order;
    linear_solver = linear_solver,
  )
end

# Test a problem s.t f(x) + tau ||c(x)|| is unbounded for a small tau.
@testset "HS56" begin
  primal_solution = [
    2.4,
    1.2,
    1.2,
    0.857071947850131,
    0.5639426413606289,
    0.5639426413606289,
    1.5707963267948966,
  ]
  dual_solution = [0.0, 0.0, 0.0, 1.44]
  linear_solver =
    !isnothing(Base.get_extension(Penelopt, :PeneloptLDLFactorizationsExt)) ? "ldlt" :
    "mumps"
  test_problem(
    "HS56",
    primal_solution,
    dual_solution,
    :first_order;
    linear_solver = linear_solver,
    ignore_null_hessian = true,
  )
end
