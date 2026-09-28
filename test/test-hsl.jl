# Tests for the MA57 linear solver provided by PeneloptHSLExt.
#
# On the licensed runner, the workflow sets PENELOPT_TEST_HSL=true so that a
# non-functional HSL_jll.jl makes the test suite fail instead of silently
# skipping every MA57 test.

const REQUIRE_HSL = lowercase(get(ENV, "PENELOPT_TEST_HSL", "false")) == "true"

@test LIBHSL_isfunctional() || !REQUIRE_HSL
@test Penelopt.hsl_functional() == LIBHSL_isfunctional()

if !LIBHSL_isfunctional()
  @info "HSL_jll.jl is not functional: only testing the fallback to MUMPS."

  @testset "Fallback to MUMPS" begin
    nlp = CUTEstModel("BT1")
    @test_warn "Penelopt.jl: HSL extension is not functional." begin
      L2Penalty(nlp, linear_solver = "ma57")
    end
    finalize(nlp)
  end
else
  @testset "MoreSorensenSolver (MA57)" begin
    for (n, m, alpha) in ((10, 2, 0.5), (10, 2, 0.0), (100, 20, 0.5), (100, 20, 0.0))
      instance, solution =
        generate_instance(n, m, alpha, Hessian_modifier = x -> sparse(tril(x)))
      solver = MoreSorensenSolver(instance; solver = :ma57)
      stats = GenericExecutionStats(
        instance.model;
        solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
      )
      solve!(solver, instance, stats, atol = 1e-9, accept_descent = false)
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[(n+1):end]) <= 1e-6
      if alpha > 0
        @test abs(solution[:tau] - norm(solver.x1[(n+1):end])) <= 1e-6
      else
        @test norm(solver.x1[(n+1):end]) <= solution[:tau]
      end
    end
  end

  # `test_problem` is defined in test-cutest.jl.
  @testset "CUTEst (MA57)" begin
    @testset "BT1" begin
      test_problem("BT1", [1, 0], [-99.5], :first_order; linear_solver = "ma57")
    end

    @testset "MARATOS" begin
      test_problem("MARATOS", [1, 0], [0.499999], :first_order; linear_solver = "ma57")
    end

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
      test_problem(
        "AIRCRFTA",
        primal_solution,
        zeros(5),
        :first_order;
        linear_solver = "ma57",
      )
    end

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
      test_problem(
        "HS56",
        primal_solution,
        [0.0, 0.0, 0.0, 1.44],
        :first_order;
        linear_solver = "ma57",
        ignore_null_hessian = true,
      )
    end
  end
end
