# Get instances 
instances = "instances/"
instances =
  filter(f -> isfile(f) && endswith(f, ".txt"), joinpath.(instances, readdir(instances)))
# Get subsolvers
solver_names = ["MoreSorensenSolver"]
solvers = [MoreSorensenSolver]

# Test on real instances
for (solver_name, solver_constructor) in zip(solver_names, solvers)
  @testset "$solver_name" begin
    @testset "Well-conditionned" begin
      n, m = 10, 2
      small_instance_boundary, solution =
        generate_instance(n, m, 0.5, Hessian_modifier = x -> SparseMatrixCOO(tril(x)))
      solver = eval(solver_constructor)(small_instance_boundary; solver = :ldlt)
      stats = GenericExecutionStats(
        small_instance_boundary.model;
        solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
      )
      if VERSION >= v"1.12"
        @test @wrappedallocs(
          solve!(
            solver,
            small_instance_boundary,
            stats,
            atol = 1e-9,
            accept_descent = false,
          )
        ) == 0
      else
        solve!(solver, small_instance_boundary, stats, atol = 1e-9, accept_descent = false)
      end
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[(n+1):end]) <= 1e-6
      @test abs(solution[:tau] - norm(solver.x1[(n+1):end])) <= 1e-6

      small_instance_interior, solution =
        generate_instance(n, m, 0.0, Hessian_modifier = x -> SparseMatrixCOO(tril(x)))
      solver = eval(solver_constructor)(small_instance_interior; solver = :ldlt)
      stats = GenericExecutionStats(
        small_instance_interior.model;
        solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
      )
      if VERSION >= v"1.12"
        @test @wrappedallocs(
          solve!(
            solver,
            small_instance_interior,
            stats,
            atol = 1e-9,
            accept_descent = false,
          )
        ) == 0
      else
        solve!(solver, small_instance_interior, stats, atol = 1e-9, accept_descent = false)
      end
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[(n+1):end]) <= 1e-6
      @test norm(solver.x1[(n+1):end]) <= solution[:tau]

      n, m = 100, 20
      medium_instance_boundary, solution =
        generate_instance(n, m, 0.5, Hessian_modifier = x -> SparseMatrixCOO(tril(x)))
      solver = eval(solver_constructor)(medium_instance_boundary; solver = :ldlt)
      stats = GenericExecutionStats(
        medium_instance_boundary.model;
        solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
      )
      if VERSION >= v"1.12"
        @test @wrappedallocs(
          solve!(
            solver,
            medium_instance_boundary,
            stats,
            atol = 1e-9,
            accept_descent = false,
          )
        ) == 0
      else
        solve!(solver, medium_instance_boundary, stats, atol = 1e-9, accept_descent = false)
      end
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[(n+1):end]) <= 1e-6
      @test abs(solution[:tau] - norm(solver.x1[(n+1):end])) <= 1e-6

      medium_instance_interior, solution =
        generate_instance(n, m, 0.0, Hessian_modifier = x -> SparseMatrixCOO(tril(x)))
      solver = eval(solver_constructor)(medium_instance_interior; solver = :ldlt)
      stats = GenericExecutionStats(
        medium_instance_interior.model;
        solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
      )
      if VERSION >= v"1.12"
        @test @wrappedallocs(
          solve!(
            solver,
            medium_instance_interior,
            stats,
            atol = 1e-9,
            accept_descent = false,
          )
        ) == 0
      else
        solve!(solver, medium_instance_interior, stats, atol = 1e-9, accept_descent = false)
      end
      @test norm(solution[:u] - stats.solution) <= 1e-6
      @test norm(solution[:y] - solver.x1[(n+1):end]) <= 1e-6
      @test norm(solver.x1[(n+1):end]) <= solution[:tau]
    end

    @testset "Ill-conditionned" begin
      for instance in instances
        reg_nlp =
          read_instance(instance, type = Float64, Hessian_modifier = x -> SparseMatrixCOO(tril(x)))
        n = reg_nlp.model.meta.nvar
        solver = eval(solver_constructor)(reg_nlp; solver = :ldlt)
        stats = GenericExecutionStats(
          reg_nlp.model;
          solver_specific = Dict{Symbol,Float64}(:alpha => 0.0),
        )
        if VERSION >= v"1.12"
          @test @wrappedallocs(solve!(solver, reg_nlp, stats, accept_descent = false)) == 0
        else
          solve!(solver, reg_nlp, stats, accept_descent = false)
        end
        instance_name = basename(instance)
        if occursin("boundary", instance_name)
          @test abs(norm(solver.x1[(n+1):end]) - reg_nlp.h.h.lambda) <= 1e-3
        end
      end
    end
  end
end
