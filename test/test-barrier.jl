@testset "LogBarrier" begin
  μ = 0.1
  l = [-Inf, 0.0, 0.1, -Inf]
  u = [1.0, Inf, 2.0, Inf]
  ϕ = Penelopt.LogBarrier(μ, l, u)

  @test ϕ.μ == μ && ϕ.τ == 0.99
  @test_throws AssertionError Penelopt.LogBarrier(-1.0, l, u)
  @test_throws AssertionError Penelopt.LogBarrier(1.0, u, l)

  @testset "push_to_interior!" begin
    # upper only, lower only, two-sided (clamped at u - p_u), free
    x = [5.0, -3.0, 5.0, 7.0]
    Penelopt.push_to_interior!(ϕ, x)
    @test x ≈ [0.99, 0.01, 2.0 - 0.019, 7.0]
    @test Penelopt.isinterior(ϕ, x)

    # two-sided, clamped at l + p_l
    x = [0.5, 0.5, 0.0, 0.0]
    Penelopt.push_to_interior!(ϕ, x)
    @test x[3] ≈ 0.1 + 0.01

    @test_throws AssertionError Penelopt.push_to_interior!(ϕ, x; κ2 = 0.6)
    @test Penelopt.push_to_interior!(nothing, x) === x
    @test !Penelopt.isinterior(ϕ, [1.0, 0.5, 1.0, 0.0])
  end

  x = [0.5, 0.5, 1.0, 0.0]
  β(x) = -(log(1 - x[1]) + log(x[2]) + log(x[3] - 0.1) + log(2 - x[3]))
  ∇β(x) = [1 / (1 - x[1]), -1 / x[2], -1 / (x[3] - 0.1) + 1 / (2 - x[3]), 0.0]

  @testset "value, gradient and Hessian" begin
    @test ϕ(x) ≈ μ * β(x)
    @test ϕ([1.5, 0.5, 1.0, 0.0]) == Inf

    g = ones(4)
    Penelopt.add_grad!(g, ϕ, x)
    @test g ≈ 1 .+ μ .* ∇β(x)

    h = zeros(4)
    Penelopt.hess_diag!(h, ϕ, x, 2.0)
    @test h ≈ 2μ .* [1 / 0.5^2, 1 / 0.5^2, 1 / 0.9^2 + 1 / 1.0^2, 0.0]
  end

  @testset "implicit bound multipliers" begin
    z_l, z_u = fill(NaN, 4), fill(NaN, 4)
    Penelopt.bound_multipliers!(z_l, z_u, ϕ, x)
    # slacks are x - l = (0.5, 0.9) and u - x = (0.5, 1.0)
    @test z_l ≈ [0.0, μ / 0.5, μ / 0.9, 0.0] && z_u ≈ [μ / 0.5, 0.0, μ / 1.0, 0.0]
    # ∇ϕ(x) = -z_l + z_u
    @test Penelopt.add_grad!(zeros(4), ϕ, x) ≈ z_u .- z_l
    # perturbed complementarity holds exactly on finite bounds
    @test z_l[2:3] .* (x[2:3] .- l[2:3]) ≈ [μ, μ]
    @test z_u[[1, 3]] .* (u[[1, 3]] .- x[[1, 3]]) ≈ [μ, μ]
    @test Penelopt.bound_multipliers!(z_l, z_u, nothing, x) == (zeros(4), zeros(4))

    @test Penelopt.get_barrier_parameter(ϕ) == μ
    @test Penelopt.get_barrier_parameter(nothing) == 0
    @test Penelopt.compute_barrier_ktol(ϕ, 10.0) ≈ 10μ
    @test Penelopt.compute_barrier_ktol(nothing, 10.0) == 0
  end

  @testset "fraction to the boundary" begin
    # x₁ → u₁ and x₂ → l₂ both give α = τ * 0.5.
    s = [1.0, -1.0, 0.0, 10.0]
    α = Penelopt.truncate_to_boundary!(s, x, ϕ)
    @test α ≈ 0.99 * 0.5
    @test s ≈ α .* [1.0, -1.0, 0.0, 10.0]
    @test Penelopt.isinterior(ϕ, x .+ s)

    # A step moving away from every bound is not truncated.
    s = [-1.0, 1.0, 0.0, 10.0]
    @test Penelopt.truncate_to_boundary!(s, x, ϕ) == 1.0
    @test s == [-1.0, 1.0, 0.0, 10.0]
    @test Penelopt.truncate_to_boundary!(s, x, nothing) == 1.0
  end

  @testset "barrier parameter updates" begin
    ψ = Penelopt.LogBarrier(0.1, l, u)
    @test Penelopt.update_barrier!(ψ, x, 1e-8) ≈ 0.02          # κμ μ
    @test Penelopt.update_barrier!(ψ, x, 1e-8) ≈ 0.02^1.5      # μ^θμ
    @test ψ.τ ≈ 1 - 0.02^1.5                                   # τ = max(τmin, 1 - μ)
    @test Penelopt.update_barrier!(ψ, x, 1.0) ≈ 0.1           # floored at tol / 10
    @test_throws AssertionError Penelopt.update_barrier!(ψ, x, 1e-8; κμ = 1.5)
    @test isnothing(Penelopt.update_barrier!(nothing, x, 1e-8))

    set_barrier!(ψ, 1e-3)
    @test ψ.μ == 1e-3 && ψ.τ == 1 - 1e-3
    @test isnothing(set_barrier!(nothing))
    @test isnothing(Penelopt.set_fraction_to_boundary!(nothing))
  end
end

@testset "LogBarrierModel" begin
  μ = 0.1
  l = [-Inf, 0.0, 0.1]
  u = [1.0, Inf, 2.0]
  x = [0.2, 0.5, 0.7]
  y = [0.3]

  f(x) = sum(abs2, x)
  c(x) = [sum(x .^ 3) - 1]
  ϕ(x) = -μ * (log(1 - x[1]) + log(x[2]) + log(x[3] - 0.1) + log(2 - x[3]))

  nlp = ADNLPModel(f, x, l, u, c, [0.0], [0.0])
  ref = ADNLPModel(x -> f(x) + ϕ(x), x, c, [0.0], [0.0])
  bnlp = LogBarrierModel(nlp; μ)

  # The barrier is primal: all derivatives match those of f + ϕ exactly.
  @test all(==(-Inf), bnlp.meta.lvar) && all(==(Inf), bnlp.meta.uvar)
  @test get_nnzh(bnlp) == get_nnzh(nlp) + 3
  @test obj(bnlp, x) ≈ obj(ref, x)
  @test grad(bnlp, x) ≈ grad(ref, x)
  @test cons(bnlp, x) ≈ cons(ref, x)
  @test Matrix(jac(bnlp, x)) ≈ Matrix(jac(ref, x))
  @test Matrix(hess(bnlp, x)) ≈ Matrix(hess(ref, x))
  @test Matrix(hess(bnlp, x, y)) ≈ Matrix(hess(ref, x, y))
  @test Matrix(hess(bnlp, x, y, obj_weight = 2.0)) ≈ Matrix(hess(ref, x, y, obj_weight = 2.0))
  @test obj(bnlp, [1.5, 0.5, 0.7]) == Inf

  @test Penelopt.get_model(bnlp) === nlp
  @test Penelopt.get_barrier(bnlp) === bnlp.ϕ
  @test isnothing(Penelopt.get_barrier(nothing))
  @test add_log_barrier(bnlp; μ) === bnlp       # no bounds left to remove
  @test add_log_barrier(nlp; μ) isa LogBarrierModel
  @test add_log_barrier(ref) === ref

  set_barrier!(bnlp.ϕ, 0.5)
  @test bnlp.ϕ.μ == 0.5

  pb = BarrierPenalizedProblem(nlp; μ)
  @test pb isa Penelopt.L2PenalizedProblem
  set_barrier!(pb.model.ϕ, 1.0)
  @test pb.model.ϕ.μ == 1.0
end