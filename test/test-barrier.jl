@testset "LogBarrier" begin
  μ = 0.1
  l = [-Inf, 0.0, 0.1, -Inf]
  u = [1.0, Inf, 2.0, Inf]
  ϕ = Penelopt.LogBarrier(μ, l, u)

  @test ϕ.μ == μ && ϕ.τ == 0.99
  @test_throws AssertionError Penelopt.LogBarrier(-1.0, l, u)
  @test_throws AssertionError Penelopt.LogBarrier(1.0, u, l)

  Penelopt.initialize_multipliers!(ϕ)
  @test ϕ.z_l == [0.0, 1.0, 1.0, 0.0]
  @test ϕ.z_u == [1.0, 0.0, 1.0, 0.0]
  @test isnothing(Penelopt.initialize_multipliers!(nothing))

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
    @test h ≈ 2 .* [1 / 0.5, 1 / 0.5, 1 / 0.9 + 1 / 1.0, 0.0]
  end

  @testset "complementarity" begin
    res_l, res_u = zeros(4), zeros(4)
    # z = 1 on finite bounds; slacks are x - l = (0.5, 0.9) and u - x = (0.5, 1.0)
    @test Penelopt.compute_compl_error!(res_l, res_u, ϕ, x) ≈ 1.0
    @test res_l ≈ [0.0, 0.5, 0.9, 0.0] && res_u ≈ [0.5, 0.0, 1.0, 0.0]
    @test Penelopt.compute_mu_compl_error!(res_l, res_u, ϕ, x) ≈ 0.9
    @test res_l ≈ [0.0, 0.4, 0.8, 0.0] && res_u ≈ [0.4, 0.0, 0.9, 0.0]
    @test !any(isnan, res_l) && !any(isnan, res_u) # infinite bounds do not produce NaN

    @test Penelopt.compute_compl_error!(res_l, res_u, nothing, x) == 0
    @test Penelopt.compute_mu_compl_error!(res_l, res_u, nothing, x) == 0
    @test Penelopt.compute_compl_ktol(ϕ, 10.0) ≈ 10μ
    @test Penelopt.compute_compl_ktol(nothing, 10.0) == 0
  end

  @testset "multiplier steps and fraction to the boundary" begin
    s = [1.0, -1.0, 0.5, 10.0]
    s_z_l, s_z_u = zeros(4), zeros(4)

    Penelopt.get_z_l_step!(s_z_l, ϕ, x, s)
    Penelopt.get_z_u_step!(s_z_u, ϕ, x, s)
    @test s_z_l ≈ [0.0, μ / 0.5 - 1 + 1 / 0.5, μ / 0.9 - 1 - 0.5 / 0.9, 0.0]
    @test s_z_u ≈ [μ / 0.5 - 1 + 1 / 0.5, 0.0, μ / 1.0 - 1 + 0.5 / 1.0, 0.0]
    @test isnothing(Penelopt.get_z_l_step!(s_z_l, nothing, x, s))
    @test isnothing(Penelopt.get_z_u_step!(s_z_u, nothing, x, s))

    # Primal: x₁ → u₁ and x₂ → l₂ both give α = τ * 0.5; dual: z_l₂ decreases fastest.
    s = [1.0, -1.0, 0.0, 10.0]
    s_z_l, s_z_u = [0.0, -4.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]
    α, α_z = Penelopt.truncate_to_boundary!(s, s_z_l, s_z_u, x, ϕ)
    @test α ≈ 0.99 * 0.5 && α_z ≈ 0.99 * 0.25
    @test s ≈ α .* [1.0, -1.0, 0.0, 10.0]
    @test s_z_l ≈ α_z .* [0.0, -4.0, 0.0, 0.0] && s_z_u ≈ α_z .* [-0.5, 0.0, 0.0, 0.0]
    @test Penelopt.isinterior(ϕ, x .+ s)
    @test Penelopt.truncate_to_boundary!(s, s_z_l, s_z_u, x, nothing) == (1.0, 1.0)

    Penelopt.update_multipliers!(ϕ, s_z_l, s_z_u)
    @test ϕ.z_l ≈ [0.0, 1.0, 1.0, 0.0] .+ s_z_l && ϕ.z_u ≈ [1.0, 0.0, 1.0, 0.0] .+ s_z_u
    @test isnothing(Penelopt.update_multipliers!(nothing, s_z_l, s_z_u))
    Penelopt.initialize_multipliers!(ϕ)
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

  # The barrier Hessian is primal-dual, Z_l (X - L)⁻¹ + Z_u (U - X)⁻¹. It coincides with
  # the primal barrier Hessian μ (X - L)⁻² + μ (U - X)⁻² when z = μ / slack.
  bnlp.ϕ.z_l .= [isfinite(l[i]) ? μ / (x[i] - l[i]) : 0.0 for i in eachindex(x)]
  bnlp.ϕ.z_u .= [isfinite(u[i]) ? μ / (u[i] - x[i]) : 0.0 for i in eachindex(x)]

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