# Exact ℓ₂-penalty method with a nonsmooth regularizer (PeneloptEnveloptExt):
#
#   minimize f(x) + h(x)  subject to  c(x) = 0.
#
# Test problem: equality-constrained matrix completion from RegularizedProblems.jl,
#
#   minimize λ rank(X)  subject to  Xᵢⱼ = Aᵢⱼ for (i, j) ∈ Ω,
#
# where A has rank r. f = 0 and c is linear, so any failure is due to the
# penalty loop / Envelopt / the regularizer, not to the smooth model.
#
# Expects `using Envelopt, RegularizedOptimization, RegularizedProblems, ShiftedProximalOperators`
# at the top level of runtests.jl before this file is included.

@testset "Envelopt extension is loaded" begin
  @test !isnothing(Base.get_extension(Penelopt, :PeneloptEnveloptExt))
end

@testset "Rank-regularized matrix completion" begin
  Random.seed!(1234) # mat_rand uses the global RNG
  m, n, r = 12, 10, 2 # small and non-square, to catch reshape/indexing mistakes
  xs, B, ω = RegularizedProblems.mat_rand(m, n, r, 0.8, 0.0, 0.0, 0.0)
  nlp, sol = random_matrix_completion_eq_model(xs, B, ω; init = :spectral, r = r)

  # Sanity checks on the model, so that a failure below can be attributed to Penelopt.
  @test equality_constrained(nlp)
  @test !has_bounds(nlp)
  @test all(iszero, get_lcon(nlp)) # no constraint shift to remove
  @test norm(cons(nlp, sol), Inf) ≤ 1e-12

  λ = 1.0e-1
  h = Rank(λ, zeros(m, n)) # ShiftedProximalOperators.Rank: acts on vec(X)
  reg_nlp = RegularizedNLPModel(nlp, h)

  stats = L2Penalty(reg_nlp; atol = 1e-6, rtol = 1e-6, max_iter = 50, print_level = 0)

  X = reshape(stats.solution, m, n)
  @test stats.status == :first_order
  @test norm(cons(nlp, stats.solution), Inf) ≤ 1e-5
  @test rank(X; atol = 1e-6) ≤ r
  # Exact recovery is likely with 80% of the entries observed, but not guaranteed:
  # relax or drop this check if it turns out to be flaky.
  @test norm(stats.solution - sol) / norm(sol) ≤ 1e-3
end
