using CUTEst,
  Penelopt, NLPModels, NLPModelsIpopt, NLPModelsModifiers, LinearAlgebra, DataFrames

const BENCHMARK_MAX_TIME = 300.0
"""
    certify_local_infeasibility(nlp, xbar; Δ=10.0, tol=1e-9, feas_tol=1e-3)

Check whether `x̄ = xbar` is a locally infeasible point of `nlp` by solving

    min_x 1/2 ||c(x)||² s.t. ||J(x̄)(x - x̄)|| ≤ Δ

with IPOPT, starting from `x̄`. Certified infeasible if the trust-region
constraint is inactive at the solution and ||c(x)|| > feas_tol.
"""
function certify_local_infeasibility(
  nlp::AbstractNLPModel,
  xbar::AbstractVector;
  Δ = 10.0,
  tol = 1e-9,
  feas_tol = 1e-3,
)
  model = FeasibilityFormNLS(FeasibilityResidual(nlp))

  # r₀ = c(x̄), so the F(x) - r = 0 block is satisfied at x0
  x0 = vcat(xbar, cons(nlp, xbar))

  saved = deepcopy(nlp.counters)
  stats = try
    ipopt(
      model,
      x0 = x0,
      print_level = 0,
      tol = tol,
      dual_inf_tol = tol,
      constr_viol_tol = tol,
      compl_inf_tol = tol,
      acceptable_iter = 0,
      max_cpu_time = BENCHMARK_MAX_TIME,
    )
  finally
    nlp.counters = saved
  end

  if stats.status != :first_order
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (inner IPOPT solve terminated with status $(stats.status))"
    return false
  end

  n = nlp.meta.nvar
  xsol = stats.solution[1:n]
  tr_residual = norm(jprod(nlp, xbar, xsol - xbar)) # ||J(x̄)(x - x̄)||
  primal_feas = norm(cons(nlp, xsol))

  if tr_residual > Δ
    @warn "Local infeasibility check for $(nlp.meta.name) was inconclusive (Δ = $Δ may be too small)"
    return false
  end

  certified = primal_feas > feas_tol

  @debug "Local infeasibility check for $(nlp.meta.name): ||c(x̄)|| = $primal_feas, ||J(x̄)(x-x̄)|| = $tr_residual -> $(certified ? "certified infeasible" : "not certified")"

  return certified
end
