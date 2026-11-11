# Exact ℓ₂-penalty method for the regularized equality-constrained problem
#
#   minimize f(x) + h(x)  subject to  c(x) = 0,
#
# where h is proper, lsc and prox-bounded. Each penalized subproblem
#
#   minimize f(x) + τ‖c(x)‖₂ + h(x)
#
# is solved by Envelopt, whose augmented Lagrangian subproblems
#
#   minimize f(x) + e_{μ τ‖⋅‖₂}(c(x) + μy) + h(x)
#
# are solved by a solver from RegularizedOptimization.jl (R2 by default) that
# handles h through its proximal operator.

function Penelopt.L2Penalty(
  reg_nlp::RegularizedNLPModel{T,V};
  kwargs...,
) where {T<:Real,V}
  solver = Penelopt.L2PenaltySolver(
    reg_nlp;
  )
  stats = PeneloptExecutionStats(reg_nlp)
  SolverCore.solve!(solver, reg_nlp, stats; kwargs...)
  return stats
end

function Penelopt.L2PenaltySolver(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  kwargs...,
) where {T,V}
  nlp = reg_nlp.model
  x0 = nlp.meta.x0
  x, xn, s, s0 = similar(x0), similar(x0), similar(x0), zero(x0)
  temp_b = similar(x0, nlp.meta.ncon)
  dual_res = similar(x0)
  cn = similar(x0, nlp.meta.ncon)
  y = similar(x0, nlp.meta.ncon)
  y_report = similar(x0, nlp.meta.ncon)
  ∇fk = similar(x0)

  penalty_subproblem = Penelopt.L2PenalizedProblem(nlp) # f(x) + τ‖c(x)‖₂
  envelopt_subproblem = EnveloptNLPModel(penalty_subproblem)
  regularized_subproblem = RegularizedNLPModel(envelopt_subproblem, reg_nlp.h, reg_nlp.selected) # f(x) + τ‖c(x)‖₂ + h(x)


  substats = Envelopt.EnveloptExecutionStats(regularized_subproblem)
  solver = Envelopt.EnveloptSolver(
    regularized_subproblem;
  )

  return Penelopt.L2PenaltySolver(
    x,
    xn,
    y,
    cn,
    dual_res,
    s,
    s0,
    ∇fk,
    temp_b,
    y_report,
    solver,
    regularized_subproblem,
    substats,
  )
end

function Envelopt.EnveloptNLPModel(nlp::M; kwargs...) where {M <: Penelopt.AbstractPenalizedProblem}
  Fmodel = CFModel(nlp)  # min 0 s.t. F(x) = c(x) = 0
  EnveloptNLPModel(nlp, Fmodel, nlp.h; kwargs...)
end