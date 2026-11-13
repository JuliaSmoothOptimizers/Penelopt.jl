"""
    watchdog_checkpoint

Iterate saved when the watchdog is activated, so that R2N can fall back to it.

The Jacobian is never stored: it is re-evaluated in `fallback!`. In the exact Hessian
case, the Hessian is not stored either and is re-evaluated as well.
"""
mutable struct watchdog_checkpoint{T,V,HV}
  xk::V
  ∇fk::V
  ck::V
  yk::V
  Hkvals::HV
  active::Bool
  iter::Int
  primal_feas::T
  dual_feas::T
  σk::T
  fk::T
  hk::T
  m_fh_hist::V
  s::V
  v::V
end

function watchdog_checkpoint(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P};
  m_monotone = 5,
) where {T,V,M,H,P}
  φ, ψ = nlp.model, nlp.h
  ∇f_model, b_model = φ.data.c, ψ.b
  xk, ∇fk = similar(∇f_model), similar(∇f_model)
  ck, yk = similar(b_model), similar(b_model)
  return watchdog_checkpoint(
    xk,
    ∇fk,
    ck,
    yk,
    nothing,
    false,
    0,
    zero(T),
    zero(T),
    zero(T),
    zero(T),
    zero(T),
    similar(∇f_model, m_monotone-1),
    similar(xk),
    similar(xk),
  )
end

function watchdog_checkpoint(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P};
  m_monotone = 5,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h
  ∇f_model, b_model = φ.data.c, ψ.b
  xk, ∇fk = similar(∇f_model), similar(∇f_model)
  ck, yk = similar(b_model), similar(b_model)
  Hk =
    (isa(φ.data.H, AbstractLinearOperator) || O <: NullHessianModel) ? nothing :
    similar(φ.data.H)
  return watchdog_checkpoint(
    xk,
    ∇fk,
    ck,
    yk,
    Hk,
    false,
    0,
    zero(T),
    zero(T),
    zero(T),
    zero(T),
    zero(T),
    similar(∇f_model, m_monotone-1),
    similar(xk),
    similar(xk),
  )
end

function save!(
  checkpoint::watchdog_checkpoint,
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  stats,
) where {T,V,M,H,P}
  φ, ψ = nlp.model, nlp.h

  checkpoint.xk .= x
  checkpoint.yk .= y
  checkpoint.∇fk .= φ.data.c
  checkpoint.σk = φ.data.σ
  checkpoint.ck .= ψ.b
  checkpoint.primal_feas = stats.primal_feas
  checkpoint.dual_feas = stats.dual_feas
  checkpoint.iter = stats.iter
  checkpoint.fk = stats.solver_specific[:smooth_obj]
  checkpoint.hk = stats.solver_specific[:nonsmooth_obj]
end

function save!(
  checkpoint::watchdog_checkpoint,
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  stats,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h

  checkpoint.xk .= x
  checkpoint.yk .= y
  checkpoint.∇fk .= φ.data.c
  !isnothing(checkpoint.Hkvals) && copy!(checkpoint.Hkvals, φ.data.H)
  checkpoint.σk = φ.data.σ
  checkpoint.ck .= ψ.b
  checkpoint.primal_feas = stats.primal_feas
  checkpoint.dual_feas = stats.dual_feas
  checkpoint.iter = stats.iter
  checkpoint.fk = stats.solver_specific[:smooth_obj]
  checkpoint.hk = stats.solver_specific[:nonsmooth_obj]
end

function fallback!(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  checkpoint::watchdog_checkpoint,
) where {T,V,M,H,P}
  φ = nlp.model

  x .= checkpoint.xk
  y .= checkpoint.yk
  φ.data.σ = checkpoint.σk

  # H and J are not stored in the exact Hessian case: reuse the stored ∇f and c,
  # and re-evaluate H(x, y) and J(x) at the checkpoint.
  shift!(nlp, x, ∇f = checkpoint.∇fk, y = y, c = checkpoint.ck)
end

function fallback!(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  checkpoint::watchdog_checkpoint,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h

  x .= checkpoint.xk
  y .= checkpoint.yk
  φ.data.c .= checkpoint.∇fk
  !isnothing(checkpoint.Hkvals) && copy!(φ.data.H, checkpoint.Hkvals)
  φ.data.σ = checkpoint.σk

  # J is not stored: reuse the stored c and re-evaluate J(x) at the checkpoint.
  # Shift ψ only, since shifting `nlp` would push a spurious pair into the approximation.
  shift!(ψ, x, c = checkpoint.ck)

  # The next quasi-Newton pair must be computed w.r.t. the checkpoint,
  # not w.r.t. the last iterate visited during the watchdog.
  nlp._qn_x_prev .= x
end

function activate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = true
end

function deactivate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = false
end

is_active(checkpoint::watchdog_checkpoint) = checkpoint.active

function check_watchdog!(
  checkpoint::watchdog_checkpoint,
  stats,
  mk,
  xk,
  watchdog_max_iter,
  η0,
)
  achieve_reduction =
    (
      checkpoint.fk + checkpoint.hk - stats.objective >
      1/2*η0*min(checkpoint.dual_feas, 1)^2
    ) || (stats.dual_feas < (1-η0)*checkpoint.dual_feas)
  max_iter = stats.iter - checkpoint.iter > watchdog_max_iter

  if !is_active(checkpoint)
    return false
  elseif achieve_reduction
    deactivate!(checkpoint)
    return false
  elseif !max_iter
    return false
  else
    return true
  end
end
