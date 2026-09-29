mutable struct watchdog_checkpoint{T,V,HV}
  xk::V
  ∇fk::V
  ck::V
  yk::V
  Jkvals::V
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
  z_l::V # Bound multipliers of the log barrier (empty if there are no bounds)
  z_u::V
end

function watchdog_checkpoint(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P};
  m_monotone = 5,
  barrier = nothing,
) where {T,V,M,H,P}
  φ, ψ = nlp.model, nlp.h
  ∇f_model, b_model = φ.data.c, ψ.b
  xk, ∇fk = similar(∇f_model), similar(∇f_model)
  ck, yk = similar(b_model), similar(b_model)
  Jkvals = similar(ψ.A.vals)
  Hkvals = similar(φ.data.H.vals)
  return watchdog_checkpoint(
    xk,
    ∇fk,
    ck,
    yk,
    Jkvals,
    Hkvals,
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
    _similar_z(barrier, ∇f_model, :z_l),
    _similar_z(barrier, ∇f_model, :z_u),
  )
end

function watchdog_checkpoint(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P};
  m_monotone = 5,
  barrier = nothing,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h
  ∇f_model, b_model = φ.data.c, ψ.b
  xk, ∇fk = similar(∇f_model), similar(∇f_model)
  ck, yk = similar(b_model), similar(b_model)
  Jkvals = similar(ψ.A.vals)
  Hk =
    (isa(φ.data.H, AbstractLinearOperator) || O <: NullHessianModel) ? nothing :
    similar(φ.data.H)
  return watchdog_checkpoint(
    xk,
    ∇fk,
    ck,
    yk,
    Jkvals,
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
    _similar_z(barrier, ∇f_model, :z_l),
    _similar_z(barrier, ∇f_model, :z_u),
  )
end

_similar_z(::Nothing, v, ::Symbol) = similar(v, 0)
_similar_z(ϕ::LogBarrier, v, name::Symbol) = similar(getfield(ϕ, name))

_save_bound_multipliers!(checkpoint::watchdog_checkpoint, ::Nothing) = nothing
function _save_bound_multipliers!(checkpoint::watchdog_checkpoint, ϕ::LogBarrier)
  checkpoint.z_l .= ϕ.z_l
  checkpoint.z_u .= ϕ.z_u
end

_restore_bound_multipliers!(::Nothing, checkpoint::watchdog_checkpoint) = nothing
function _restore_bound_multipliers!(ϕ::LogBarrier, checkpoint::watchdog_checkpoint)
  ϕ.z_l .= checkpoint.z_l
  ϕ.z_u .= checkpoint.z_u
end

function save!(
  checkpoint::watchdog_checkpoint,
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  stats;
  barrier = nothing,
) where {T,V,M,H,P}
  φ, ψ = nlp.model, nlp.h

  checkpoint.xk .= x
  checkpoint.yk .= y
  checkpoint.∇fk .= φ.data.c
  checkpoint.Hkvals .= φ.data.H.vals
  checkpoint.σk = φ.data.σ
  checkpoint.ck .= ψ.b
  checkpoint.Jkvals .= ψ.A.vals
  checkpoint.primal_feas = stats.primal_feas
  checkpoint.dual_feas = stats.dual_feas
  checkpoint.iter = stats.iter
  checkpoint.fk = stats.solver_specific[:smooth_obj]
  checkpoint.hk = stats.solver_specific[:nonsmooth_obj]
  _save_bound_multipliers!(checkpoint, barrier)
end

function save!(
  checkpoint::watchdog_checkpoint,
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  stats;
  barrier = nothing,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h

  checkpoint.xk .= x
  checkpoint.yk .= y
  checkpoint.∇fk .= φ.data.c
  !isnothing(checkpoint.Hkvals) && copy!(checkpoint.Hkvals, φ.data.H)
  checkpoint.σk = φ.data.σ
  checkpoint.ck .= ψ.b
  checkpoint.Jkvals .= ψ.A.vals
  checkpoint.primal_feas = stats.primal_feas
  checkpoint.dual_feas = stats.dual_feas
  checkpoint.iter = stats.iter
  checkpoint.fk = stats.solver_specific[:smooth_obj]
  checkpoint.hk = stats.solver_specific[:nonsmooth_obj]
  _save_bound_multipliers!(checkpoint, barrier)
end

function fallback!(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  checkpoint::watchdog_checkpoint;
  barrier = nothing,
) where {T,V,M,H,P}
  φ, ψ = nlp.model, nlp.h

  x .= checkpoint.xk
  y .= checkpoint.yk
  φ.data.c .= checkpoint.∇fk
  φ.data.H.vals .= checkpoint.Hkvals
  φ.data.σ = checkpoint.σk
  ψ.b .= checkpoint.ck
  ψ.A.vals .= checkpoint.Jkvals
  _restore_bound_multipliers!(barrier, checkpoint)
end

function fallback!(
  nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  x,
  y,
  checkpoint::watchdog_checkpoint;
  barrier = nothing,
) where {T,V,M,H,O<:QuasiNewtonModel,P<:L2PenalizedProblem{T,V,O}}
  φ, ψ = nlp.model, nlp.h

  x .= checkpoint.xk
  y .= checkpoint.yk
  φ.data.c .= checkpoint.∇fk
  !isnothing(checkpoint.Hkvals) && copy!(φ.data.H, checkpoint.Hkvals)
  φ.data.σ = checkpoint.σk
  ψ.b .= checkpoint.ck
  ψ.A.vals .= checkpoint.Jkvals
  _restore_bound_multipliers!(barrier, checkpoint)
end

function activate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = true
end

function deactivate!(checkpoint::watchdog_checkpoint)
  checkpoint.active = false
end

is_active(checkpoint::watchdog_checkpoint) = checkpoint.active

function check_watchdog!(
  checkpoint::watchdog_checkpoint{T,V,V},
  stats,
  mk,
  xk,
  watchdog_max_iter,
  η0,
) where {T,V}

  achieve_reduction =
    (checkpoint.fk + checkpoint.hk - stats.objective > 1/2*η0*checkpoint.dual_feas^2) ||
    (stats.dual_feas < (1-η0)*checkpoint.dual_feas)
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

function check_watchdog!(
  checkpoint::watchdog_checkpoint{T,V,HV},
  stats,
  mk,
  xk,
  watchdog_max_iter,
  η0,
) where {T,V,HV<:CompactBFGS}

  achieve_reduction =
    (checkpoint.fk + checkpoint.hk - stats.objective > 1/2*η0*checkpoint.dual_feas^2) ||
    (stats.dual_feas < (1-η0)*checkpoint.dual_feas)
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

function check_watchdog!(
  checkpoint::watchdog_checkpoint{T,V,HV},
  stats,
  mk,
  xk,
  watchdog_max_iter,
  η0,
) where {T,V,HV<:Nothing}

  achieve_reduction =
    (checkpoint.fk + checkpoint.hk - stats.objective > 1/2*η0*checkpoint.dual_feas^2) ||
    (stats.dual_feas < (1-η0)*checkpoint.dual_feas)
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
