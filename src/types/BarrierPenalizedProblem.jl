export BarrierPenalizedProblem, LogBarrierModel, add_log_barrier, set_barrier!

"""
    LogBarrierModel(nlp; μ) <: AbstractNLPModel

Wraps `nlp` and adds the log barrier of its bounds to the objective: `f(x) + ϕ(x)`.
The bounds are removed from the `meta` and the Hessian gets `diag(∇²ϕ)` appended
to its structure (duplicate entries are summed).
"""
struct LogBarrierModel{
  T,
  S,
  M<:AbstractNLPModel{T,S},
  B<:LogBarrier{T},
  Meta<:AbstractNLPModelMeta{T,S},
} <: AbstractNLPModel{T,S}
  meta::Meta
  counters::Counters
  model::M
  ϕ::B
end

function LogBarrierModel(nlp::AbstractNLPModel{T,S}; μ) where {T,S}
  n = get_nvar(nlp)
  ϕ = LogBarrier(T(μ), get_lvar(nlp), get_uvar(nlp))
  meta = NLPModelMeta(
    nlp.meta;
    lvar = fill!(similar(get_lvar(nlp)), T(-Inf)),
    uvar = fill!(similar(get_uvar(nlp)), T(Inf)),
    nnzh = get_nnzh(nlp) + n,
    islp = false,
    name = string(get_name(nlp), " (log barrier)"),
  )
  return LogBarrierModel(meta, Counters(), nlp, ϕ)
end

get_model(nlp::LogBarrierModel) = nlp.model
get_barrier(nlp::LogBarrierModel) = nlp.ϕ
get_barrier(::Nothing) = nothing

"""
    add_log_barrier(nlp; μ)

Return `LogBarrierModel(nlp; μ)` if `nlp` has bounds, and `nlp` otherwise.
"""
add_log_barrier(nlp::AbstractNLPModel{T,V}; μ = one(T)) where{T,V} = has_bounds(nlp) ? LogBarrierModel(nlp; μ) : nlp

"""
    BarrierPenalizedProblem(nlp; μ)

Given `min f(x) s.t. c(x) = 0, l ≤ x ≤ u`, construct the penalized barrier problem

    minimize f(x) + μϕ(x) + τ‖c(x)‖₂,

where `ϕ` is the log barrier of the bounds. Returns an `L2PenalizedProblem`
whose smooth part is a `LogBarrierModel`.
"""
BarrierPenalizedProblem(nlp::AbstractNLPModel; μ) = L2PenalizedProblem(LogBarrierModel(nlp; μ))

# NLPModels API
function NLPModels.obj(nlp::LogBarrierModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return obj(nlp.model, x) + nlp.ϕ(x)
end

function NLPModels.grad!(nlp::LogBarrierModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(nlp, :neval_grad)
  grad!(nlp.model, x, g)
  return add_grad!(g, nlp.ϕ, x)
end

function NLPModels.cons!(nlp::LogBarrierModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(nlp, :neval_cons)
  return cons!(nlp.model, x, c)
end

function NLPModels.jac_structure!(nlp::LogBarrierModel, rows::AbstractVector, cols::AbstractVector)
  jac_structure!(nlp.model, rows, cols)
  return rows, cols
end

function NLPModels.jac_coord!(nlp::LogBarrierModel, x::AbstractVector, vals::AbstractVector)
  NLPModels.increment!(nlp, :neval_jac)
  return jac_coord!(nlp.model, x, vals)
end

function NLPModels.hess_structure!(nlp::LogBarrierModel, rows::AbstractVector, cols::AbstractVector)
  k = get_nnzh(nlp.model)
  hess_structure!(nlp.model, view(rows, 1:k), view(cols, 1:k))
  rows[(k+1):end] .= 1:get_nvar(nlp)
  cols[(k+1):end] .= 1:get_nvar(nlp)
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::LogBarrierModel{T},
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  k = get_nnzh(nlp.model)
  hess_coord!(nlp.model, x, view(vals, 1:k); obj_weight)
  hess_diag!(view(vals, (k+1):length(vals)), nlp.ϕ, x, obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  nlp::LogBarrierModel{T},
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  k = get_nnzh(nlp.model)
  hess_coord!(nlp.model, x, y, view(vals, 1:k); obj_weight)
  hess_diag!(view(vals, (k+1):length(vals)), nlp.ϕ, x, obj_weight)
  return vals
end
