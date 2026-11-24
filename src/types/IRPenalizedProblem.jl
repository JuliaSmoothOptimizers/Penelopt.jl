abstract type AbstractIRPenalizedProblem{T,S} <: AbstractNLPModel{T,S} end

"""
  penalty_problem = IRPenalizedProblem(nlp::AbstractNLPModel)

Given an NLP model `nlp` representing the bound constrained problem

  minimize f(x) subject to l ≤ x ≤ u,

construct the smooth inequality relaxation penalty problem

  minimize f(x) + τφ(x; l, u, β)ᵀe.

The IRPenalizedProblem is made of the following components:
- `model`: the original NLP model `nlp`;
- `h`: the penalty term, representing τφ(x; l, u, β)ᵀe, which is an `AbstractSmoothReLU` object.
"""
mutable struct IRPenalizedProblem{
  T,
  S,
  M<:AbstractNLPModel{T,S},
  P<:AbstractSmoothReLU{T,S},
} <: AbstractIRPenalizedProblem{T,S}
  model::M
  h::P
  meta::NLPModelMeta{T,S}
end

function IRPenalizedProblem(nlp::AbstractNLPModel{T,S}) where {T,S}
  m = nlp.meta
  lvar = fill!(similar(m.lvar), T(-Inf))
  uvar = fill!(similar(m.uvar), T(Inf))
  meta = NLPModelMeta(m; lvar = lvar, uvar = uvar, nnzh = m.nnzh + m.nvar)
  return IRPenalizedProblem(
    nlp,
    SquarePlus(nlp.meta.lvar, nlp.meta.uvar, zero(nlp.meta.lvar), zero(nlp.meta.uvar), one(T), one(T)),
    meta
  )
end

## Utils

NLPModels.@default_counters IRPenalizedProblem model
NLPModelsModifiers.get_model(nlp::IRPenalizedProblem) = nlp.model

set_penalty!(nlp::IRPenalizedProblem, λ) = set_penalty!(nlp.h, λ)
set_smoothing!(nlp::IRPenalizedProblem, β) = set_smoothing!(nlp.h, β)

function get_bound_multipliers!(nlp::IRPenalizedProblem, x; β = nothing)
  z_l, z_u = nlp.h.z_l, nlp.h.z_u

  z_l .= 0
  z_u .= 0

  add_l_grad!(nlp.h, x, z_l; β = β)
  add_u_grad!(nlp.h, x, z_u; β = β)

  z_l .*= - nlp.h.lambda
  z_u .*= - nlp.h.lambda
  return z_l, z_u
end

## NLPModels API
function NLPModels.obj(nlp::IRPenalizedProblem, x; β = nothing)
  return obj(nlp.model, x) + obj(nlp.h, x; β = β)
end

function NLPModels.grad!(nlp::IRPenalizedProblem, x, g; β = nothing)
  grad!(nlp.model, x, g)
  add_grad!(nlp.h, x, g; β = β)
  return g
end

function NLPModels.hess_structure!(nlp::IRPenalizedProblem, rows::Vector{Int}, cols::Vector{Int})
  @views hess_structure!(nlp.model, rows[1:get_nnzh(nlp.model)], cols[1:get_nnzh(nlp.model)])
  @views rows[get_nnzh(nlp.model)+1:end] .= 1:get_nvar(nlp)
  @views cols[get_nnzh(nlp.model)+1:end] .= 1:get_nvar(nlp)
  return rows, cols
end

function NLPModels.hess_coord!(nlp::IRPenalizedProblem, x, y, vals; obj_weight = one(eltype(vals)), β = nothing)
  @views hess_coord!(nlp.model, x, y, vals[1:get_nnzh(nlp.model)]; obj_weight = obj_weight)
  @views vals[get_nnzh(nlp.model)+1:end] .= zero(eltype(vals))
  @views add_hess_diag!(nlp.h, x, vals[get_nnzh(nlp.model)+1:end]; β = β)
  return vals
end

# Pass to the original model
NLPModels.cons!(nlp::IRPenalizedProblem, x::AbstractVector, c::AbstractVector) = cons!(nlp.model, x, c)
NLPModels.jac_structure!(nlp::IRPenalizedProblem, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = jac_structure!(nlp.model, rows, cols)
NLPModels.jac_coord!(nlp::IRPenalizedProblem, x::AbstractVector, vals::AbstractVector) = jac_coord!(nlp.model, x, vals)