export NullHessianModel

"""
    NullHessianModel(nlp)

Wrap `nlp` so that its Hessian of the Lagrangian is replaced by the zero matrix.
All other evaluations (objective, constraints, gradients, Jacobians) are forwarded to `nlp`.

This is what `L2Penalty` uses with `qn_hessian_approximation = "null"`.
Construct it yourself only when using a preallocated [`L2PenaltySolver`](@ref),
see [Preallocation](@ref).
"""
mutable struct NullHessianModel{
  T,
  S,
  M<:AbstractNLPModel{T,S},
  Meta<:AbstractNLPModelMeta{T,S},
} <: QuasiNewtonModel{T,S}
  meta::Meta
  model::M
  op::SparseMatrixCOO{T}
end

function NullHessianModel(nlp::AbstractNLPModel{T,S}) where {T,S}
  return NullHessianModel(nlp.meta, nlp, coo_spzeros(T, nlp.meta.nvar, nlp.meta.nvar))
end

get_op(nlp::NullHessianModel) = nlp.op
get_model(nlp::NullHessianModel) = nlp.model
@default_counters NullHessianModel model

# Hessian API 

NLPModels.hess_op(nlp::NullHessianModel{T,V}, x::AbstractVector; kwargs...) where {T,V} =
  opZeros(T, nlp.meta.nvar, nlp.meta.nvar)
NLPModels.hprod(nlp::NullHessianModel, x::AbstractVector, v::AbstractVector; kwargs...) =
  zero(v)

function NLPModels.hprod!(
  nlp::NullHessianModel,
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  Hv .= 0
end

function NLPModels.hprod!(
  nlp::NullHessianModel,
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  kwargs...,
)
  Hv .= 0
end

function NLPModels.hess(nlp::NullHessianModel{T,V}, args...; kwargs...) where {T,V}
  return spzeros(T, nlp.meta.nvar, nlp.meta.nvar)
end

function NLPModels.hess_coord(nlp::NullHessianModel{T,V}, args...; kwargs...) where {T,V}
  return T[]
end

function NLPModels.hess_coord!(nlp::NullHessianModel{T,V}, args...; kwargs...) where {T,V}
  return
end

function NLPModels.hess_structure(
  nlp::NullHessianModel{T,V},
  args...;
  kwargs...,
) where {T,V}
  return Int[], Int[]
end

function NLPModels.hess_structure!(
  nlp::NullHessianModel{T,V},
  args...;
  kwargs...,
) where {T,V}
  return
end
