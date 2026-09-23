abstract type AbstractShiftedPenalizedProblem{T,S} <: AbstractPenalizedProblem{T,S} end

"""
    shifted_penalty_problem = ShiftedL2PenalizedProblem(nlp::L2PenalizedProblem, x::AbstractVector)

Given a `L2PenalizedProblem` model `nlp` representing the penalized problem

    minimize f(x) + τ‖c(x)‖₂,

construct the shifted L2 penalty problem around the point `x`:

    minimize f(x) + ∇f(x)ᵀ s + 1/2 sᵀ H(x) s + τ‖c(x) + J(x)s‖₂. 

The L2PenalizedProblem is made of the following components:
- `model`: the quadratic model of the original NLP model `nlp` around the point `x`;
- `h`: the shifted penalty term, which is a `ShiftedCompositeNormL2` object, see `jl`.
- `parent`: the original `L2PenalizedProblem` model `nlp`.
"""
mutable struct ShiftedL2PenalizedProblem{
  T,
  S,
  M<:AbstractQuadraticModel{T,S},
  H<:ShiftedCompositeNormL2,
  P<:Union{Nothing,L2PenalizedProblem},
  SN<:Union{Nothing,S},
  meta<:AbstractNLPModelMeta,
} <: AbstractShiftedPenalizedProblem{T,S}
  model::M
  h::H
  parent::P
  meta::meta
  _qn_y::SN
  _qn_x_prev::SN
  _is_first_shift::Bool
end

function ShiftedL2PenalizedProblem(
  penalty_nlp::L2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,VN1<:Union{Nothing,V}, VN2<:Union{Nothing,V}}

  nlp, h = penalty_nlp.model, penalty_nlp.h
  ∇f = isnothing(∇f) ? grad(nlp, x) : ∇f
  ψ = shifted(h, x)

  # Quasi-Newton Constructor
  if !isnothing(find_model(QuasiNewtonModel, nlp))
    B = get_op(nlp)
    φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

    return ShiftedL2PenalizedProblem(
      φ,
      ψ,
      penalty_nlp,
      penalty_nlp.meta,
      similar(∇f),
      zero(∇f),
      true,
    )
  else # Full Hessian Constructor
    n = length(x)
    y = isnothing(y) ? zeros(T, nlp.meta.ncon) : y

    Bi, Bj = hess_structure(nlp)
    Bv = hess_coord(nlp, x, y)
    B = SparseMatrixCOO(n, n, Bi, Bj, Bv)

    φ = QuadraticModel(∇f, B, x0 = x, regularize = true)

    ψ = shifted(h, x)

    return ShiftedL2PenalizedProblem(
      φ,
      ψ,
      penalty_nlp,
      penalty_nlp.meta,
      nothing,
      nothing,
      true,
    )
  end
end

# ShiftedProximalOperators API
function shifted(
  penalty_nlp::L2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,VN1<:Union{Nothing,V}, VN2<:Union{Nothing,V}}
  return ShiftedL2PenalizedProblem(penalty_nlp, x; ∇f = ∇f, y = y)
end

# TODO: Move to QuadraticModels.jl ?
function shift!(
  φ::QuadraticModel{T,V,M},
  nlp::AbstractNLPModel{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,M <: SparseMatrixCOO,VN1<:Union{Nothing,V},VN2<:Union{Nothing,V}}
  g = φ.data.c
  isnothing(∇f) ? grad!(nlp, x, g) : (g .= ∇f)
  isnothing(y) ? hess_coord!(nlp, x, φ.data.H.vals) : hess_coord!(nlp, x, y, φ.data.H.vals)
end

function shift!(
  φ::QuadraticModel{T,V,M},
  nlp::AbstractNLPModel{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
) where {T,V,M,VN1<:Union{Nothing,V},VN2<:Union{Nothing,V}}
  g = φ.data.c
  isnothing(∇f) ? grad!(nlp, x, g) : (g .= ∇f)
end

# Pre shift callbacks
_pre_shift!(::AbstractNLPModel, shifted_nlp, x; kwargs...) = nothing
function _pre_shift!(::CompactBFGSModel, shifted_nlp, x; y = nothing, kwargs...)
  shifted_nlp._is_first_shift && return
  φ, ψ = shifted_nlp.model, shifted_nlp.h
  qn_y = shifted_nlp._qn_y
  qn_y .= φ.data.c                                          # g_prev = ∇f(x_prev)
  isnothing(y) || mul!(qn_y, ψ.A', y, -one(eltype(qn_y)), -one(eltype(qn_y)))
  # qn_y = -(g_prev + J(x_prev)ᵀ y)
end

# Post shift callbacks
_post_shift!(::AbstractNLPModel, shifted_nlp, x; kwargs...) = nothing
function _post_shift!(::CompactBFGSModel, shifted_nlp, x; y = nothing, kwargs...)
  φ, ψ = shifted_nlp.model, shifted_nlp.h
  qn_y, qn_x_prev = shifted_nlp._qn_y, shifted_nlp._qn_x_prev

  if shifted_nlp._is_first_shift
    shifted_nlp._is_first_shift = false
  else
    qn_y .+= φ.data.c                                       # += g_new
    isnothing(y) || mul!(qn_y, ψ.A', y, one(eltype(qn_y)), one(eltype(qn_y)))
    # qn_y = g_new - g_prev - J(x_prev)ᵀy + J(x)ᵀy
    qn_s = qn_x_prev            # reuse x_prev's storage as scratch, it's about to be overwritten
    qn_s .= x .- qn_s
    push!(φ.data.H, qn_s, qn_y)
  end
  qn_x_prev .= x
end

function shift!(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T,V},
  x::V;
  ∇f::VN1 = nothing,
  y::VN2 = nothing,
  J::Ma = nothing,
  c::VN3 = nothing,
) where {
  T,
  V,
  VN1<:Union{Nothing,V},
  VN2<:Union{Nothing,V},
  VN3<:Union{Nothing,V},
  Ma<:Union{Nothing,AbstractMatrix{T}},
}
  nlp, h = shifted_penalty_nlp.parent.model, shifted_penalty_nlp.parent.h
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  foreach(layer -> _pre_shift!(layer, shifted_penalty_nlp, x; ∇f = ∇f, y = y, J = J, c = c), layers(nlp))

  shift!(φ, nlp, x, ∇f = ∇f, y = y)
  shift!(ψ, x, J = J, c = c)

  foreach(layer -> _post_shift!(layer, shifted_penalty_nlp, x; ∇f = ∇f, y = y, J = J, c = c), layers(nlp))
end

# Miscellaneous
function set_penalty!(nlp::ShiftedL2PenalizedProblem{T}, τ::T) where {T}
  nlp.h.h.lambda = τ
  nlp.parent.h.h.lambda = τ
end

function check_descent(
  shifted_penalty_nlp::ShiftedL2PenalizedProblem{T},
  s::AbstractVector,
) where {T}
  φ, ψ = shifted_penalty_nlp.model, shifted_penalty_nlp.h

  cx, τ = ψ.b, ψ.h.lambda
  ψ0 = τ * norm(cx) # φ0 = 0
  return ψ0 - obj(φ, s) - ψ(s) >= 0
end

function reset!(shifted_penalty_nlp::ShiftedL2PenalizedProblem)
  parent = shifted_penalty_nlp.parent
  isnothing(parent) && return shifted_penalty_nlp
  φ = shifted_penalty_nlp.model

  if !isnothing(find_model(CompactBFGSModel, parent.model))
    shifted_penalty_nlp._qn_x_prev .= 0
    LinearOperators.reset!(φ.data.H)
    shifted_penalty_nlp._is_first_shift = true
  end
  return shifted_penalty_nlp
end