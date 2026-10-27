# F model for the ℓ₂ penalty term: F(x) = c(x), the body of the equality constraints,
# so that the penalized problem
#
#   minimize f(x) + τ‖c(x)‖₂
#
# is an EnveloptNLPModel with smooth part f, F = c and h = τ‖⋅‖₂.
#
# Counterpart of Envelopt.IdentityFModel. Every evaluation is forwarded to the
# underlying model, so its counters are incremented too.

"""
    CFModel(model::AbstractNLPModel)
    CFModel(nlp::Penelopt.L2PenalizedProblem)

Represent the mapping F(x) = c(x), where c is the body of the constraints of `model`
(or of `nlp.model`). The constraints must be of the form c(x) = 0; see
`remove_constraint_shift`.
"""
struct CFModel{T,S,M<:AbstractNLPModel{T,S}} <: Envelopt.AbstractFModel{T,S}
  meta::NLPModelMeta{T,S}
  counters::Counters
  model::M
end

function CFModel(model::AbstractNLPModel{T,S}) where {T,S}
  equality_constrained(model) ||
    error("CFModel: the model should only have equality constraints")
  any(!iszero, get_lcon(model)) &&
    error("CFModel: the constraints should be of the form c(x) = 0; see remove_constraint_shift")
  x0 = fill!(similar(get_x0(model)), zero(T))
  # Internal Envelopt helper, also used by IdentityFModel.
  meta = Envelopt._fmodel_meta(x0, get_ncon(model), get_nnzj(model), "c-F")
  return CFModel{T,S,typeof(model)}(meta, Counters(), model)
end

CFModel(nlp::Penelopt.L2PenalizedProblem) = CFModel(nlp.model)

# obj and grad! (identically zero) come from AbstractFModel.

function NLPModels.cons!(F::CFModel, x::AbstractVector, c::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.meta.ncon c
  NLPModels.increment!(F, :neval_cons)
  return cons!(F.model, x, c)
end

function NLPModels.jprod!(F::CFModel, x::AbstractVector, v::AbstractVector, Jv::AbstractVector)
  @lencheck F.meta.nvar x v
  @lencheck F.meta.ncon Jv
  NLPModels.increment!(F, :neval_jprod)
  return jprod!(F.model, x, v, Jv)
end

function NLPModels.jtprod!(F::CFModel, x::AbstractVector, v::AbstractVector, Jtv::AbstractVector)
  @lencheck F.meta.nvar x Jtv
  @lencheck F.meta.ncon v
  NLPModels.increment!(F, :neval_jtprod)
  return jtprod!(F.model, x, v, Jtv)
end

function NLPModels.jac_structure!(
  F::CFModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck F.meta.nnzj rows cols
  return jac_structure!(F.model, rows, cols)
end

function NLPModels.jac_coord!(F::CFModel, x::AbstractVector, vals::AbstractVector)
  @lencheck F.meta.nvar x
  @lencheck F.meta.nnzj vals
  NLPModels.increment!(F, :neval_jac)
  return jac_coord!(F.model, x, vals)
end

# Envelopt evaluates h on F(x) and needs prox_{γh}; for h = λ‖⋅‖₂,
# prox_{γh}(x) = max(1 - γλ / ‖x‖₂, 0) x. Must return h(y).
function ProximalCore.prox!(y, f::NormL2, x, γ)
  normx = norm(x)
  γλ = γ * f.lambda
  if normx > γλ
    scale = 1 - γλ / normx
    y .= scale .* x
    return f.lambda * scale * normx
  end
  fill!(y, zero(eltype(y)))
  return zero(f.lambda * normx)
end

# f(x) + τ‖c(x)‖₂ as an EnveloptNLPModel:
# - smooth part: f without its constraints (EqualityLessModel), since they are moved into F,
#   and R2/R2N refuse models with constraints;
# - h: the NormL2 stored in the penalty term, so that `set_penalty!(nlp, τ)` also updates τ
#   in the Moreau envelope (same mutable object).
function Envelopt.EnveloptNLPModel(nlp::Penelopt.L2PenalizedProblem; kwargs...)
  Fmodel = CFModel(nlp) # min 0 s.t. F(x) = c(x) = 0
  return EnveloptNLPModel(EqualityLessModel(nlp.model), Fmodel, nlp.h.h; kwargs...)
end