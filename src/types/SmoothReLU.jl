abstract type AbstractSmoothReLU{T, S} end

set_penalty!(h::AbstractSmoothReLU, lambda) = (h.lambda = lambda)
set_smoothing!(h::AbstractSmoothReLU, β) = (h.β = β)

# TODO: add softplus

mutable struct SquarePlus{T, S} <: AbstractSmoothReLU{T, S}
  l::S
  u::S
  z_l::S
  z_u::S
  β::T
  lambda::T
end

function NLPModels.obj(φ::SquarePlus, x; β = nothing)
  v, vβ = zero(eltype(x)), isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.l[i]) && (v += squareplus(x[i] - φ.l[i], vβ))
    isfinite(φ.u[i]) && (v += squareplus(φ.u[i] - x[i], vβ))
  end
  return v * φ.lambda
end

function add_grad!(φ::SquarePlus, x, g; β = nothing)
  vβ = isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.l[i]) && (g[i] += φ.lambda * dsquareplus(x[i] - φ.l[i], vβ))
    isfinite(φ.u[i]) && (g[i] -= φ.lambda * dsquareplus(φ.u[i] - x[i], vβ))
  end
end

function add_l_grad!(φ::SquarePlus, x, g; β = nothing)
  vβ = isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.l[i]) && (g[i] += φ.lambda * dsquareplus(x[i] - φ.l[i], vβ))
  end
end

function add_u_grad!(φ::SquarePlus, x, g; β = nothing)
  vβ = isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.u[i]) && (g[i] -= φ.lambda * dsquareplus(φ.u[i] - x[i], vβ))
  end
end

function add_grad!(φ::SquarePlus, x, g; β = nothing)
  vβ = isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.l[i]) && (g[i] += φ.lambda * dsquareplus(x[i] - φ.l[i], vβ))
    isfinite(φ.u[i]) && (g[i] -= φ.lambda * dsquareplus(φ.u[i] - x[i], vβ))
  end
end

function add_hess_diag!(φ::SquarePlus, x, d; β = nothing)
  vβ = isnothing(β) ? φ.β : β
  for i in eachindex(x)
    isfinite(φ.l[i]) && (d[i] += φ.lambda * d2squareplus(x[i] - φ.l[i], vβ))
    isfinite(φ.u[i]) && (d[i] += φ.lambda * d2squareplus(φ.u[i] - x[i], vβ))
  end
end

function squareplus(x, β)
  iszero(β) && return max(zero(x), -x)
  r = sqrt(x^2 + 4 * β^2)
  return x ≥ 0 ? (2 * β^2) / (r + x) : (r - x) / 2
end

function dsquareplus(x, β)
  if iszero(β)
    return x > 0 ? zero(x) : -one(x)
  else
    return - squareplus(x, β) / sqrt(x^2 + 4 * β^2)
  end
end

function d2squareplus(x, β)
  iszero(β) && return zero(x)
  return (2 * β^2) / sqrt(x^2 + 4 * β^2)^3
end
