abstract type AbstractBarrier end

@doc raw"""
    LogBarrier(μ, l, u)

Logarithmic barrier for the bounds `l ≤ x ≤ u`:

```math
ϕ(x) = -μ \sum_i [\log(u_i - x_i) + \log(x_i - l_i)],
```
where `μ > 0`. Infinite bounds are ignored, and `ϕ(x) = Inf` if `x` is not strictly feasible.
"""
mutable struct LogBarrier{T<:Real,V<:AbstractVector{T}} <: AbstractBarrier
  μ::T
  τ::T # Fraction to boundary
  τmin::T # Minimum fraction to boundary
  l::V
  u::V
  z_l::V
  z_u::V

  function LogBarrier(μ::T, l::V, u::V; τmin = T(0.99)) where {T<:Real,V<:AbstractVector{T}}
    @assert μ > 0 "Barrier parameter μ must be positive."
    @assert length(l) == length(u) && all(l .< u) "Bounds must satisfy l < u."
    z_l, z_u = similar(l), similar(u)
    τ = max(τmin, 1 - μ)
    return new{T,V}(μ, τ, τmin, l, u, z_l, z_u)
  end
end

@inline logterm(b, d) = isfinite(b) ? (d > 0 ? -log(d) : oftype(d, Inf)) : zero(d)
@inline invd(b, d) = isfinite(b) ? inv(d) : zero(d)

function set_fraction_to_boundary!(::Nothing) end
function set_fraction_to_boundary!(ϕ::LogBarrier)
  ϕ.τ = max(ϕ.τmin, 1 - ϕ.μ)
end

function initialize_multipliers!(::Nothing) end
function initialize_multipliers!(ϕ::LogBarrier)
  for i in eachindex(ϕ.l)
    l, u = ϕ.l[i], ϕ.u[i]
    if isfinite(l)
      ϕ.z_l[i] = 1
    else
      ϕ.z_l[i] = 0
    end
    if isfinite(u)
      ϕ.z_u[i] = 1
    else
      ϕ.z_u[i] = 0
    end
  end
end

@doc raw"""
    push_to_interior!(ϕ::LogBarrier, x; κ1 = 1e-2, κ2 = 1e-2)

Move `x` sufficiently far from the bounds so that it is strictly feasible (Wächter & Biegler, §3.6).

- One-sided lower bound: `x ← max(x, l + κ1 max(1, |l|))`.
- One-sided upper bound: `x ← min(x, u - κ1 max(1, |u|))`.
- Two-sided bounds: `x` is projected onto `[l + p_l, u - p_u]` with
  `p_l = min(κ1 max(1, |l|), κ2 (u - l))` and `p_u = min(κ1 max(1, |u|), κ2 (u - l))`.

Requires `κ1 > 0` and `0 < κ2 < 1/2`. Free variables are left unchanged.
"""
function push_to_interior!(ϕ::LogBarrier{T}, x; κ1 = T(1e-2), κ2 = T(1e-2)) where {T}
  @assert κ1 > 0 && 0 < κ2 < 1 / 2 "Need κ1 > 0 and 0 < κ2 < 1/2."
  for i in eachindex(x)
    l, u = ϕ.l[i], ϕ.u[i]
    if isfinite(l) && isfinite(u)
      p_l = min(κ1 * max(one(T), abs(l)), κ2 * (u - l))
      p_u = min(κ1 * max(one(T), abs(u)), κ2 * (u - l))
      x[i] = clamp(x[i], l + p_l, u - p_u)
    elseif isfinite(l)
      x[i] = max(x[i], l + κ1 * max(one(T), abs(l)))
    elseif isfinite(u)
      x[i] = min(x[i], u - κ1 * max(one(T), abs(u)))
    end
  end
  return x
end

push_to_interior!(::Nothing, x; kwargs...) = x

@doc raw"""
    update_barrier!(ϕ::LogBarrier, x, tol; κμ = 0.2, θμ = 1.5)

Decrease the barrier parameter (Wächter & Biegler, eq. (7)):

```math
μ_{j+1} = \max\left\{\frac{ε_{\mathrm{tol}}}{10}, \min\{κ_μ μ_j, μ_j^{θ_μ}\}\right\},
```
with `κμ ∈ (0, 1)` and `θμ ∈ (1, 2)`, then update the fraction to the boundary
`τ = max(τmin, 1 - μ)`. Returns the new `μ`.
"""
function update_barrier!(ϕ::LogBarrier{T}, x, tol; κμ = T(0.2), θμ = T(1.5)) where {T}
  @assert 0 < κμ < 1 && 1 < θμ < 2 "Need 0 < κμ < 1 and 1 < θμ < 2."
  μ = ϕ.μ
  ϕ.μ = max(tol / 10, min(κμ * μ, μ^θμ))
  set_fraction_to_boundary!(ϕ)
  return ϕ.μ
end

update_barrier!(::Nothing, x, tol; kwargs...) = nothing

@doc raw"""
    compute_mu_compl_error!(compl_res_l, compl_res_u, ϕ::LogBarrier, xk)

Perturbed complementarity error `max(‖Z_l (x - l) - μe‖∞, ‖Z_u (u - x) - μe‖∞)`,
using the bound multipliers stored in `ϕ`. Entries for infinite bounds are set to zero.
"""
function compute_mu_compl_error!(compl_res_l, compl_res_u, ϕ::LogBarrier, xk)
  _compl_residual!(compl_res_l, compl_res_u, ϕ, xk, ϕ.μ)
  return max(norm(compl_res_l, Inf), norm(compl_res_u, Inf))
end

@doc raw"""
    compute_compl_error!(compl_res_l, compl_res_u, ϕ::LogBarrier, xk)

Complementarity error `max(‖Z_l (x - l)‖∞, ‖Z_u (u - x)‖∞)`, using the bound multipliers
stored in `ϕ`. Entries for infinite bounds are set to zero.
"""
function compute_compl_error!(compl_res_l, compl_res_u, ϕ::LogBarrier{T}, xk) where {T}
  _compl_residual!(compl_res_l, compl_res_u, ϕ, xk, zero(T))
  return max(norm(compl_res_l, Inf), norm(compl_res_u, Inf))
end

function _compl_residual!(compl_res_l, compl_res_u, ϕ::LogBarrier, xk, μ)
  for i in eachindex(xk)
    l, u = ϕ.l[i], ϕ.u[i]
    compl_res_l[i] = isfinite(l) ? ϕ.z_l[i] * (xk[i] - l) - μ : zero(μ)
    compl_res_u[i] = isfinite(u) ? ϕ.z_u[i] * (u - xk[i]) - μ : zero(μ)
  end
  return compl_res_l, compl_res_u
end

compute_mu_compl_error!(compl_res_l, compl_res_u, ::Nothing, xk) = zero(eltype(xk))
compute_compl_error!(compl_res_l, compl_res_u, ::Nothing, xk) = zero(eltype(xk))

"""
    update_multipliers!(ϕ, s_z_l, s_z_u)

Take the (already truncated) step on the bound multipliers stored in `ϕ`.
"""
function update_multipliers!(ϕ::LogBarrier, s_z_l, s_z_u)
  ϕ.z_l .+= s_z_l
  ϕ.z_u .+= s_z_u
  return ϕ
end
update_multipliers!(::Nothing, s_z_l, s_z_u) = nothing

function set_barrier!(::Nothing) end
function set_barrier!(ϕ::LogBarrier, μ) where {T,S}
  ϕ.μ = μ
  set_fraction_to_boundary!(ϕ)
end

function compute_compl_ktol(ϕ::LogBarrier, κε)
  return κε * ϕ.μ
end
compute_compl_ktol(ϕ::Nothing, κε) = one(κε)

@doc raw"""
    truncate_to_boundary!(s, s_z_l, s_z_u, xk, ϕ::LogBarrier)

Apply the fraction-to-the-boundary rule (Wächter & Biegler, eq. (15)) so that the next
iterate stays strictly inside the bounds, and the bound multipliers stay strictly positive.

The largest primal step `α ∈ (0, 1]` is computed so that

    xk + α s - l ≥ (1 - τ)(xk - l)   and   u - xk - α s ≥ (1 - τ)(u - xk),

and, separately, the largest dual step `α_z ∈ (0, 1]` so that

    zk_L + α_z s_z_l ≥ (1 - τ) zk_L   and   zk_U + α_z s_z_u ≥ (1 - τ) zk_U,

where `τ = ϕ.τ`, `zk_L = ϕ.z_l` and `zk_U = ϕ.z_u`. Then `s` is scaled in place by `α`
and `s_z = (s_z_l, s_z_u)` is scaled in place by `α_z`. Infinite bounds are ignored. Returns `(α, α_z)`.
"""
function truncate_to_boundary!(s, s_z_l, s_z_u, xk, ϕ::LogBarrier{T}) where {T}
  τ = ϕ.τ
  zk_L, zk_U = ϕ.z_l, ϕ.z_u
  α = one(T)
  α_z = one(T)

  for i in eachindex(xk)
    l, u = ϕ.l[i], ϕ.u[i]

    if isfinite(l)
      # Primal: only a step towards l can hit the bound.
      if s[i] < 0
        α = min(α, -τ * (xk[i] - l) / s[i])
      end
      # Dual: only a decreasing multiplier can hit 0.
      if s_z_l[i] < 0
        α_z = min(α_z, -τ * zk_L[i] / s_z_l[i])
      end
    end

    if isfinite(u)
      # Primal: only a step towards u can hit the bound.
      if s[i] > 0
        α = min(α, τ * (u - xk[i]) / s[i])
      end
      if s_z_u[i] < 0
        α_z = min(α_z, -τ * zk_U[i] / s_z_u[i])
      end
    end
  end

  s .*= α
  s_z_l .*= α_z
  s_z_u .*= α_z

  return α, α_z
end

truncate_to_boundary!(s, s_z_l, s_z_u, xk, ::Nothing) = (one(eltype(s)), one(eltype(s)))

function (ϕ::LogBarrier)(x)
  val = zero(eltype(x))
  for i in eachindex(x)
    val += logterm(ϕ.u[i], ϕ.u[i] - x[i]) + logterm(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return ϕ.μ * val
end

function add_grad!(g, ϕ::LogBarrier, x)
  for i in eachindex(g)
    g[i] += ϕ.μ * invd(ϕ.u[i], ϕ.u[i] - x[i]) - ϕ.μ * invd(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return g
end

"`h = α * diag(z_l/(x-l) + z_u/(u-x))`"
function hess_diag!(h, ϕ::LogBarrier, x, α = one(ϕ.μ))
  for i in eachindex(h)
    h[i] = α * (ϕ.z_l[i] * invd(ϕ.l[i], x[i] - ϕ.l[i]) + ϕ.z_u[i] * invd(ϕ.u[i], ϕ.u[i] - x[i]))
  end
  return h
end

function get_z_l_step!(s_z_l, ϕ::Nothing, x, s_x) end
function get_z_u_step!(s_z_u, ϕ::Nothing, x, s_x) end

@doc raw"""
    get_z_l_step!(s_z_l, ϕ::LogBarrier, x, s_x)

Compute the primal-dual step for the lower-bound multipliers (Wächter & Biegler, eq. (12))
with slack ``x - l``:

    s_z_l = μ/(x-l) - z_l - z_l/(x-l) .* s_x

Entries for infinite lower bounds are set to zero.
"""
function get_z_l_step!(s_z_l, ϕ::LogBarrier, x, s_x)
  μ = ϕ.μ
  for i in eachindex(x)
    l = ϕ.l[i]
    if isfinite(l)
      d = inv(x[i] - l)
      s_z_l[i] = μ * d - ϕ.z_l[i] - ϕ.z_l[i] * d * s_x[i]
    else
      s_z_l[i] = zero(eltype(s_z_l))
    end
  end
  return s_z_l
end

@doc raw"""
    get_z_u_step!(s_z_u, ϕ::LogBarrier, x, s_x)

Compute the primal-dual step for the upper-bound multipliers (Wächter & Biegler, eq. (12))
with slack ``u - x``, whose step is ``-s_x``:

    s_z_u = μ/(u-x) - z_u + z_u/(u-x) .* s_x

Entries for infinite upper bounds are set to zero.
"""
function get_z_u_step!(s_z_u, ϕ::LogBarrier, x, s_x)
  μ = ϕ.μ
  for i in eachindex(x)
    u = ϕ.u[i]
    if isfinite(u)
      d = inv(u - x[i])
      s_z_u[i] = μ * d - ϕ.z_u[i] + ϕ.z_u[i] * d * s_x[i]
    else
      s_z_u[i] = zero(eltype(s_z_u))
    end
  end
  return s_z_u
end

isinterior(ϕ::LogBarrier, x) = all(i -> ϕ.l[i] < x[i] < ϕ.u[i], eachindex(x))
