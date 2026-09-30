abstract type AbstractBarrier end

@doc raw"""
    LogBarrier(μ, l, u)

Logarithmic barrier for the bounds `l ≤ x ≤ u`:

```math
ϕ(x) = -μ \sum_i [\log(u_i - x_i) + \log(x_i - l_i)],
```
where `μ > 0`. Infinite bounds are ignored, and `ϕ(x) = Inf` if `x` is not strictly feasible.

The barrier is purely primal: no bound multipliers are stored. Whenever they are needed,
they are the implicit multipliers `z_l = μ (X - L)⁻¹ e` and `z_u = μ (U - X)⁻¹ e`
(see [`bound_multipliers!`](@ref)), for which the perturbed complementarity
`Z_l (x - l) = Z_u (u - x) = μ e` holds exactly.
"""
mutable struct LogBarrier{T<:Real,V<:AbstractVector{T}} <: AbstractBarrier
  μ::T
  τ::T # Fraction to boundary
  τmin::T # Minimum fraction to boundary
  l::V
  u::V

  function LogBarrier(μ::T, l::V, u::V; τmin = T(0.99)) where {T<:Real,V<:AbstractVector{T}}
    @assert μ > 0 "Barrier parameter μ must be positive."
    @assert length(l) == length(u) && all(l .< u) "Bounds must satisfy l < u."
    τ = max(τmin, 1 - μ)
    return new{T,V}(μ, τ, τmin, l, u)
  end
end

@inline logterm(b, d) = isfinite(b) ? (d > 0 ? -log(d) : oftype(d, Inf)) : zero(d)
@inline invd(b, d) = isfinite(b) ? inv(d) : zero(d)

function set_fraction_to_boundary!(::Nothing) end
function set_fraction_to_boundary!(ϕ::LogBarrier)
  ϕ.τ = max(ϕ.τmin, 1 - ϕ.μ)
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
    Δf = update_barrier!(g, ϕ::LogBarrier, x, tol; kwargs...)

Decrease the barrier parameter from `μ` to `μ₊` (see `update_barrier!(ϕ, x, tol)`) and
correct in place the gradient `g` of a model whose objective contains `ϕ`, without
evaluating the rest of the objective. With `ϕ(x) = μ B(x)`:

    g ← g + (μ₊ - μ) ∇B(x),

and the objective correction `Δf = (μ₊ - μ) B(x)` is returned.

The Hessian block of `ϕ` is primal (see `hess_diag!`) and scales with `μ`: it is *not*
corrected here, the caller is responsible for re-evaluating the Hessian of the model.
"""
function update_barrier!(g, ϕ::LogBarrier{T}, x, tol; kwargs...) where {T}
  μ = ϕ.μ
  Δμ = update_barrier!(ϕ, x, tol; kwargs...) - μ
  B = zero(T)
  for i in eachindex(x)
    l, u = ϕ.l[i], ϕ.u[i]
    B += logterm(u, u - x[i]) + logterm(l, x[i] - l)
    g[i] += Δμ * (invd(u, u - x[i]) - invd(l, x[i] - l))
  end
  return Δμ * B
end

update_barrier!(g, ::Nothing, x, tol; kwargs...) = zero(eltype(x))

@doc raw"""
    bound_multipliers!(z_l, z_u, ϕ::LogBarrier, x)

Compute the implicit bound multipliers of the primal barrier,

    z_l = μ (X - L)⁻¹ e   and   z_u = μ (U - X)⁻¹ e,

for which `∇f(x) + ∇ϕ(x) = ∇f(x) - z_l + z_u` and `Z_l (x - l) = Z_u (u - x) = μ e`.
Entries for infinite bounds are set to zero.
"""
function bound_multipliers!(z_l, z_u, ϕ::LogBarrier, x)
  μ = ϕ.μ
  for i in eachindex(x)
    l, u = ϕ.l[i], ϕ.u[i]
    z_l[i] = μ * invd(l, x[i] - l)
    z_u[i] = μ * invd(u, u - x[i])
  end
  return z_l, z_u
end

bound_multipliers!(z_l, z_u, ::Nothing, x) = (fill!(z_l, 0), fill!(z_u, 0))

function set_barrier!(::Nothing) end
function set_barrier!(ϕ::LogBarrier, μ) where {T,S}
  ϕ.μ = μ
  set_fraction_to_boundary!(ϕ)
end

function get_barrier(ϕ::LogBarrier)
  return ϕ.μ
end

"""
    get_barrier_parameter(ϕ)

Return `μ`, or zero if there is no barrier. With the implicit bound multipliers
(see [`bound_multipliers!`](@ref)), `μ` is exactly the complementarity residual
`max(‖Z_l (x - l)‖∞, ‖Z_u (u - x)‖∞)` of the original problem.
"""
get_barrier_parameter(ϕ::LogBarrier) = ϕ.μ
get_barrier_parameter(::Nothing) = false

"""
    compute_barrier_ktol(ϕ, κε)

Tolerance `κε μ` on the dual feasibility of the barrier subproblem (Wächter & Biegler, eq. (7)).
"""
compute_barrier_ktol(ϕ::LogBarrier, κε) = κε * ϕ.μ
compute_barrier_ktol(::Nothing, κε) = zero(κε)

@doc raw"""
    truncate_to_boundary!(s, xk, ϕ::LogBarrier)

Apply the fraction-to-the-boundary rule (Wächter & Biegler, eq. (15)) so that the next
iterate stays strictly inside the bounds. The largest `α ∈ (0, 1]` is computed so that

    xk + α s - l ≥ (1 - τ)(xk - l)   and   u - xk - α s ≥ (1 - τ)(u - xk),

where `τ = ϕ.τ`, and `s` is scaled in place by `α`. Infinite bounds are ignored. Returns `α`.
"""
function truncate_to_boundary!(s, xk, ϕ::LogBarrier{T}) where {T}
  τ = ϕ.τ
  α = one(T)

  for i in eachindex(xk)
    l, u = ϕ.l[i], ϕ.u[i]
    # Only a step towards l (resp. u) can hit the bound.
    if isfinite(l) && s[i] < 0
      α = min(α, -τ * (xk[i] - l) / s[i])
    end
    if isfinite(u) && s[i] > 0
      α = min(α, τ * (u - xk[i]) / s[i])
    end
  end

  s .*= α
  return α
end

truncate_to_boundary!(s, xk, ::Nothing) = one(eltype(s))

function (ϕ::LogBarrier)(x)
  val = zero(eltype(x))
  for i in eachindex(x)
    val += logterm(ϕ.u[i], ϕ.u[i] - x[i]) + logterm(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return ϕ.μ * val
end

function add_grad!(g, ϕ::LogBarrier, x; α = one(ϕ.μ))
  for i in eachindex(g)
    g[i] += α * ϕ.μ * invd(ϕ.u[i], ϕ.u[i] - x[i]) - α * ϕ.μ * invd(ϕ.l[i], x[i] - ϕ.l[i])
  end
  return g
end

"`h = α * μ * diag(1/(x-l)² + 1/(u-x)²)`, the (primal) Hessian of `ϕ`."
function hess_diag!(h, ϕ::LogBarrier, x, α = one(ϕ.μ))
  μ = ϕ.μ
  for i in eachindex(h)
    dl = invd(ϕ.l[i], x[i] - ϕ.l[i])
    du = invd(ϕ.u[i], ϕ.u[i] - x[i])
    h[i] = α * μ * (dl^2 + du^2)
  end
  return h
end

isinterior(ϕ::LogBarrier, x) = all(i -> ϕ.l[i] < x[i] < ϕ.u[i], eachindex(x))
