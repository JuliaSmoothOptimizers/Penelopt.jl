"""
    extrapolate!(x, solver, τ₂, τ₁) -> Bool

Extrapolate the approximate solution `(x, y)` of the penalized subproblem

    min_x f(x) + τ₁ ‖c(x)‖₂

to a first-order estimate of the solution for the new penalty parameter `τ₂ > τ₁`,
following the implicit-function-theorem argument of the section "Extrapolation
Technique" of the implementation paper.

# Background

Stationary points of the penalized problem satisfy `F(x, y, α, τ) = 0`, where

    F(x, y, α, τ) = [ ∇f(x) + J(x)ᵀ y ]
                    [ c(x) - α y      ]
                    [ α (‖y‖₂² - τ²)  ].

Differentiating along the path `τ ↦ (x_τ, y_τ, α_τ)` gives the 3×3 block system

    [ H   Jᵀ    0  ] [ ẋ ]   [ 0    ]
    [ J  -αI   -y  ] [ ẏ ] = [ 0    ]
    [ 0  2αyᵀ   d  ] [ α̇']   [ 2ατ  ],     d = ‖y‖₂² - τ²,

where `H = ∇²ₓₓL(x, y)`.

# Reduction to a 2×2 system

Let `K = [H Jᵀ; J -αI]`. The first two block rows read

    K [ẋ; ẏ] = α̇'[0; y],

so if `p = (px, py)` solves

    K p = [0; y],

then `(ẋ, ẏ) = α̇'(px, py)`. Substituting into the third row yields

    α̇'= 2ατ / (2α yᵀpy + d).

The extrapolated point is

    x_extr = x + (τ₂ - τ₁) ẋ,       y_extr = y + (τ₂ - τ₁) ẏ.

# Choice of τ₁

If `‖y‖₂ < τ₁`, complementarity forces `α = 0` and the solution does not move
with `τ` (`ẋ = 0`), so no extrapolation is performed. Otherwise,
Otherwise, `τ₁` is replaced by `‖y‖₂`. The first two components of `F` do not
depend on `τ`, and the third vanishes at `τ = ‖y‖₂`, so

    ‖F(x, y, α, ‖y‖₂)‖₂ ≤ ‖F(x, y, α, τ)‖₂   for all τ.

Hence `(x, y, α)` is closest to the solution path at `τ = ‖y‖₂`, and we
extrapolate from `‖y‖₂` to `τ₂`. If `‖y‖₂ ≥ τ₂`, no extrapolation is performed.

# Arguments

- `x::V`: current outer iterate.
- `solver::L2PenaltySolver`: solver workspace.
- `τ₂::T`: new penalty parameter.
- `τ₁::T`: penalty parameter for which `(x, y)` was computed.

# Returns

`true` if the extrapolation was applied, `false` otherwise.

# Side effects

On success, overwrites `solver.x`, `solver.y`, `solver.cn` and
`solver.substats.solver_specific[:smooth_obj]`.
"""
function extrapolate!(
  x::V,
  solver::L2PenaltySolver{T,V,S,PB},
  τ₂::T,
  τ₁::T,
) where {T,V,S,N<:AbstractNLPModel{T,V},PB<:L2PenalizedProblem{T,V,N}}

  # Step 0: Retrieve workspace
  r2n_solver = solver.subsolver
  ms_solver, ms_stats = r2n_solver.subsolver, r2n_solver.substats
  mk = r2n_solver.subpb
  φ, ψ, nlp = mk.model, mk.h, mk.parent.model
  y, α = solver.y, ms_stats.solver_specific[:alpha]
  n, m = nlp.meta.nvar, nlp.meta.ncon

  # Step 1: Check if extrapolation is needed and update τ₁ if necessary.
  # If ‖y‖₂ < τ₁, then no extrapolation is needed.
  norm_y = norm(y, 2)
  norm_y < τ₁ && return false
  τ₁ = norm_y

  # Step 1.1: Check if τ₂ is smaller than τ₁, in which case we cannot extrapolate.
  τ₁ >= τ₂ && return false

  # Step 2: Solve K p = [0; y]
  update_workspace!(ms_solver.workspace, φ.data.H, ψ.A, zero(T), α)

  # [ H  Jᵀ ][px] = [0]
  # [ J -αI ][py] = [y] 
  @views ms_solver.u2[1:n] .= zero(T)
  @views @. ms_solver.u2[(n+1):(n+m)] = y
  solve_system!(ms_solver.workspace, ms_solver.u2)
  get_solution!(ms_solver.x2, ms_solver.workspace)
  @views px, py = ms_solver.x2[1:n], ms_solver.x2[(n+1):(n+m)]

  # Step 2.1: Check inertia and safeguard the solution.
  npos, nzero, nneg = get_inertia(ms_solver.workspace)
  status = get_status(ms_solver.workspace)
  if status == :failed || npos != n || nneg != m || nzero != 0
    return false
  end

  # Step 3: Compute α' = 2ατ / (2α yᵀpy + ‖y‖₂² - τ²).
  α_dot = 2 * α * τ₁ / (2 * α * dot(y, py) + norm_y^2 - τ₁^2)

  # Step 4: Compute x_extr = x + (τ₂ - τ₁) ẋ = x + (τ₂ - τ₁) α' px
  px .*= α_dot
  x_extrap = px
  x_extrap .= x .+ (τ₂ - τ₁) .* px

  # Step 5: Basic safeguard
  obj_extrap = obj(nlp, x_extrap)
  cons!(nlp, x_extrap, solver.cn)

  if obj_extrap < Inf && norm(solver.cn) < Inf
    solver.x .= x_extrap
    set_solver_specific!(solver.substats, :smooth_obj, obj_extrap)
    # Step 6: Update y_extr = y + (τ₂ - τ₁) ẏ = y + (τ₂ - τ₁) α' py
    solver.y .+= ((τ₂ - τ₁) * α_dot) .* py
    return true
  end

  return false
end
