mutable struct PenaltyLDLTWorkspace{
  WP<:LDLFactorization,
  K2<:AbstractK2Matrix,
  V<:AbstractVector,
} <: AbstractLDLTWorkspace{WP, K2}
  M::WP
  H::K2
  x::V
  dx::V
  r::V
  status::Symbol
  _n_fact::Int
end

function construct_ldlt_workspace(
  K::AbstractK2Matrix{T,M1,M2,M3},
  u1,
  n,
  m,
) where {T,M1,M2,M3<:SparseSymmetricCSC{T}}
  S = ldl_analyze(K.H)
  return PenaltyLDLTWorkspace(
    S,
    K,
    similar(u1),
    similar(u1),
    similar(u1),
    :uninitialized,
    0,
  )
end

factorized(workspace::PenaltyLDLTWorkspace) = LDLFactorizations.factorized(workspace.M)

function invalidate_factorization!(workspace::PenaltyLDLTWorkspace)
  workspace.M.__factorized = false
end

# Given Ax ≈ b, refine the solution by solving AΔx = b - Ax and updating x += Δx
function refine!(
  workspace::PenaltyLDLTWorkspace,
  u::V;
  max_iter::Int = 5,
  tol::T = eps(T),
) where {T,V<:AbstractVector{T}}

  # Compute the residual r = u - H*x
  r, H, x = workspace.r, workspace.H.H, workspace.x
  n, m = get_n(workspace), get_m(workspace)
  dx = workspace.dx
  solved = false
  k = 0
  while k < max_iter && !solved
    r .= u

    # https://github.com/JuliaSparse/SparseArrays.jl/issues/685: mul!(r, H, x, -one(T), one(T)) is somehow extremely slow.
    mul!(r, H.data, x, -one(T), one(T)) # r = u - H*x
    mul!(r, H.data', x, -one(T), one(T))
    @inbounds for i = 1:(n+m)
      r[i] += H.data[i, i]*x[i]
    end

    ldiv!(dx, workspace.M, r) # H*dx = r
    x .+= dx
    k = k + 1
    solved = norm(dx) < tol*norm(x)
  end
  return workspace.x
end

function factorize!(
  workspace::PenaltyLDLTWorkspace
)
  factorized(workspace) && return
  ldl_factorize!(workspace.H.H, workspace.M)
  set_n_fact!(workspace, get_n_fact(workspace) + 1)

  if !factorized(workspace)
    set_status!(workspace, :failed)
  end
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace,
  u::AbstractVector,
)
  set_status!(workspace, :success)

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  ldiv!(workspace.x, workspace.M, u)

  if any(isnan, workspace.x)
    workspace.status = :failed
    return
  end

  refine!(workspace, u)
  if any(isnan, workspace.x) ||
     norm(workspace.dx)/norm(workspace.x) > eps(eltype(workspace.x))^(0.5)
    workspace.status = :failed
    return
  end
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace,
  Z::AbstractMatrix,
)
  set_status!(workspace, :success)

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  ldiv!(workspace.M, Z)
  if any(isnan, Z)
    set_status!(workspace, :failed)
    return
  end
end

function get_inertia(workspace::PenaltyLDLTWorkspace)
  LDL = workspace.M

  n = LDL.n
  (npos, nzero, nneg) = (0, 0, 0)

  D = LDL.d
  for i = 1:n
    d = D[i]
    if real(d) > 0
      npos += 1
    elseif abs(real(d)) < eps(eltype(d))
      nzero += 1
    else
      nneg += 1
    end
  end

  return npos, nzero, nneg
end
