mutable struct PenaltyLDLTWorkspace{
  WP<:LDLFactorization,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  VI<:Union{Nothing,AbstractVector},
  T<:Real,
  WPC<:Union{Nothing,LDLFactorization},
  KC<:Union{Nothing,Symmetric},
} <: AbstractLDLTWorkspace
  M::WP
  H::K2
  x::V
  dx::V
  r::V
  _ipiv::VI # For CompactBFGS LU factorization
  _info::Base.RefValue{BlasInt} # For CompactBFGS LU factorization
  σ::T
  n::Int
  m::Int
  status::Symbol
  _n_fact::Int
  _primal_diag::V # Preallocated buffer for up_lb_is_pos_def
  _primal_row_sum::V # Preallocated buffer for up_lb_is_pos_def
  _Hcheck::WPC # Preallocated, reused LDL factorization for up_lb_is_pos_def_exact!
  _Hcheck_mat::KC # Preallocated H + σI block for _Hcheck, values refreshed on each call
end

function get_H(
  solver_workspace::PenaltyLDLTWorkspace{WP,K2},
) where {T,WP,K2<:Symmetric{T,SparseMatrixCSC{T,Int}}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyLDLTWorkspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

"""
    build_up_lb_check(H::SparseMatrixCSC{T}, n) -> (S, Hp)

Preallocate an LDL factorization `S`, sized n×n, dedicated to the exact check
performed by `up_lb_is_pos_def_exact!`: factorizing H + σI (the leading
n×n block of the K2 matrix stored in `H`) on its own. Since `H` stores the
upper triangle of the K2 matrix in CSC format, the leading n×n block consists
exactly of the first `H.colptr[n+1] - 1` stored entries of `H`. `Hp` is the
preallocated `Symmetric` n×n matrix `S` is associated with -- refresh its values
in place (e.g. `Hp.data.nzval .= H.nzval[1:nnz(Hp.data)]`) before each
`ldl_factorize!(Hp, S)` call.
"""
function build_up_lb_check(H::SparseMatrixCSC{T,Ti}, n) where {T,Ti}
  nnz_primal = H.colptr[n+1] - 1
  Hp = SparseMatrixCSC{T,Ti}(
    n,
    n,
    H.colptr[1:(n+1)],
    H.rowval[1:nnz_primal],
    H.nzval[1:nnz_primal],
  )
  Hp = Symmetric(Hp, :U)
  S = ldl_analyze(Hp)
  return S, Hp
end

function construct_ldlt_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric{T,SparseMatrixCSC{T,Int}}}
  S = ldl_analyze(H)
  Scheck, Hp = build_up_lb_check(H.data, n)
  return PenaltyLDLTWorkspace(
    S,
    H,
    similar(u1),
    similar(u1),
    similar(u1),
    nothing,
    Ref{BlasInt}(0),
    zero(T),
    n,
    m,
    :uninitialized,
    0,
    zeros(T, n),
    zeros(T, n),
    Scheck,
    Hp,
  )
end

function construct_ldlt_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  S = ldl_analyze(H.H)
  return PenaltyLDLTWorkspace(
    S,
    H,
    similar(u1),
    similar(u1),
    similar(u1),
    Vector{LinearAlgebra.BlasInt}(undef, 2 * H.B._mem),
    Ref{BlasInt}(0),
    zero(T),
    n,
    m,
    :uninitialized,
    0,
    zeros(T, 0),
    zeros(T, 0),
    nothing,
    nothing,
  )
end

function update_workspace!(
  solver_workspace::PenaltyLDLTWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCSC}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @views H[1:n, 1:n] .= B'

  @inbounds for i = 1:n
    H[i, i] += σ
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] -= α
  end

  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyLDLTWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCOO}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @inbounds for i = 1:n
    H[i, i] = σ
  end

  @inbounds for i = 1:length(B.vals)
    if B.cols[i] == B.rows[i]
      H[B.cols[i], B.rows[i]] += B.vals[i]
    else
      H[B.cols[i], B.rows[i]] = B.vals[i]
    end
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end

  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function update_workspace!(solver_workspace::PenaltyLDLTWorkspace, A, σ, α)
  # Warning: Considers tht B is a zero matrix.
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  H.nzval .= 0

  @inbounds for i = 1:n
    H[i, i] = σ
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end

  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyLDLTWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:CompactBFGS}
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)

  @inbounds for i = 1:n
    H[i, i] = σ + B.ξ
  end

  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], n+A.rows[i]] = A.vals[i]
  end

  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyLDLTWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  @inbounds for i = 1:m
    H[n+i, n+i] = -α
  end
  solver_workspace.M.__factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyLDLTWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  σ_prev = solver_workspace.σ
  @inbounds for i = 1:n
    H[i, i] += σ - σ_prev
  end
  solver_workspace.σ = σ
  solver_workspace.M.__factorized = false
end

# Given Ax ≈ b, refine the solution by solving AΔx = b - Ax and updating x += Δx
function refine!(
  workspace::PenaltyLDLTWorkspace,
  u::V;
  max_iter::Int = 5,
  tol::T = eps(T),
) where {T,V<:AbstractVector{T}}

  # Compute the residual r = u - H*x
  r, H, x = workspace.r, workspace.H, workspace.x
  n, m = workspace.n, workspace.m
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

function solve_system!(
  workspace::PenaltyLDLTWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  if !factorized(workspace.M)
    ldl_factorize!(workspace.H, workspace.M)
    workspace._n_fact += 1
  end
  if !factorized(workspace.M)
    workspace.status = :failed
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
  workspace::PenaltyLDLTWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  n, m = workspace.n, workspace.m
  p = min(B._insert - 1, B._mem)
  x1, x2, x3, y1 = H.x1, H.x2, H.x3, H.y1
  Z1, Z2 = H.Z1, H.Z2
  Uk = @view B.Uk[:, 1:p]
  Vk = @view B.Vk[:, 1:p]

  # Step 0: Write
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + [-U V]([U V])ᵀ
  # [A -αI] = [A     -αI] + [ 0 0]([0 0])
  # Hence,
  # [B  Aᵀ] = [σI+ξI  Aᵀ] + EFᵀ
  # [A -αI] = [A     -αI] + EFᵀ

  # Step 1: Factorize
  # [σI+ξI  Aᵀ]
  # [A     -αI]
  if !factorized(workspace.M)
    ldl_factorize!(workspace.H.H, workspace.M)
    workspace._n_fact += 1
  end
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]
  ldiv!(x1, workspace.M, u)
  if any(isnan, x1)
    workspace.status = :failed
    return
  end

  if p > 0

    # Step 3: Compute
    # y₁ = Fᵀx₁ = [Uᵀx₁(1:n)]
    # y₁ = Fᵀx₁ = [Vᵀx₁(1:n)]
    @views mul!(y1[1:p], Uk', x1[1:n])
    @views mul!(y1[(p+1):(2*p)], Vk', x1[1:n])


    # Step 4: Assemble Schur complement (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )
    #                                   (       [A     -αI]     )
    # Step 4.1: Compute 
    # Z₁ = [σI+ξI  Aᵀ]⁻¹ E = [σI+ξI  Aᵀ]⁻¹[-U V]
    # Z₁ = [A     -αI]   E = [A     -αI]  [ 0 0]
    Z1 .= 0

    @views Z1[1:n, 1:p] .= Uk .* (-1)
    @views Z1[1:n, (p+1):(2*p)] .= Vk
    ldiv!(workspace.M, Z1)
    if any(isnan, Z1)
      workspace.status = :failed
      return
    end

    # Step 4.2: Compute 
    # Z₂ = FᵀZ₁ = UᵀZ₁[1:n]
    # Z₂ = FᵀZ₁ = VᵀZ₁[1:n]
    Z2 .= 0
    @views mul!(Z2[1:p, 1:(2*p)], Uk', Z1[1:n, (1:(2*p))])
    @views mul!(Z2[(p+1):(2*p), 1:(2*p)], Vk', Z1[1:n, (1:(2*p))])

    # Step 4.3: Compute 
    # Z₂ = I + Z₂
    for i = 1:(2*p)
      Z2[i, i] += 1
    end

    # Step 5: Solve
    # (I + Fᵀ [σI+ξI  Aᵀ]⁻¹ E )⁻¹[y₁]
    # (       [A     -αI]     )  [y₁]
    # using LAPACK
    @views info_f =
      getrf!(BlasInt(2p), BlasInt(2p), Z2, stride(Z2, 2), workspace._ipiv, workspace._info)
    if info_f != 0
      workspace.status = :failed
      return
    end

    @views info_s = getrs!(
      'N',
      BlasInt(2p),
      BlasInt(1),
      Z2[1:(2p), 1:(2p)],
      stride(Z2, 2),
      workspace._ipiv,
      y1[1:(2p)],
      BlasInt(2p),
      workspace._info,
    )
    if info_s != 0
      workspace.status = :failed
      return
    end
    if any(isnan, @view y1[1:(2*p)])
      workspace.status = :failed
      return
    end

    # Step 6: Compute
    # x₂ = E[y₂] = [-U V][y₂] = [-Uy₂ + Vy₂]
    # x₂ = E[y₂] = [ 0 0][y₂] = [0]
    @views mul!(x2[1:n], Vk, y1[(p+1):(2*p)])
    @views mul!(x2[1:n], Uk, y1[1:p], -one(eltype(y1)), one(eltype(y1)))

    # Step 7: Solve
    # [x₃] = [σI+ξI  Aᵀ]⁻¹[x₂]
    # [x₃] = [A     -αI]  [x₂]
    ldiv!(x3, workspace.M, x2)
    if any(isnan, x3)
      workspace.status = :failed
      return
    end

    # Step 8:
    # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
    # [A -αI]  [u] = x₁ - x₃
    workspace.x .= x1 .- x3
  else
    workspace.x .= x1
  end
end

function get_solution!(x::V, workspace::PenaltyLDLTWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyLDLTWorkspace)
  return workspace.status
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

function up_lb_is_pos_def(workspace::PenaltyLDLTWorkspace, ::Symmetric)
  n, m = workspace.n, workspace.m

  # Step 1: check inertia
  npos, nzero, nneg = get_inertia(workspace)
  status = get_status(workspace)
  (status == :failed || npos != n || nneg != m) && return false

  # Step 2 & 3: cheap sufficient/necessary conditions:
  # 1. Check if the diagonal has a negative entry,
  # 2. Check if the matrix is diagonally dominant.
  d, s = primal_diagonal_and_row_sums(workspace)
  is_diagonally_dominant = true
  for i = 1:n
    d[i] <= 0 && return false
    is_diagonally_dominant &= d[i] >= s[i]
  end
  is_diagonally_dominant && return true

  # Step 4: fallback to Cholesky factorization.
  return up_lb_is_pos_def_exact!(workspace)
end

# Perform LDLᵀ facto of H + σI to check positive definiteness.
# LDLFactorizations.jl does not pivot (and no dynamic regularization is set on
# _Hcheck), so H + σI is positive definite iff the factorization succeeds with
# all pivots positive.
function up_lb_is_pos_def_exact!(workspace::PenaltyLDLTWorkspace)
  H = get_H(workspace)
  Hp = workspace._Hcheck_mat
  copyto!(Hp.data.nzval, 1, H.nzval, 1, length(Hp.data.nzval))

  S = workspace._Hcheck
  ldl_factorize!(Hp, S)

  factorized(S) || return false
  return all(>(0), S.d)
end
