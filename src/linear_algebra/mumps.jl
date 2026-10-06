mutable struct PenaltyMUMPSWorkspace{
  WP<:Mumps,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  VI<:Union{Nothing,AbstractVector},
  T<:Real,
  WPC<:Union{Nothing,Mumps},
} <: AbstractMUMPSWorkspace
  M::WP
  H::K2
  x::V
  _ipiv::VI # For CompactBFGS LU factorization
  _info::Base.RefValue{BlasInt}  # For CompactBFGS LU factorization
  σ::T
  n::Int
  m::Int
  status::Symbol
  factorized::Bool
  _n_fact::Int
  _primal_diag::V # Preallocated buffer for up_lb_is_pos_def
  _primal_row_sum::V # Preallocated buffer for up_lb_is_pos_def
  _Hcheck::WPC # Preallocated, reused MUMPS instance for up_lb_is_pos_def_exact!
  _Hcheck_idx::Vector{Int} # Fixed indices into H.data.vals for the H + σI block
  _Hcheck_a::V # Preallocated values buffer for _Hcheck, refreshed on each call
  _n_precise::Int # Counter for how many times the relative error has been below threshold
end

function get_H(
  solver_workspace::PenaltyMUMPSWorkspace{WP,K2},
) where {T,M,WP,K2<:Symmetric{T,M}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

"""
    build_up_lb_check(H::SparseMatrixCOO{T}, n) -> (S, idx, a)

Preallocate a MUMPS instance `S`, sized n×n, dedicated to the exact check
performed by `up_lb_is_pos_def_exact!`: factorizing H + σI (the leading
n×n block of the K2 matrix stored in `H`) on its own. `idx` are the fixed
indices into `H.vals` for the entries of that block (its sparsity pattern
does not change across calls, only the values do), and `a` is the
preallocated values buffer `S` is associated with -- refresh it in place
(e.g. `a .= H.vals[idx]`) before each `factorize!(S)` call.
"""
function build_up_lb_check(H::SparseMatrixCOO{T}, n) where {T}
  idx = findall(k -> H.rows[k] <= n && H.cols[k] <= n, eachindex(H.vals))
  irn = Int32.(H.rows[idx])
  jcn = Int32.(H.cols[idx])
  a = Vector{T}(undef, length(idx))

  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl
  cntl[1] = eps(T)
  icntl[2], icntl[3], icntl[4] = 0, 0, 0
  # CNTL(13) controls the parallelism of the root node; needed to get a
  # reliable inertia from INFOG(12) (see construct_mumps_workspace).
  icntl[13] = 1
  # ICNTL(24): null pivot row detection.
  icntl[24] = 1

  S = Mumps{T}(mumps_symmetric, icntl, cntl, backend = MUMPS.Sequential())
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = n
  S.nnz = length(idx)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  return S, idx, a
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters

  # CNTL(1) is the relative threshold for numerical pivoting.
  # Remarks: It forms a trade-off between preserving sparsity and ensuring numerical stability during
  # the factorization. In general, a larger value of CNTL(1) increases fill-in but leads to a more accurate
  # factorization.
  cntl[1] = eps(T)

  cntl[2] = eps(T) # Tolerance for iterative refinement

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 5

  # ICNTL(11): error analysis
  # 2: Main statistics (recommended)
  icntl[11] = 2

  # ICNTL(24) controls the detection of “null pivot rows”.
  # 1: Null pivot row detection.
  icntl[24] = 1

  # CNTL(13) controls the parallelism of the root node
  # Remarks: Processing the root sequentially (ICNTL(13) > 0) can be useful when the user is
  # interested in the inertia of the matrix (see INFO(12) and INFOG(12)), or when the user wants
  # to detect null pivots (see Subsection 5.13) or to activate BLR compression (Subsection 5.20) on the
  # root node.
  icntl[13] = 1

  S = Mumps{T}(mumps_symmetric, icntl, cntl, backend = MUMPS.Sequential())

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.data.rows, H.data.cols, H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  Scheck, idx, a_check = build_up_lb_check(H.data, n)

  return PenaltyMUMPSWorkspace(
    S,
    H,
    x,
    nothing,
    Ref{BlasInt}(0),
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
    zeros(T, n),
    zeros(T, n),
    Scheck,
    idx,
    a_check,
    0,
  )
end

function construct_mumps_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  # Set params : TODO
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters
  # CNTL(1) is the relative threshold for numerical pivoting.
  # Remarks: It forms a trade-off between preserving sparsity and ensuring numerical stability during
  # the factorization. In general, a larger value of CNTL(1) increases fill-in but leads to a more accurate
  # factorization.
  cntl[1] = eps(T)

  cntl[2] = eps(T) # Tolerance for iterative refinement

  # Deactivate Logging
  icntl[2], icntl[3], icntl[4] = 0, 0, 0

  # Max number of iterative refinement steps
  icntl[10] = 5

  # ICNTL(11): error analysis
  # 2: Main statistics (recommended)
  icntl[11] = 2

  # CNTL(13) controls the parallelism of the root node
  # Remarks: Processing the root sequentially (ICNTL(13) > 0) can be useful when the user is
  # interested in the inertia of the matrix (see INFO(12) and INFOG(12)), or when the user wants
  # to detect null pivots (see Subsection 5.13) or to activate BLR compression (Subsection 5.20) on the
  # root node.
  icntl[13] = 1

  # ICNTL(24) controls the detection of “null pivot rows”.
  # 1: Null pivot row detection.
  icntl[24] = 1

  S = Mumps{T}(mumps_symmetric, icntl, cntl, backend = MUMPS.Sequential())

  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.H.data.rows, H.H.data.cols, H.H.data.vals
  S.irn, S.jcn, S.a = pointer.((irn, jcn, a))
  S.n = m+n
  S.nnz = length(irn)
  S._irn_gc_haven = irn
  S._jcn_gc_haven = jcn
  S._a_gc_haven = a

  # Associate the size and number of the right hand side
  x = similar(u1)
  S.lrhs = n + m
  S.nrhs = 1
  S.rhs = pointer(x)
  S._y_gc_haven = x

  return PenaltyMUMPSWorkspace(
    S,
    H,
    x,
    Vector{LinearAlgebra.BlasInt}(undef, 2 * H.B._mem),
    Ref{BlasInt}(0),
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
    zeros(T, 0),
    zeros(T, 0),
    nothing,
    Int[],
    zeros(T, 0),
    0,
  )
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:SparseMatrixCOO}
  n, m = solver_workspace.n, solver_workspace.m
  nnz_B, nnz_A = length(B.vals), length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_B] .= B.vals
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(solver_workspace::PenaltyMUMPSWorkspace, A, σ, α)
  # Warning: Considers tht B is a zero matrix.
  n, m = solver_workspace.n, solver_workspace.m
  nnz_A = length(A.vals)
  H = get_H(solver_workspace)
  nnz_B = length(H.vals) - nnz_A - n - m

  H.vals .= 0
  H.vals[(nnz_B+1):(nnz_B+nnz_A)] .= A.vals
  H.vals[(nnz_B+nnz_A+1):(nnz_B+nnz_A+n)] .= σ
  H.vals[(nnz_B+nnz_A+n+1):(nnz_B+nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(
  solver_workspace::PenaltyMUMPSWorkspace,
  B::M,
  A,
  σ,
  α,
) where {M<:CompactBFGS}
  n, m = solver_workspace.n, solver_workspace.m
  nnz_A = length(A.vals)

  H = get_H(solver_workspace)

  H.vals[1:nnz_A] .= A.vals
  H.vals[(nnz_A+1):(nnz_A+n)] .= σ + B.ξ
  H.vals[(nnz_A+n+1):(nnz_A+n+m)] .= -α

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMUMPSWorkspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m+1):end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMUMPSWorkspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success
  mumps, H = workspace.M, workspace.H

  if !workspace.factorized
    job = mumps.job
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)
    workspace._n_fact += 1

    k, max_iter = 0, 5
    # MUMPS Documentation - infog(1) = -9
    # The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number
    # of entries that are missing in S at the moment when the error is raised is available in INFO(2).
    # If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error –9
    # occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=
    # 2) again, except if LWK USER is provided LWK USER should be increased.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      workspace._n_fact += 1
      k = k + 1
    end

    # MUMPS infog(1): a negative value is an error in the factorization.
    if mumps.infog[1] < 0
      workspace.status = :failed
      return
    else
      workspace.factorized = true
    end
  end

  workspace.x .= u
  MUMPS.mumps_solve!(workspace.x, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, workspace.x) || mumps.infog[1] < 0
    workspace.status = :failed
  end

  update_pivtol!(workspace)

  return
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  mumps = workspace.M
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

  if !workspace.factorized
    job = mumps.job
    mumps.job = MUMPS.INITIALIZE
    factorize!(mumps)
    workspace._n_fact += 1

    k, max_iter = 0, 5
    # MUMPS Documentation - infog(1) = -9
    # The main internal real/complex workarray S is too small. If INFO(2) is positive, then the number
    # of entries that are missing in S at the moment when the error is raised is available in INFO(2).
    # If INFO(2) is negative, then its absolute value should be multiplied by 1 million. If an error –9
    # occurs, the user should increase the value of ICNTL(14) before calling the factorization (JOB=
    # 2) again, except if LWK USER is provided LWK USER should be increased.
    while mumps.infog[1] == -9 && k < max_iter
      MUMPS.set_icntl!(mumps, 14, mumps.icntl[14] * 2)
      mumps.job = MUMPS.FACTOR
      factorize!(mumps)
      workspace._n_fact += 1
      k = k + 1
    end

    # MUMPS infog(1): a negative value is an error in the factorization.
    if mumps.infog[1] < 0
      workspace.status = :failed
      return
    else
      workspace.factorized = true
    end
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]

  x1 .= u
  MUMPS.associate_rhs!(mumps, x1; unsafe = true)
  MUMPS.mumps_solve!(x1, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, x1) || mumps.infog[1] < 0
    workspace.status = :failed
    return
  end

  update_pivtol!(workspace)

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

    MUMPS.associate_rhs!(mumps, Z1; unsafe = true)
    MUMPS.mumps_solve!(Z1, mumps; rhs_changed = true)

    # MUMPS infog(1): a negative value is an error in the factorization.
    if any(isnan, Z1) || mumps.infog[1] < 0
      workspace.status = :failed
      return
    end

    update_pivtol!(workspace)

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
    x3 .= x2
    MUMPS.associate_rhs!(mumps, x3; unsafe = true)
    MUMPS.mumps_solve!(x3, mumps; rhs_changed = true)

    # MUMPS infog(1): a negative value is an error in the factorization.
    if any(isnan, x3) || mumps.infog[1] < 0
      workspace.status = :failed
      return
    end

    update_pivtol!(workspace)

    # Step 8:
    # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
    # [A -αI]  [u] = x₁ - x₃
    workspace.x .= x1 .- x3
  else
    workspace.x .= x1
  end
end

function get_solution!(x::V, workspace::PenaltyMUMPSWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyMUMPSWorkspace)
  return workspace.status
end

function get_inertia(workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2}

  n, m = workspace.n, workspace.m
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.infog[12]
  rank = n + m - workspace.M.infog[28]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end

function relative_error!(workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2}
  mumps = workspace.M

  return max(mumps.rinfog[6], mumps.rinfog[7], mumps.rinfog[8])
end

function increase_pivtol!(workspace::PenaltyMUMPSWorkspace)
  mumps = workspace.M

  MUMPS.set_cntl!(mumps, 1, 1e-2)
  MUMPS.set_icntl!(mumps, 10, -5)
end

function decrease_pivtol!(workspace::PenaltyMUMPSWorkspace)
  mumps = workspace.M

  MUMPS.set_cntl!(mumps, 1, max(mumps.cntl[1] / 10, eps(eltype(workspace.x))))
  MUMPS.set_icntl!(mumps, 10, 5)
end

function update_pivtol!(workspace::PenaltyMUMPSWorkspace)
  mumps = workspace.M

  relative_error = relative_error!(workspace)
  if relative_error > sqrt(eps(eltype(workspace.x)))
    increase_pivtol!(workspace)
    workspace.status = :failed
    workspace._n_precise = 0
  elseif relative_error < eps(eltype(workspace.x)) * 100
    workspace._n_precise += 1
    if workspace._n_precise > 20
      decrease_pivtol!(workspace)
    end
  else
    workspace._n_precise = 0
  end
end

function SolverCore.reset!(workspace::PenaltyMUMPSWorkspace)
  set_n_fact!(workspace, 0)
  MUMPS.set_icntl!(workspace.M, 10, 5)
  MUMPS.set_cntl!(workspace.M, 1, eps(eltype(workspace.x)))
end

function up_lb_is_pos_def(workspace::PenaltyMUMPSWorkspace, ::Symmetric)
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

# Perform Cholesky facto of H + σI to check positive definiteness.
function up_lb_is_pos_def_exact!(workspace::PenaltyMUMPSWorkspace)
  n = workspace.n
  H = get_H(workspace)
  idx = workspace._Hcheck_idx
  a = workspace._Hcheck_a
  @inbounds for k in eachindex(idx)
    a[k] = H.vals[idx[k]]
  end

  S = workspace._Hcheck
  S.job = MUMPS.INITIALIZE
  factorize!(S)

  is_pos_def = if S.infog[1] < 0
    false
  else
    nneg = S.infog[12]
    rank = n - S.infog[28]
    npos = rank - nneg
    npos == n
  end

  return is_pos_def
end
