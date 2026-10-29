mutable struct PenaltyMUMPSWorkspace{WP<:Mumps,K2<:AbstractK2Matrix,V<:AbstractVector,Opt<:PenaltyLinearSolverOptions} <:
               AbstractMUMPSWorkspace
  M::WP
  H::K2
  x::V
  status::Symbol
  factorized::Bool
  _n_fact::Int
  _n_precise::Int # Counter for how many times the relative error has been below threshold
  options::Opt
end

function construct_mumps_workspace(
  K::AbstractK2Matrix{T,M1,M2,M3},
  u1,
  n,
  m;
  min_pivtol::T = eps(T),
  initial_pivtol::T = eps(T),
  max_pivtol::T = T(1e-2),
  decrease_pivtol_factor::T = T(0.1),
  kwargs...
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  cntl = T == Float64 ? default_cntl64 : default_cntl32
  icntl = default_icntl

  ## Set Parameters

  # CNTL(1) is the relative threshold for numerical pivoting.
  # Remarks: It forms a trade-off between preserving sparsity and ensuring numerical stability during
  # the factorization. In general, a larger value of CNTL(1) increases fill-in but leads to a more accurate
  # factorization.
  cntl[1] = initial_pivtol

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

  H = get_H(K)
  # Associate the row, cols and vals of the mumps structure with those of H.
  irn, jcn, a = H.rows, H.cols, H.vals
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

  return PenaltyMUMPSWorkspace(S, K, x, :uninitialized, false, 0, 0, PenaltyLinearSolverOptions(; min_pivtol, initial_pivtol, max_pivtol, decrease_pivtol_factor, kwargs...))
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
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2Matrix}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
  mumps = workspace.M
  n, m = get_n(workspace), get_m(workspace)
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
    @views info_f = getrf!(
      BlasInt(2p),
      BlasInt(2p),
      Z2,
      stride(Z2, 2),
      workspace.H.ipiv,
      workspace.H.info,
    )
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
      workspace.H.ipiv,
      y1[1:(2p)],
      BlasInt(2p),
      workspace.H.info,
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

function get_inertia(workspace::PenaltyMUMPSWorkspace{WP,K2}) where {WP,K2}

  n, m = get_n(workspace), get_m(workspace)
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

function set_pivtol!(workspace::PenaltyMUMPSWorkspace, pivtol)
  MUMPS.set_cntl!(workspace.M, 1, pivtol)
end

function get_pivtol(workspace::PenaltyMUMPSWorkspace)
  return workspace.M.cntl[1]
end

function force_iterative_refinement!(workspace::PenaltyMUMPSWorkspace)
  MUMPS.set_icntl!(workspace.M, 10, -workspace.options.n_itref)
end

function relax_iterative_refinement!(workspace::PenaltyMUMPSWorkspace)
  MUMPS.set_icntl!(workspace.M, 10, workspace.options.n_itref)
end

function set_n_precise_solve!(workspace::PenaltyMUMPSWorkspace, n_precise::Int)
  workspace._n_precise = n_precise
end

function get_n_precise_solve(workspace::PenaltyMUMPSWorkspace)
  return workspace._n_precise
end

function SolverCore.reset!(workspace::PenaltyMUMPSWorkspace)
  set_n_fact!(workspace, 0)
  relax_iterative_refinement!(workspace)
  set_pivtol!(workspace, workspace.options.initial_pivtol)
  set_n_precise_solve!(workspace, 0)
end

# Perform Cholesky facto of H + σI to check positive definiteness.
function up_lb_is_pos_def_exact!(workspace::PenaltyMUMPSWorkspace)

  # Step 1: Factorize
  # [B + σI  0]
  # [0      -I]
  previous_inertia = workspace.H.α
  reset_jacobian!(workspace.H)
  set_dual_regularization!(workspace, one(previous_inertia))

  mumps = workspace.M
  mumps.job = MUMPS.FACTOR
  factorize!(mumps)
  workspace._n_fact += 1

  # Step 2: Revert values of H to original values
  set_jacobian!(workspace.H, workspace.H.A)
  set_dual_regularization!(workspace, previous_inertia)

  # Step 3: Check the ineria
  npos, nzero, nneg = get_inertia(workspace)
  is_pos_def = (npos == get_n(workspace) && mumps.infog[1] >= 0)

  return is_pos_def
end
