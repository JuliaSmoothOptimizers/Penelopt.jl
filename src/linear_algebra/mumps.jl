mutable struct PenaltyMUMPSWorkspace{WP<:Mumps,K2<:AbstractK2Matrix,V<:AbstractVector} <:
               AbstractMUMPSWorkspace{WP, K2}
  M::WP
  H::K2
  x::V
  status::Symbol
  factorized::Bool
  analyzed::Bool
  _n_fact::Int
  _n_precise::Int # Counter for how many times the relative error has been below threshold
end

function construct_mumps_workspace(
  K::AbstractK2Matrix{T,M1,M2,M3},
  u1,
  n,
  m,
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
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

  return PenaltyMUMPSWorkspace(S, K, x, :uninitialized, false, false, 0, 0)
end

function factorize!(workspace::PenaltyMUMPSWorkspace)
  factorized(workspace) && return
  mumps = workspace.M
  mumps.job = workspace.analyzed ? MUMPS.FACTOR : MUMPS.ANALYZE_FACTOR
  MUMPS.invoke_mumps!(mumps)
  set_n_fact!(workspace, get_n_fact(workspace) + 1)

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
    MUMPS.invoke_mumps!(mumps)
    set_n_fact!(workspace, get_n_fact(workspace) + 1)
    k = k + 1
  end

  # MUMPS infog(1): a negative value is an error in the factorization.
  if mumps.infog[1] < 0
    workspace.status = :failed
    return
  else
    workspace.analyzed = true
    workspace.factorized = true
  end
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace,
  u::AbstractVector,
)
  set_status!(workspace, :success)
  mumps, H = workspace.M, workspace.H

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  workspace.x .= u
  MUMPS.mumps_solve!(workspace.x, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, workspace.x) || mumps.infog[1] < 0
    set_status!(workspace, :failed)
    return
  end

  update_pivtol!(workspace)

  return
end

function solve_system!(
  workspace::PenaltyMUMPSWorkspace,
  Z::AbstractMatrix,
)
  workspace.status = :success
  mumps, H = workspace.M, workspace.H

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  MUMPS.associate_rhs!(mumps, Z1; unsafe = true)
  MUMPS.mumps_solve!(Z1, mumps; rhs_changed = true)

  # MUMPS infog(1): a negative value is an error in the factorization.
  if any(isnan, Z1) || mumps.infog[1] < 0
    set_status!(workspace, :failed)
    return
  end

  update_pivtol!(workspace)

  return
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

relative_error!(workspace::PenaltyMUMPSWorkspace) = max(workspace.M.rinfog[6], workspace.M.rinfog[7], workspace.M.rinfog[8])

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
    set_status!(workspace, :failed)
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
  workspace._n_precise = 0
end

function invalidate_analysis!(workspace::PenaltyMUMPSWorkspace)
  workspace.analyzed = false
end