mutable struct PenaltyMA57Workspace{WP<:Ma57,K2<:AbstractK2Matrix,V<:AbstractVector} <:
               AbstractHSLWorkspace{WP, K2}
  M::WP
  H::K2
  x::V
  work::V
  _qn_work::V
  dx::V
  status::Symbol
  factorized::Bool
  _n_fact::Int
end

function construct_ma57_workspace(
  H::AbstractK2Matrix{T,M1,M2,M3},
  u1,
  n,
  m,
) where {T,M1,M2,M3<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  S = ma57_coord(
    n + m,
    H.H.data.rows,
    H.H.data.cols,
    H.H.data.vals,
    sqd = true,
    print_level = -1,
  )
  _qn_work = M1 <: CompactBFGS ? similar(u1, 2*length(u1)*H.B._mem) : similar(u1, 0)

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 3 performs no pivoting: for a sufficiently large value of σ and a positive value of α,
  # The matrix is strongly factorizable, so we don't need pivoting.
  S.control.icntl[7] = 3

  # Ma57 icntl(9): Controls the max number of iterative refinement steps.
  S.control.icntl[9] = 10

  # Ma57 control.cntl(3): If the norm of the scaled residuals does not decrease by a factor of at least cntl(3), 
  # then the iterative refinement stops.
  S.control.cntl[3] = one(eltype(u)) # Perform iterative refinement

  return PenaltyMA57Workspace(
    S,
    H,
    similar(u1),                # x
    similar(u1, 4*length(u1)),  # work
    _qn_work,                   # _qn_work
    similar(u1),                # dx
    :uninitialized,
    false,
    0,
  )
end

function factorize!(workspace::PenaltyMA57Workspace)
  factorized(workspace) && return
  try
    ma57_factorize!(workspace.M)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  set_n_fact!(workspace, get_n_fact(workspace) + 1)

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  if workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4
    set_status!(workspace, :failed)
    return
  else
    workspace.factorized = true
  end
end

function solve_system!(
  workspace::PenaltyMA57Workspace,
  u::AbstractVector,
)
  set_status!(workspace, :success)

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  try
    ma57_solve!(workspace.M, u, workspace.x, workspace.dx, workspace.work, 10)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end

  if any(isnan, workspace.x) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end
end

function solve_system!(
  workspace::PenaltyMA57Workspace,
  u::AbstractMatrix,
)
  set_status!(workspace, :success)

  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  try
    ma57_solve!(workspace.M, u, workspace._qn_work)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end

  if any(isnan, u) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end
end

function get_inertia(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}

  n, m = get_n(workspace), get_m(workspace)
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.info.info[24]
  rank = workspace.M.info.info[25]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end