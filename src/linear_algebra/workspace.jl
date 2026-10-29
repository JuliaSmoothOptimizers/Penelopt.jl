abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace <: PenaltyWorkspace end

function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :mumps) where {M,V}
  if solver == :ldlt
    return construct_ldlt_workspace(H, u1, n, m)
  elseif solver == :ma57
    return construct_ma57_workspace(H, u1, n, m)
  elseif solver == :mumps
    return construct_mumps_workspace(H, u1, n, m)
  else
    error(
      "Penelopt.jl: unknown linear solver `$(solver)`. Supported values are \"mumps\", \"ldlt\" and \"ma57\".",
    )
  end
end

function update_workspace!(
  workspace::PenaltyDirectWorkspace,
  B::M1,
  A::M2,
  σ::T,
  α::T;
) where {T,M1,M2}
  K = workspace.H
  initialize_primal_regularization!(K, σ)
  set_dual_regularization!(K, α)
  set_jacobian!(K, A)
  set_hessian!(K, B)
  invalidate_factorization!(workspace)
end

function update_workspace!(workspace::PenaltyDirectWorkspace, A::M, σ::T, α::T;) where {T,M}
  K = workspace.H
  reset_hessian!(K)
  set_jacobian!(K, A)
  initialize_primal_regularization!(K, σ)
  set_dual_regularization!(K, α)
  invalidate_factorization!(workspace)
end

function update_workspace!(
  workspace::PenaltyDirectWorkspace,
  B::M1,
  A::M2,
  σ::T,
  α::T;
) where {T,M1<:CompactBFGS{T},M2}
  K = workspace.H
  initialize_primal_regularization!(K, σ + B.ξ)
  set_jacobian!(K, A)
  set_dual_regularization!(K, α)
  K.σ = σ
  invalidate_factorization!(workspace)
end

function set_primal_regularization!(workspace::PenaltyDirectWorkspace, σ::T) where {T}
  set_primal_regularization!(workspace.H, σ)
  invalidate_factorization!(workspace)
end

function set_dual_regularization!(workspace::PenaltyDirectWorkspace, α::T) where {T}
  set_dual_regularization!(workspace.H, α)
  invalidate_factorization!(workspace)
end

function invalidate_factorization!(workspace::PenaltyDirectWorkspace)
  workspace.factorized = false
end

function get_n(workspace::PenaltyDirectWorkspace)
  return workspace.H.n
end

function get_m(workspace::PenaltyDirectWorkspace)
  return workspace.H.m
end

function get_solution!(x::V, workspace::PenaltyDirectWorkspace) where {V<:AbstractVector}
  x .= workspace.x
end

function get_status(workspace::PenaltyDirectWorkspace)
  return workspace.status
end

function up_lb_is_pos_def(workspace::PenaltyDirectWorkspace)

  (typeof(workspace.H) <: CompactBFGSK2Matrix) && return true
  n, m = get_n(workspace), get_m(workspace)

  # Step 1: check inertia
  npos, nzero, nneg = get_inertia(workspace)
  status = get_status(workspace)
  (status == :failed || npos != n || nneg != m) && return false

  # Step 2 & 3: cheap sufficient/necessary conditions:
  # 1. Check if the diagonal has a negative entry,
  # 2. Check if the matrix is diagonally dominant.
  d, s = workspace.H._primal_diag, workspace.H._primal_row_sum
  B = workspace.H.B
  s .= 0
  d .= workspace.H.σ
  nnz_B = length(B.vals)
  for i = 1:nnz_B
    row, col, val = B.rows[i], B.cols[i], B.vals[i]
    if row == col
      d[row] += val
    else
      s[row] += abs(val)
      s[col] += abs(val)
    end
  end

  is_diagonally_dominant = true
  for i = 1:n
    d[i] <= 0 && return false
    is_diagonally_dominant &= d[i] > s[i]
  end
  is_diagonally_dominant && return true

  # Step 4: fallback to Cholesky factorization.
  return up_lb_is_pos_def_exact!(workspace)
end

# HSL Misc.: Checks whether the user has the license in the extension.
const _HSL_AVAILABLE = Ref(false)
hsl_functional() = _HSL_AVAILABLE[]
_set_hsl_available(flag::Bool) = (_HSL_AVAILABLE[] = flag)

function construct_ma57_workspace(H, u1, n, m)
  error("MA57 not available. Load PeneloptHSLExt.")
end

abstract type AbstractHSLWorkspace <: PenaltyDirectWorkspace end

# LDLFactorizations Misc.
function construct_ldlt_workspace(H, u1, n, m)
  error("LDLFactorizations not available. Load PeneloptLDLFactorizationsExt.")
end

abstract type AbstractLDLTWorkspace <: PenaltyDirectWorkspace end

# MUMPS Misc.
abstract type AbstractMUMPSWorkspace <: PenaltyDirectWorkspace end

get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end

function SolverCore.reset!(workspace::PenaltyWorkspace)
  set_n_fact!(workspace, 0)
end
