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

# up_lb_is_pos_def: whether the H + σI of
# the augmented K2 system is positive definite. `H` is the current Hessian
# (of the Lagrangian) approximation stored in `workspace` and σ is the
# associated primal regularization parameter.
up_lb_is_pos_def(workspace::PenaltyWorkspace) = up_lb_is_pos_def(workspace, workspace.H)

up_lb_is_pos_def(::PenaltyWorkspace, ::CompactBFGSK2) = true

# Direct-solver K2 systems where H is stored as a concrete (sparse) matrix:
# try simple necessary/sufficient conditions. #TODO: implement a necessary and 
# sufficient condition (Cholesky facto).
function up_lb_is_pos_def(workspace::PenaltyWorkspace, ::Symmetric)
  npos, nzero, nneg = get_inertia(workspace)
  (npos == workspace.n && nneg == workspace.m) || return false
  d, _ = primal_diagonal_and_row_sums(workspace)
  return all(>(0), d)
end

"""
    primal_diagonal_and_row_sums(workspace) -> (d, s)

For the sparse symmetric K2 matrix `H` stored in `workspace` (see `get_H`),
fill and return `d` and `s`, two length-`n` buffers preallocated on
`workspace` (`n = workspace.n`), such that `d[i]` is the `i`-th diagonal
entry of `H + σI`, and `s[i]` is the sum of the absolute values of the
off-diagonal entries of row `i` of `H + σI`. Only the leading n×n
("primal") block is considered; entries coupling to the constraint
Jacobian, or to the dual (-αI) block, are ignored.

`H` is assumed to store a triangular view of a symmetric
matrix; duplicate entries in `H`, if any, are correctly accumulated.

The returned vectors alias `workspace`'s internal buffers and are
overwritten on the next call; this function performs no allocation.
"""
function primal_diagonal_and_row_sums(workspace::PenaltyWorkspace)
  d, s = workspace._primal_diag, workspace._primal_row_sum
  fill_primal_diagonal_and_row_sums!(d, s, get_H(workspace))
  return d, s
end

function fill_primal_diagonal_and_row_sums!(
  d::AbstractVector,
  s::AbstractVector,
  H::SparseMatrixCOO,
)
  n = length(d)
  fill!(d, zero(eltype(d)))
  fill!(s, zero(eltype(s)))
  rows_, cols_, vals_ = H.rows, H.cols, H.vals
  for k in eachindex(vals_)
    i, j = rows_[k], cols_[k]
    (i > n || j > n) && continue
    v = vals_[k]
    if i == j
      d[i] += v
    else
      s[i] += abs(v)
      s[j] += abs(v)
    end
  end
  return d, s
end

function fill_primal_diagonal_and_row_sums!(
  d::AbstractVector,
  s::AbstractVector,
  H::SparseMatrixCSC,
)
  n = length(d)
  fill!(d, zero(eltype(d)))
  fill!(s, zero(eltype(s)))
  rowval, nzval = H.rowval, H.nzval
  for j = 1:n
    for k in nzrange(H, j)
      i = rowval[k]
      i > n && continue
      v = nzval[k]
      if i == j
        d[i] += v
      else
        s[i] += abs(v)
        s[j] += abs(v)
      end
    end
  end
  return d, s
end
