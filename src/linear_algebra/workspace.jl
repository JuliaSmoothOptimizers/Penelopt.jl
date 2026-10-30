abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace{WP, K2} <: PenaltyWorkspace end

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

function factorize!(::PenaltyDirectWorkspace) end
function solve_system!(::PenaltyDirectWorkspace, u) end

function invalidate_factorization!(workspace::PenaltyDirectWorkspace)
  workspace.factorized = false
end

factorized(workspace::PenaltyDirectWorkspace) = workspace.factorized

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

function SolverCore.set_status!(workspace::PenaltyDirectWorkspace, status::Symbol)
  workspace.status = status
end

# Sherman-Morrison-Woodbury formula for solving linear systems with low-rank updates.
# ([σI+ξI  Aᵀ] + EFᵀ)[x] = [u]
# ([A     -αI] + EFᵀ)[x] = [u]
# Requires the functions
# - factorize!(workspace)
# - solve_system!(workspace, u::AbstractVector) -> Writes the solution into workspace.x
# - solve_system!(workspace, Z::AbstractMatrix) -> Overwrites Z
function solve_system!(
  workspace::PenaltyDirectWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2Matrix}
  workspace.status = :success
  H = workspace.H
  B = H.B

  n, m = get_n(workspace), get_m(workspace)
  p = min(B._insert - 1, B._mem)

  x1, x2, x3, y1 = H.x1, H.x2, H.x3, H.y1
  Z1, Z2 = H.Z1, H.Z2
  ipiv, info = H.ipiv, H.info

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
  factorize!(workspace)

  if !factorized(workspace)
    set_status!(workspace, :failed)
    return
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]
  solve_system!(workspace, u)
  get_status(workspace) == :failed && return
  x1 .= workspace.x

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

    solve_system!(workspace, Z1)
    get_status(workspace) == :failed && return

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
      ipiv,
      info,
    )
    if info_f != 0
      set_status!(workspace, :failed)
      return
    end

    @views info_s = getrs!(
      'N',
      BlasInt(2p),
      BlasInt(1),
      Z2[1:(2p), 1:(2p)],
      stride(Z2, 2),
      ipiv,
      y1[1:(2p)],
      BlasInt(2p),
      info,
    )
    if info_s != 0
      set_status!(workspace, :failed)
      return
    end
    if any(isnan, @view y1[1:(2*p)])
      set_status!(workspace, :failed)
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
    solve_system!(workspace, x2)
    get_status(workspace) == :failed && return
    x3 .= workspace.x

    # Step 8:
    # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
    # [A -αI]  [u] = x₁ - x₃
    workspace.x .= x1 .- x3

  else # p = 0
    workspace.x .= x1
  end
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

# Perform Cholesky facto of H + σI to check positive definiteness.
function up_lb_is_pos_def_exact!(workspace::PenaltyDirectWorkspace)

  # Step 1: Factorize
  # [B + σI  0]
  # [0      -I]
  previous_inertia = workspace.H.α
  reset_jacobian!(workspace.H)
  set_dual_regularization!(workspace, one(previous_inertia))

  factorize!(workspace)
  success = factorized(workspace)

  # Step 2: Revert values of H to original values
  set_jacobian!(workspace.H, workspace.H.A)
  set_dual_regularization!(workspace, previous_inertia)

  # Step 3: Check the ineria
  npos, nzero, nneg = get_inertia(workspace)
  is_pos_def = (npos == get_n(workspace) && success)

  return is_pos_def
end

# HSL Misc.: Checks whether the user has the license in the extension.
const _HSL_AVAILABLE = Ref(false)
hsl_functional() = _HSL_AVAILABLE[]
_set_hsl_available(flag::Bool) = (_HSL_AVAILABLE[] = flag)

function construct_ma57_workspace(H, u1, n, m)
  error("MA57 not available. Load PeneloptHSLExt.")
end

abstract type AbstractHSLWorkspace{WP, K2} <: PenaltyDirectWorkspace{WP, K2} end

# LDLFactorizations Misc.
function construct_ldlt_workspace(H, u1, n, m)
  error("LDLFactorizations not available. Load PeneloptLDLFactorizationsExt.")
end

abstract type AbstractLDLTWorkspace{WP, K2} <: PenaltyDirectWorkspace{WP, K2} end

# MUMPS Misc.
abstract type AbstractMUMPSWorkspace{WP, K2} <: PenaltyDirectWorkspace{WP, K2} end

get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end

function SolverCore.reset!(workspace::PenaltyWorkspace)
  set_n_fact!(workspace, 0)
end
