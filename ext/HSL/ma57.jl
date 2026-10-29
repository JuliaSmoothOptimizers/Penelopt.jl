mutable struct PenaltyMA57Workspace{
  WP<:Ma57,
  K2<:AbstractK2Matrix,
  V<:AbstractVector,
} <: AbstractHSLWorkspace
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
  H::AbstractK2Matrix{T, M1, M2, M3},
  u1,
  n,
  m,
) where {T,M1,M2,M3<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  S = ma57_coord(n + m, H.H.data.rows, H.H.data.cols, H.H.data.vals, sqd = true, print_level = -1)
  _qn_work = M1 <: CompactBFGS ? similar(u1, 2*length(u1)*H.B._mem) : similar(u1, 0)
  
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

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 3 performs no pivoting: for a sufficiently large value of σ and a positive value of α,
  # The matrix is strongly factorizable, so we don't need pivoting.
  workspace.M.control.icntl[7] = 3

  if !workspace.factorized
    try
      ma57_factorize!(workspace.M)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    workspace._n_fact += 1
  end

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  if workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4
    workspace.status = :failed
    return
  else
    workspace.factorized = true
  end

  # Ma57 icntl(9): Controls the max number of iterative refinement steps.
  workspace.M.control.icntl[9] = 10

  # Ma57 control.cntl(3): If the norm of the scaled residuals does not decrease by a factor of at least cntl(3), 
  # then the iterative refinement stops.
  workspace.M.control.cntl[3] = one(eltype(u)) # Perform iterative refinement
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
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2<:CompactBFGSK2Matrix}
  workspace.status = :success
  H = workspace.H
  B = workspace.H.B
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

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 3 performs no pivoting: for a sufficiently large value of σ and a positive value of α,
  # The matrix is strongly factorizable, so we don't need pivoting.
  workspace.M.control.icntl[7] = 3

  if !workspace.factorized
    try
      ma57_factorize!(workspace.M)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    workspace._n_fact += 1
  end

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  if workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4
    workspace.status = :failed
    return
  else
    workspace.factorized = true
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]

  # Ma57 icntl(9): Controls the max number of iterative refinement steps.
  workspace.M.control.icntl[9] = 10

  # Ma57 control.cntl(3): If the norm of the scaled residuals does not decrease by a factor of at least cntl(3), 
  # then the iterative refinement stops.
  workspace.M.control.cntl[3] = one(eltype(u)) # Perform iterative refinement
  try
    ma57_solve!(workspace.M, u, x1, workspace.dx, workspace.work, 10)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, x1) || workspace.M.info.info[1] < 0
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
    try
      ma57_solve!(workspace.M, Z1, workspace._qn_work)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    if any(isnan, Z1) || workspace.M.info.info[1] < 0
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
      getrf!(BlasInt(2p), BlasInt(2p), Z2, stride(Z2, 2), workspace.H.ipiv, workspace.H.info)
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
    try
      ma57_solve!(workspace.M, x2, x3, workspace.dx, workspace.work, 10)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    if any(isnan, x3) || workspace.M.info.info[1] < 0
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

function get_inertia(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}

  n, m = get_n(workspace), get_m(workspace)
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.info.info[24]
  rank = workspace.M.info.info[25]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end

# Perform Cholesky facto of H + σI to check positive definiteness.
function up_lb_is_pos_def_exact!(workspace::PenaltyMA57Workspace)
  
  # Step 1: Factorize
  # [B + σI  0]
  # [0      -I]
  previous_inertia = workspace.H.α
  reset_jacobian!(workspace.H)
  set_dual_regularization!(workspace, one(previous_inertia))

  try
    ma57_factorize!(workspace.M)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  workspace._n_fact += 1

  # Step 2: Revert values of H to original values
  set_jacobian!(workspace.H, workspace.H.A)
  set_dual_regularization!(workspace, previous_inertia)

  # Step 3: Check the ineria
  npos, nzero, nneg = get_inertia(workspace)
  is_pos_def = (npos == get_n(workspace) && !(workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4))

  return is_pos_def
end
