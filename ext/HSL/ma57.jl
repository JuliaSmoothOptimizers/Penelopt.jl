mutable struct PenaltyMA57Workspace{WP<:Ma57,K2<:AbstractK2Matrix,V<:AbstractVector,Opt<:PenaltyLinearSolverOptions} <:
               AbstractHSLWorkspace
  M::WP
  H::K2
  x::V
  work::V
  _qn_work::V
  dx::V
  status::Symbol
  factorized::Bool
  _n_fact::Int
  _n_precise::Int # Counter for how many times the relative error has been below threshold
  options::Opt
end

function construct_ma57_workspace(
  H::AbstractK2Matrix{T,M1,M2,M3},
  u1,
  n,
  m;
  min_pivtol::T = eps(T),
  initial_pivtol::T = eps(T),
  max_pivtol::T = T(1e-2),
  decrease_pivtol_factor::T = T(0.1),
  kwargs...
) where {T,M1,M2,M3<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  S = ma57_coord(
    n + m,
    H.H.data.rows,
    H.H.data.cols,
    H.H.data.vals,
    sqd = true,
    print_level = -1,
  )
  S.control.cntl[1] = initial_pivtol # Ma57 cntl(1): pivot threshold, see update_pivtol!
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
    0,
    PenaltyLinearSolverOptions(; min_pivtol, initial_pivtol, max_pivtol, decrease_pivtol_factor, kwargs...)
  )
end

# Linear solver options interface, following src/linear_algebra/mumps.jl.
# Ma57 cntl(1): threshold for numerical pivoting, used when icntl(7) = 1.
set_pivtol!(workspace::PenaltyMA57Workspace, pivtol) = (workspace.M.control.cntl[1] = pivtol; nothing)
get_pivtol(workspace::PenaltyMA57Workspace) = workspace.M.control.cntl[1]
# Ma57 rinfo(6), rinfo(7): backward errors, computed by ma57_solve! when iterative refinement is performed.
relative_error!(workspace::PenaltyMA57Workspace) = max(workspace.M.info.rinfo[6], workspace.M.info.rinfo[7])
# Ma57 cntl(3): if the norm of the scaled residuals does not decrease by a factor of at least cntl(3),
# iterative refinement stops with info(1) = -8. Forcing keeps refining as long as the residual decreases,
# relaxing restores the Ma57 default of 0.5.
force_iterative_refinement!(workspace::PenaltyMA57Workspace) = (workspace.M.control.cntl[3] = 1; nothing)
relax_iterative_refinement!(workspace::PenaltyMA57Workspace) = (workspace.M.control.cntl[3] = 0.5; nothing)
set_n_precise_solve!(workspace::PenaltyMA57Workspace, n::Int) = (workspace._n_precise = n; nothing)
get_n_precise_solve(workspace::PenaltyMA57Workspace) = workspace._n_precise

function reset!(workspace::PenaltyMA57Workspace)
  set_n_fact!(workspace, 0)
  relax_iterative_refinement!(workspace)
  set_pivtol!(workspace, workspace.options.initial_pivtol)
  set_n_precise_solve!(workspace, 0)
end

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  # Ma57 icntl(7): Controls the pivotting strategy.
  # Ma57 icntl(7): A value of 1 performs threshold pivoting with tolerance cntl(1), which is managed by
  # update_pivtol! as CNTL(1) is for MUMPS. The default cntl(1) = eps(T) barely restricts the choice of pivots.
  workspace.M.control.icntl[7] = 1

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
  workspace.M.control.icntl[9] = workspace.options.n_itref

  try
    ma57_solve!(workspace.M, u, workspace.x, workspace.dx, workspace.work, workspace.options.n_itref)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, workspace.x) || workspace.M.info.info[1] < 0
    workspace.status = :failed
    return
  end

  update_pivtol!(workspace)
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
  # Ma57 icntl(7): A value of 1 performs threshold pivoting with tolerance cntl(1), which is managed by
  # update_pivtol! as CNTL(1) is for MUMPS. The default cntl(1) = eps(T) barely restricts the choice of pivots.
  workspace.M.control.icntl[7] = 1

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
  workspace.M.control.icntl[9] = workspace.options.n_itref

  try
    ma57_solve!(workspace.M, u, x1, workspace.dx, workspace.work, workspace.options.n_itref)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  if any(isnan, x1) || workspace.M.info.info[1] < 0
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
    try
      ma57_solve!(workspace.M, x2, x3, workspace.dx, workspace.work, workspace.options.n_itref)
    catch e
      !(e isa HSL.Ma57Exception) && rethrow(e)
    end
    if any(isnan, x3) || workspace.M.info.info[1] < 0
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
  is_pos_def = (
    npos == get_n(workspace) &&
    !(workspace.M.info.info[1] < 0 || workspace.M.info.info[1] == 4)
  )

  return is_pos_def
end
