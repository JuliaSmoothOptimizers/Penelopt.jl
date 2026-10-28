mutable struct PenaltyMA57Workspace{
  WP<:Ma57,
  K2<:AbstractMatrix,
  V<:AbstractVector,
  VI<:Union{Nothing,AbstractVector},
  T<:Real,
} <: AbstractHSLWorkspace
  M::WP
  H::K2
  x::V
  work::V
  _qn_work::V
  _ipiv::VI # For CompactBFGS LU factorization
  _info::Base.RefValue{BlasInt}  # For CompactBFGS LU factorization
  dx::V
  σ::T
  n::Int
  m::Int
  status::Symbol
  factorized::Bool
  _n_fact::Int
  _primal_diag::V # Preallocated buffer for up_lb_is_pos_def
  _primal_row_sum::V # Preallocated buffer for up_lb_is_pos_def
  _n_precise::Int # Counter for how many times the relative error has been below threshold
  _ls_params::LinearSolverParams{T}
end

# Ma57 cntl(3): stop iterative refinement if the error does not decrease by cntl(3).
const MA57_ITREF_STAGNATION = 0.5
const MA57_ITREF_STAGNATION_ROBUST = 1.0

function get_H(
  solver_workspace::PenaltyMA57Workspace{WP,K2},
) where {T,WP,K2<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  return solver_workspace.H.data
end

function get_H(solver_workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2<:CompactBFGSK2}
  return solver_workspace.H.H.data
end

function mumps_like_ma57_control(params::LinearSolverParams{T}) where {T}
  # Deactivate Logging
  control = Ma57_Control{T}(print_level = -1)
  icntl, cntl = control.icntl, control.cntl

  # Ma57 icntl(7): 1 performs threshold partial pivoting with relative threshold cntl(1).
  icntl[7] = 1
  cntl[1] = params.pivtol_min

  # Ma57 icntl(6): 5 chooses the ordering automatically.
  icntl[6] = 5

  # Ma57 icntl(15): 1 scales the matrix with MC64.
  icntl[15] = 1

  # Max number of iterative refinement steps (at least 2 for the error analysis)
  icntl[9] = max(params.n_itref, 2)
  cntl[3] = MA57_ITREF_STAGNATION

  # Ma57 icntl(10): 0 skips the condition number estimates in the error analysis.
  icntl[10] = 0

  return control
end

function construct_ma57_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:Symmetric{T,SparseMatrixCOO{T,Int}}}
  params = LinearSolverParams{T}()
  S = ma57_coord(
    n + m,
    H.data.rows,
    H.data.cols,
    H.data.vals,
    mumps_like_ma57_control(params),
    Ma57_Info{T}(),
  )
  return PenaltyMA57Workspace(
    S,
    H,
    similar(u1),                # x
    similar(u1, 4*length(u1)),  # work
    similar(u1, 0),             # _qn_work
    nothing,                    # _ipiv
    Ref{BlasInt}(0),            # _info
    similar(u1),                # dx
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
    zeros(T, n),
    zeros(T, n),
    0,                          # _n_precise
    params,                     # _ls_params
  )
end

function construct_ma57_workspace(
  H::M,
  u1::V,
  n,
  m,
) where {T,V<:AbstractVector{T},M<:CompactBFGSK2}
  params = LinearSolverParams{T}()
  S = ma57_coord(
    n + m,
    H.H.data.rows,
    H.H.data.cols,
    H.H.data.vals,
    mumps_like_ma57_control(params),
    Ma57_Info{T}(),
  )
  return PenaltyMA57Workspace(
    S,
    H,
    similar(u1),                                         # x
    similar(u1, 4*length(u1)),                           # work
    similar(u1, 2*length(u1)*H.B._mem),                  # _qn_work
    Vector{LinearAlgebra.BlasInt}(undef, 2 * H.B._mem),  # _ipiv
    Ref{BlasInt}(0),                                     # _info
    similar(u1),                                         # dx
    zero(T),
    n,
    m,
    :uninitialized,
    false,
    0,
    zeros(T, 0),
    zeros(T, 0),
    0,                                                   # _n_precise
    params,                                              # _ls_params
  )
end

function update_workspace!(
  solver_workspace::PenaltyMA57Workspace,
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

  solver_workspace.M.vals .= H.vals

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function update_workspace!(solver_workspace::PenaltyMA57Workspace, A, σ, α)
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
  solver_workspace::PenaltyMA57Workspace,
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

  solver_workspace.M.vals .= H.vals

  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function set_dual_inertia!(solver_workspace::PenaltyMA57Workspace, α)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m+1):end] .= -α
  solver_workspace.M.vals[(end-m+1):end] .= -α
  solver_workspace.factorized = false
end

function set_primal_inertia!(solver_workspace::PenaltyMA57Workspace, σ)
  n, m = solver_workspace.n, solver_workspace.m
  H = get_H(solver_workspace)
  H.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.M.vals[(end-m-n+1):(end-m)] .= σ
  solver_workspace.σ = σ
  solver_workspace.factorized = false
end

function factorize_workspace!(workspace::PenaltyMA57Workspace)
  workspace.factorized && return true

  try
    ma57_factorize!(workspace.M)
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end
  workspace._n_fact += 1

  # Ma57 info(1): a negative value is an error in the factorization.
  # Ma57 info(1): a value of 4 indicates that the matrix is singular.
  info = workspace.M.info.info[1]
  if info < 0 || info == 4
    workspace.status = :failed
    return false
  end
  workspace.factorized = true
  return true
end

function solve_with_refinement!(workspace::PenaltyMA57Workspace, b, x)
  M = workspace.M
  rinfo = M.info.rinfo

  # Discard the error analysis of the previous solve.
  rinfo[6] = rinfo[7] = rinfo[10] = Inf

  try
    ma57_solve!(M, b, x, workspace.dx, workspace.work, Int(M.control.icntl[9]))
  catch e
    !(e isa HSL.Ma57Exception) && rethrow(e)
  end

  # Ma57 info(1): a value of -8 indicates that iterative refinement did not converge.
  info = M.info.info[1]
  if any(isnan, x) || (info < 0 && info != -8)
    workspace.status = :failed
    workspace._n_precise = 0
    return false
  end
  info > 0 && return true

  update_pivtol!(workspace)
  return true
end

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success
  factorize_workspace!(workspace) || return
  solve_with_refinement!(workspace, u, workspace.x)
  return
end

function solve_system!(
  workspace::PenaltyMA57Workspace{WP,K2},
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

  factorize_workspace!(workspace) || return

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]
  solve_with_refinement!(workspace, u, x1) || return

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
    solve_with_refinement!(workspace, x2, x3) || return

    # Step 8:
    # [B  Aᵀ]⁻¹[u] = x₁ - x₃ 
    # [A -αI]  [u] = x₁ - x₃
    workspace.x .= x1 .- x3
  else
    workspace.x .= x1
  end
end

function get_solution!(
  x::V,
  workspace::PenaltyMA57Workspace{WP,K2},
) where {V<:AbstractVector,WP,K2}
  x .= workspace.x
end

function get_status(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}
  return workspace.status
end

function get_inertia(workspace::PenaltyMA57Workspace{WP,K2}) where {WP,K2}

  n, m = workspace.n, workspace.m
  (npos, nzero, nneg) = (0, 0, 0)

  nneg = workspace.M.info.info[24]
  rank = workspace.M.info.info[25]
  nzero = n + m - rank
  npos = n + m - nzero - nneg

  return npos, nzero, nneg
end

function relative_error!(workspace::PenaltyMA57Workspace)
  rinfo = workspace.M.info.rinfo
  return max(rinfo[6], rinfo[7], rinfo[10])
end

function increase_pivtol!(workspace::PenaltyMA57Workspace)
  control, params = workspace.M.control, get_ls_params(workspace)
  control.cntl[1] = params.pivtol_robust
  control.cntl[3] = MA57_ITREF_STAGNATION_ROBUST
end

function decrease_pivtol!(workspace::PenaltyMA57Workspace)
  control, params = workspace.M.control, get_ls_params(workspace)
  control.cntl[1] = max(control.cntl[1] / params.pivtol_decrease_factor, params.pivtol_min)
  control.cntl[3] = MA57_ITREF_STAGNATION
end

function init_pivtol!(workspace::PenaltyMA57Workspace)
  control, params = workspace.M.control, get_ls_params(workspace)
  workspace._n_precise = 0
  control.cntl[1] = params.pivtol_min
  control.icntl[9] = max(params.n_itref, 2)
  control.cntl[3] = MA57_ITREF_STAGNATION
end
