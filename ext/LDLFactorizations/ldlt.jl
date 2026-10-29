mutable struct PenaltyLDLTWorkspace{
  WP<:LDLFactorization,
  K2<:AbstractK2Matrix,
  V<:AbstractVector,
} <: AbstractLDLTWorkspace
  M::WP
  H::K2
  x::V
  dx::V
  r::V
  status::Symbol
  _n_fact::Int
end

function construct_ldlt_workspace(
  K::AbstractK2Matrix{T, M1, M2, M3},
  u1,
  n,
  m,
) where {T, M1, M2, M3 <:SparseSymmetricCSC{T}}
  S = ldl_analyze(K.H)
  return PenaltyLDLTWorkspace(
    S,
    K,
    similar(u1),
    similar(u1),
    similar(u1),
    :uninitialized,
    0,
  )
end

function invalidate_factorization!(
  workspace::PenaltyLDLTWorkspace
)
  workspace.M.__factorized = false
end


# Given Ax ≈ b, refine the solution by solving AΔx = b - Ax and updating x += Δx
function refine!(
  workspace::PenaltyLDLTWorkspace,
  u::V;
  max_iter::Int = 5,
  tol::T = eps(T),
) where {T,V<:AbstractVector{T}}

  # Compute the residual r = u - H*x
  r, H, x = workspace.r, workspace.H.H, workspace.x
  n, m = get_n(workspace), get_m(workspace)
  dx = workspace.dx
  solved = false
  k = 0
  while k < max_iter && !solved
    r .= u

    # https://github.com/JuliaSparse/SparseArrays.jl/issues/685: mul!(r, H, x, -one(T), one(T)) is somehow extremely slow.
    mul!(r, H.data, x, -one(T), one(T)) # r = u - H*x
    mul!(r, H.data', x, -one(T), one(T))
    @inbounds for i = 1:(n+m)
      r[i] += H.data[i, i]*x[i]
    end

    ldiv!(dx, workspace.M, r) # H*dx = r
    x .+= dx
    k = k + 1
    solved = norm(dx) < tol*norm(x)
  end
  return workspace.x
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace{WP,K2},
  u::V,
) where {V<:AbstractVector,WP,K2}
  workspace.status = :success

  if !factorized(workspace.M)
    ldl_factorize!(workspace.H.H, workspace.M)
    workspace._n_fact += 1
  end
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end

  ldiv!(workspace.x, workspace.M, u)
  if any(isnan, workspace.x)
    workspace.status = :failed
    return
  end

  refine!(workspace, u)
  if any(isnan, workspace.x) ||
     norm(workspace.dx)/norm(workspace.x) > eps(eltype(workspace.x))^(0.5)
    workspace.status = :failed
    return
  end
end

function solve_system!(
  workspace::PenaltyLDLTWorkspace{WP,K2},
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
  if !factorized(workspace.M)
    ldl_factorize!(workspace.H.H, workspace.M)
    workspace._n_fact += 1
  end
  if !factorized(workspace.M)
    workspace.status = :failed
    return
  end

  # Step 2: Compute
  # [x₁] = [σI+ξI  Aᵀ]⁻¹[u]
  # [x₁] = [A     -αI]  [u]
  ldiv!(x1, workspace.M, u)
  if any(isnan, x1)
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
    ldiv!(workspace.M, Z1)
    if any(isnan, Z1)
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
    ldiv!(x3, workspace.M, x2)
    if any(isnan, x3)
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

function get_inertia(workspace::PenaltyLDLTWorkspace)
  LDL = workspace.M

  n = LDL.n
  (npos, nzero, nneg) = (0, 0, 0)

  D = LDL.d
  for i = 1:n
    d = D[i]
    if real(d) > 0
      npos += 1
    elseif abs(real(d)) < eps(eltype(d))
      nzero += 1
    else
      nneg += 1
    end
  end

  return npos, nzero, nneg
end

# Perform Cholesky facto of H + σI to check positive definiteness.
function up_lb_is_pos_def_exact!(workspace::PenaltyLDLTWorkspace)
  
  # Step 1: Factorize
  # [B + σI  0]
  # [0      -I]
  previous_inertia = workspace.H.α
  reset_jacobian!(workspace.H)
  set_dual_regularization!(workspace, one(previous_inertia))

  ldl_factorize!(workspace.H.H, workspace.M)
  workspace._n_fact += 1

  # Step 2: Revert values of H to original values
  set_jacobian!(workspace.H, workspace.H.A)
  set_dual_regularization!(workspace, previous_inertia)

  # Step 3: Check the ineria
  status = get_status(workspace)
  npos, nzero, nneg = get_inertia(workspace)
  is_pos_def = (npos == get_n(workspace) && status == :success)

  return is_pos_def
end