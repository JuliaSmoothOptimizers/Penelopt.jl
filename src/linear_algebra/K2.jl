# An abstract type for the 2x2 block KKT matrix.
# [ B + σI   Aᵀ ]
# [ A       -αI ]
abstract type AbstractK2Matrix{T,M1,M2,M3} <: AbstractMatrix{T} end
# T is the element type of the matrix.
# M1 is the type of the (1,1) block.
# M2 is the type of the (1,2) block.
# M3 is the type of the full matrix.

const SparseSymmetricCOO{T} = Union{SparseMatrixCOO{T},Symmetric{T,<:SparseMatrixCOO{T}}}
const SparseSymmetricCSC{T} = Union{SparseMatrixCSC{T},Symmetric{T,<:SparseMatrixCSC{T}}}

mutable struct K2Matrix{
  T<:Real,
  M1<:AbstractMatrix{T},
  M2<:AbstractMatrix{T},
  M3<:AbstractMatrix{T},
  V<:AbstractVector{T},
} <: AbstractK2Matrix{T,M1,M2,M3}
  B::M1
  A::M2
  H::M3
  α::T
  σ::T
  n::Int
  m::Int
  nnz_A::Int
  nnz_B::Int
  _primal_diag::V    # Preallocated buffer for up_lb_is_pos_def
  _primal_row_sum::V # Preallocated buffer for up_lb_is_pos_def
end

# CompactBFGSK2Matrix is a 2x2 block KKT matrix with a compact BFGS approximation for the (1,1) block.
mutable struct CompactBFGSK2Matrix{
  T<:Real,
  M1<:CompactBFGS,
  M2<:AbstractMatrix{T},
  M3<:AbstractMatrix{T},
  V<:AbstractVector{T},
  M4<:AbstractMatrix{T},
  VI<:AbstractVector{BlasInt},
} <: AbstractK2Matrix{T,M1,M2,M3}
  B::M1
  A::M2
  H::M3
  α::T
  σ::T
  n::Int
  m::Int
  nnz_A::Int
  nnz_B::Int
  _primal_diag::V    # Preallocated buffer for up_lb_is_pos_def
  _primal_row_sum::V # Preallocated buffer for up_lb_is_pos_def
  Z1::M4  # Extra storage for the Sherman–Morrison–Woodbury formula
  Z2::M4  # Extra storage for the Sherman–Morrison–Woodbury formula
  x1::V   # Extra storage for the Sherman–Morrison–Woodbury formula
  x2::V   # Extra storage for the Sherman–Morrison–Woodbury formula
  x3::V   # Extra storage for the Sherman–Morrison–Woodbury formula
  y1::V   # Extra storage for the Sherman–Morrison–Woodbury formula
  ipiv::VI # For LAPACK LU factorization of the compact BFGS matrix
  info::Base.RefValue{BlasInt}  # For LAPACK LU factorization of the compact BFGS matrix
end

function K2(
  n::Int,
  m::Int,
  α::T,
  σ::T,
  A::M1,
  B::M2;
  format::Symbol = :coo,
  int_type::Type = Int64,
) where {T,M1<:SparseMatrixCOO,M2<:SparseMatrixCSC}

  I, J, V = Vector{int_type}(), Vector{int_type}(), Vector{T}()

  # Step 1: Add the transpose of B to the K2 matrix.
  # Produces
  # [ Bᵀ 0 ]
  # [ 0  0 ]
  #
  # Note: LDLFactorizations requires an upper triangular view;
  # Meanwhile, NLPModels produces a lower triangular view...
  for j = 1:n
    for k = B.colptr[j]:(B.colptr[j+1]-1)
      i = B.rowval[k]
      push!(I, j)
      push!(J, i)
      push!(V, B.nzval[k])
    end
  end

  # Step 2: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ Bᵀ Aᵀ ]
  # [ 0  0  ]
  #
  for k = 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Initialize inertia corrections
  # Produces 
  # [ Bᵀ+σI Aᵀ ]
  # [ 0    -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  append!(I, 1:(n+m))
  append!(J, 1:(n+m))
  append!(V, fill(σ_temp, n))
  append!(V, fill(-α_temp, m))

  # Step 4: Construct the sparse matrix
  H = format == :coo ? SparseMatrixCOO(n+m, n+m, I, J, V) : sparse(I, J, V, n+m, n+m)

  return K2Matrix(B, A, Symmetric(H), α_temp, σ_temp, n, m, nnz(A), nnz(B), zeros(T, n), zeros(T, n))
end

function K2(
  n::Int,
  m::Int,
  α::T,
  σ::T,
  A::M1,
  B::M2;
  format = :coo,
  int_type::Type = Int64,
) where {T,M1<:SparseMatrixCOO,M2<:CompactBFGS}

  I, J, V = Vector{int_type}(), Vector{int_type}(), Vector{T}()

  # Step 1: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ 0  Aᵀ ]
  # [ 0  0  ]
  #
  for k = 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Initialize inertia corrections
  # Produces 
  # [ Bᵀ+σI Aᵀ ]
  # [ 0    -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  append!(I, 1:(n+m))
  append!(J, 1:(n+m))
  append!(V, fill(σ_temp, n))
  append!(V, fill(-α_temp, m))

  # Step 4: Construct the sparse matrix
  H = format == :coo ? SparseMatrixCOO(n+m, n+m, I, J, V) : sparse(I, J, V, n+m, n+m)

  return CompactBFGSK2Matrix(
    B,
    A,
    Symmetric(H),
    α_temp,
    σ_temp,
    n,
    m,
    nnz(A),
    0,
    zeros(T, 0),
    zeros(T, 0),
    zeros(T, n + m, 2*B._mem),
    zeros(T, 2*B._mem, 2*B._mem),
    zeros(T, n+m),
    zeros(T, n+m),
    zeros(T, n+m),
    zeros(T, 2*B._mem),
    Vector{LinearAlgebra.BlasInt}(undef, 2 * B._mem),
    Ref{BlasInt}(0),
  )
end

function K2(
  n::Int,
  m::Int,
  α::T,
  σ::T,
  A::M1,
  B::M2;
  format::Symbol = :coo,
  int_type::Type = Int64,
) where {T,M1<:SparseMatrixCOO,M2<:SparseMatrixCOO}

  I, J, V = Vector{int_type}(), Vector{int_type}(), Vector{T}()

  # Step 1: Add the transpose of B to the K2 matrix.
  # Produces
  # [ Bᵀ 0 ]
  # [ 0  0 ]
  #
  # Note: LDLFactorizations requires an upper triangular view;
  # Meanwhile, NLPModels produces a lower triangular view...
  append!(I, B.cols)
  append!(J, B.rows)
  append!(V, B.vals)

  # Step 2: Add the transpose of A to the K2 matrix.
  # Produces 
  # [ Bᵀ Aᵀ ]
  # [ 0  0  ]
  #
  for k = 1:nnz(A)
    push!(I, A.cols[k])          # transpose: row becomes column
    push!(J, A.rows[k] + n)
    push!(V, A.vals[k])
  end

  # Step 3: Initialize inertia corrections
  # Produces 
  # [ Bᵀ+σI Aᵀ ]
  # [ 0    -αI ]
  α_temp = iszero(α) ? eps(T) : α
  σ_temp = iszero(σ) ? eps(T) : σ

  append!(I, 1:(n+m))
  append!(J, 1:(n+m))
  append!(V, fill(σ_temp, n))
  append!(V, fill(-α_temp, m))

  # Step 4: Construct the sparse matrix
  H = format == :coo ? SparseMatrixCOO(n+m, n+m, I, J, V) : sparse(I, J, V, n+m, n+m)

  return K2Matrix(B, A, Symmetric(H), α_temp, σ_temp, n, m, nnz(A), nnz(B), zeros(T, n), zeros(T, n))
end

# K2 utils API
function get_H(K::AbstractK2Matrix{T,M1,M2,M3}) where {T,M1,M2,M3}
  return K.H
end

function get_H(K::AbstractK2Matrix{T,M1,M2,M3}) where {T,M1,M2,M3<:Symmetric{T}}
  return K.H.data
end

function set_primal_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  σ::T,
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  K.σ = σ
  H = get_H(K)
  H.vals[(end-K.m-K.n+1):(end-K.m)] .= σ
end

function set_primal_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  σ::T,
) where {T,M1,M2,M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  @inbounds for i = 1:K.n
    H[i, i] += σ - K.σ
  end
  K.σ = σ
end

function initialize_primal_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  σ::T,
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  set_primal_regularization!(K, σ)
end

function initialize_primal_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  σ::T,
) where {T,M1,M2,M3<:SparseSymmetricCSC{T}}
  K.σ = σ
  H = get_H(K)
  @inbounds for i = 1:K.n
    H[i, i] = σ
  end
end

function set_dual_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  α::T,
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  K.α = α
  H = get_H(K)
  H.vals[(end-K.m+1):end] .= -α
end

function set_dual_regularization!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  α::T,
) where {T,M1,M2,M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  @inbounds for i = 1:K.m
    H[K.n+i, K.n+i] = -α
  end
  K.α = α
end

function set_jacobian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  A::M2,
) where {T,M1,M2<:SparseSymmetricCOO{T},M3<:SparseSymmetricCOO{T}}
  H = get_H(K)
  K.A = A
  H.vals[(K.nnz_B+1):(K.nnz_B+K.nnz_A)] .= A.vals
end

function set_jacobian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  A::M2,
) where {T,M1,M2<:SparseSymmetricCOO{T},M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  K.A = A
  @inbounds for i = 1:length(A.vals)
    H[A.cols[i], K.n+A.rows[i]] = A.vals[i]
  end
end

function reset_jacobian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  H = get_H(K)
  H.vals[(K.nnz_B+1):(K.nnz_B+K.nnz_A)] .= 0
end

function reset_jacobian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
) where {T,M1,M2<:SparseSymmetricCOO{T},M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  @inbounds for i = 1:length(K.A.vals)
    H[K.A.cols[i], K.n+K.A.rows[i]] = zero(T)
  end
end

function set_hessian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  B::M1,
) where {T,M1<:SparseSymmetricCOO{T},M2,M3<:SparseSymmetricCOO{T}}
  H = get_H(K)
  K.B = B
  H.vals[1:K.nnz_B] .= B.vals
end

function set_hessian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
  B::M1,
) where {T,M1<:SparseSymmetricCOO{T},M2,M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  K.B = B
  @inbounds for i = 1:length(B.vals)
    if B.cols[i] == B.rows[i]
      H[B.cols[i], B.rows[i]] += B.vals[i]
    else
      H[B.cols[i], B.rows[i]] = B.vals[i]
    end
  end
end

function reset_hessian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
) where {T,M1,M2,M3<:SparseSymmetricCOO{T}}
  H = get_H(K)
  H.vals[1:K.nnz_B] .= zero(T)
end

function reset_hessian!(
  K::AbstractK2Matrix{T,M1,M2,M3},
) where {T,M1,M2,M3<:SparseSymmetricCSC{T}}
  H = get_H(K)
  B = K.B
  @inbounds for i = 1:length(B.vals)
    H[B.cols[i], B.rows[i]] = zero(T)
  end
end
