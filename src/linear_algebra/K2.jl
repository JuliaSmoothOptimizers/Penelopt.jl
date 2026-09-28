export CompactBFGSK2

mutable struct CompactBFGSK2{
  T<:Real,
  V,
  M1<:CompactBFGS,
  M2<:AbstractMatrix{T},
  M3<:AbstractMatrix{T},
} <: AbstractMatrix{T}
  B::M1
  H::M2
  Z1::M3
  Z2::M3
  x1::V
  x2::V
  x3::V
  y1::V
end

function K2(
  n::Int,
  m::Int,
  nrow::Int,
  ncol::Int,
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

  return Symmetric(H)
end

function K2(
  n::Int,
  m::Int,
  nrow::Int,
  ncol::Int,
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

  return CompactBFGSK2(
    B,
    Symmetric(H),
    zeros(T, n + m, 2*B._mem),
    zeros(T, 2*B._mem, 2*B._mem),
    zeros(T, n+m),
    zeros(T, n+m),
    zeros(T, n+m),
    zeros(T, 2*B._mem),
  )
end

function K2(
  n::Int,
  m::Int,
  nrow::Int,
  ncol::Int,
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

  return Symmetric(H)
end
