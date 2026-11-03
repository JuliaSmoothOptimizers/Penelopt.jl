for (getrf, getrs, T) in ((:sgetrf_, :sgetrs_, :Float32), (:dgetrf_, :dgetrs_, :Float64))
  @eval begin

    function getrf!(
      m::BlasInt,
      n::BlasInt,
      a::AbstractMatrix{$T},
      lda::BlasInt,
      ipiv::AbstractVector{BlasInt},
      info::Base.RefValue{BlasInt},
    )
      ccall(
        (@blasfunc($getrf), libblastrampoline),
        Cvoid,
        (Ref{BlasInt}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}),
        m,
        n,
        a,
        lda,
        ipiv,
        info,
      )
      return info[]
    end

    function getrs!(
      trans::Char,
      n::BlasInt,
      nrhs::BlasInt,
      a::AbstractMatrix{$T},
      lda::BlasInt,
      ipiv::AbstractVector{BlasInt},
      b::AbstractVecOrMat{$T},
      ldb::BlasInt,
      info::Base.RefValue{BlasInt},
    )
      ccall(
        (@blasfunc($getrs), libblastrampoline),
        Cvoid,
        (
          Ref{UInt8},
          Ref{BlasInt},
          Ref{BlasInt},
          Ptr{$T},
          Ref{BlasInt},
          Ptr{BlasInt},
          Ptr{$T},
          Ref{BlasInt},
          Ref{BlasInt},
          Clong,
        ),
        trans,
        n,
        nrhs,
        a,
        lda,
        ipiv,
        b,
        ldb,
        info,
        1,
      )
      return info[]
    end
  end
end

for (potrf, T) in ((:spotrf_, :Float32), (:dpotrf_, :Float64))
  @eval begin

    function potrf!(
      uplo::Char,
      n::BlasInt,
      a::AbstractMatrix{$T},
      lda::BlasInt,
      info::Base.RefValue{BlasInt},
    )
      ccall(
        (@blasfunc($potrf), libblastrampoline),
        Cvoid,
        (Ref{UInt8}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong),
        uplo,
        n,
        a,
        lda,
        info,
        1,
      )
      return info[]
    end
  end
end

