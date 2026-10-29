module PeneloptLDLFactorizationsExt

using LDLFactorizations
using Penelopt

using LinearAlgebra, SparseArrays, SparseMatricesCOO

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import Penelopt: AbstractLDLTWorkspace, AbstractK2Matrix
import Penelopt: SparseSymmetricCOO, SparseSymmetricCSC
import Penelopt: invalidate_factorization!, get_n, get_m
import Penelopt: construct_ldlt_workspace, solve_system!
import Penelopt: get_H, get_inertia, get_solution!, get_status
import Penelopt: getrf!, getrs!
import Penelopt: CompactBFGS, CompactBFGSK2Matrix

include("LDLFactorizations/ldlt.jl")

end
