module PeneloptLDLFactorizationsExt

using LDLFactorizations
using Penelopt

using LinearAlgebra, SparseArrays, SparseMatricesCOO

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import Penelopt: AbstractLDLTWorkspace, AbstractK2Matrix, SparseSymmetricCSC
import Penelopt: CompactBFGSK2Matrix, getrf!, getrs!
import Penelopt: get_n, get_m, get_status
import Penelopt: invalidate_factorization!, construct_ldlt_workspace, solve_system!
import Penelopt: get_inertia, up_lb_is_pos_def_exact!
import Penelopt: reset_jacobian!, set_jacobian!, set_dual_regularization!
import Penelopt: PenaltyLinearSolverOptions

include("LDLFactorizations/ldlt.jl")

end
