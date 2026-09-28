module PeneloptLDLFactorizationsExt

using LDLFactorizations
using Penelopt

using LinearAlgebra, SparseArrays, SparseMatricesCOO

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import Penelopt: AbstractLDLTWorkspace
import Penelopt: construct_ldlt_workspace, solve_system!, update_workspace!
import Penelopt: get_H, get_inertia, get_solution!, get_status
import Penelopt: getrf!, getrs!
import Penelopt: CompactBFGS, CompactBFGSK2
import Penelopt: set_dual_inertia!, set_primal_inertia!

include("LDLFactorizations/ldlt.jl")

end
