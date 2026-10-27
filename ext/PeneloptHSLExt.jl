module PeneloptHSLExt

using HSL
using Penelopt

using LinearAlgebra, SparseMatricesCOO

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import Penelopt: AbstractHSLWorkspace
import Penelopt: construct_ma57_workspace, solve_system!, update_workspace!
import Penelopt: get_H, get_inertia, get_solution!, get_status
import Penelopt: getrf!, getrs!
import Penelopt: CompactBFGS, CompactBFGSK2
import Penelopt: set_dual_inertia!, set_primal_inertia!
import Penelopt: relative_error!, increase_pivtol!, decrease_pivtol!, reset!
using Penelopt: update_pivtol!, set_n_fact!
using Penelopt: pivtol_min, PIVTOL_ROBUST, PIVTOL_DECREASE_FACTOR, N_ITREF

function __init__()
  Penelopt._set_hsl_available(HSL.LIBHSL_isfunctional())
end

include("HSL/ma57_workspace.jl")

end
