module PeneloptHSLExt

using HSL
using Penelopt

using LinearAlgebra, SparseMatricesCOO

# Import BLAS functions
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra: BlasInt, libblastrampoline

import Penelopt: AbstractHSLWorkspace, AbstractK2Matrix
import Penelopt: construct_ma57_workspace, solve_system!, update_workspace!
import Penelopt: get_H, get_inertia, get_solution!, get_status, get_n, get_m
import Penelopt: up_lb_is_pos_def_exact!, reset_jacobian!, set_jacobian!, set_dual_regularization!
import Penelopt: getrf!, getrs!
import Penelopt: CompactBFGS, CompactBFGSK2Matrix

function __init__()
  Penelopt._set_hsl_available(HSL.LIBHSL_isfunctional())
end

include("HSL/ma57.jl")

end
