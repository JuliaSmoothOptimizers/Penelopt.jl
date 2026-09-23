function get_linear_solver(
  solver::L2PenaltySolver{T,V,S,PB},
) where {T,V,H,WP,MSS<:MoreSorensenSolver{T,V,H,WP},S<:PenaltyR2NSolver{T,V,MSS},PB}
  if WP <: AbstractMUMPSWorkspace
    # The MUMPS struct exposes the version number.
    version = solver.subsolver.subsolver.workspace.M.version_number
    version = decode_mumps_version(version)
    return "MUMPS v$(version)"
  elseif WP <: AbstractLDLTWorkspace
    # LDLFactorizations.jl is loaded through the PeneloptLDLFactorizationsExt extension.
    ext = Base.get_extension(@__MODULE__, :PeneloptLDLFactorizationsExt)
    return "LDLFactorizations.jl v$(pkgversion(ext.LDLFactorizations))"
  else
    return ""
  end
end

function decode_mumps_version(v::NTuple)
  i = findfirst(==(0), v)
  if i === nothing
    return String(collect(UInt8.(v)))
  else
    return String(collect(UInt8.(v[1:(i-1)])))
  end
end

function introduction_message(solver::L2PenaltySolver, nlp::AbstractNLPModel)
  return """

  This is Penelopt.jl v$(pkgversion(@__MODULE__)).
  Running with linear solver $(get_linear_solver(solver)).

  $nlp
  """
end

function introduction_message(
  solver::PenaltyR2NSolver,
  nlp::AbstractNLPModel,
  stats::GenericExecutionStats,
)
  return separator(type = :inner_loop) * @sprintf(
    "\n            |  Solving subproblem minₓ f(x) + %-3.2e ‖c(x)‖₂ with tolerance %-3.2e...",
    stats.solver_specific[:tau],
    stats.solver_specific[:dual_ktol]
  )
end

const W_ITER = 7
const W_LARGE = 16
const W_MED = 12
const W_SMALL = 8
const W_ARROW = 6

const FMT_OBJ = "%+-16.7e"
const FMT_MED = "%-12.2e"
const FMT_SMALL = "%+-8.1f"

function separator(; type = :outer_loop)
  if type == :outer_loop
    return repeat("-", textwidth(header_message(type = type)))
  elseif type == :inner_loop
    return "      | " * repeat("-", textwidth(header_message(type = type))-4)
  elseif type == :ms_loop
    return "            | " * repeat("-", textwidth(header_message(type = type))-4)
  end
end

function header_message(; type = :outer_loop)
  if type == :outer_loop
    return @sprintf(
      "%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
      W_ITER,
      "Iter",
      W_ITER,
      "sIter",
      W_LARGE,
      "Objective",
      W_MED,
      "pfeas",
      W_MED,
      "dfeas",
      W_MED,
      "τ",
      W_MED,
      "ptol",
      W_MED,
      "dtol",
      W_MED,
      "‖x‖",
    )
  elseif type == :inner_loop
    return @sprintf(
      "      | %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
      W_ITER,
      "Iter",
      W_ITER,
      "sIter",
      W_LARGE,
      "Objective",
      W_MED,
      "pfeas",
      W_MED,
      "dfeas",
      W_MED,
      "σ",
      W_MED,
      "ρ",
      W_ARROW,
      "Type",
      W_MED,
      "‖x‖",
      W_MED,
      "‖s‖",
    )
  elseif type == :ms_loop
    return @sprintf(
      "            | %-*s%-*s%-*s%-*s%-*s%-*s%-*s%-*s",
      W_ITER,
      "Iter",
      W_MED,
      "σ",
      W_MED,
      "α",
      W_MED,
      "‖y‖",
      W_MED,
      "Δ",
      W_MED,
      "inertia",
      W_MED,
      "lsolve",
      W_MED,
      "descent",
    )
  end
end

"""
    log_ms_iteration(stats, σ, α, norm_y, Δ, npos, nzero, nneg, lin_status, is_descent)

Log one Moré-Sorensen iteration: the regularized linear system being solved
(parametrized by `σ`, the primal regularization, and `α`, the dual one),
the norm of the resulting dual step against the trust-region radius `Δ`,
the observed inertia of the augmented system, and the status reported by
the linear solver.
"""
function log_ms_iteration(stats, σ, α, norm_y, Δ, npos, nzero, nneg, lin_status, is_descent)
  return @sprintf(
    "            | %-7d%-12.2e%-12.2e%-12.2e%-12.2e%-12s%-12s%-12s",
    stats.iter,
    σ,
    α,
    norm_y,
    Δ,
    "($npos,$nzero,$nneg)",
    string(lin_status),
    is_descent ? "true" : "false",
  )
end

"""
    step_trend(ρk, η1, η2, active_watchdog)

Return a short arrow summarizing how the regularization parameter `σ` was
updated in response to `ρk`.
"""
function step_trend(ρk, η1, η2, active_watchdog)
  active_watchdog && return 'w'
  (ρk == Inf || ρk < η1) && return '↗'
  ρk < η2 && return '='
  return '↘'
end

function log_iteration(
  solver,
  nlp,
  stats;
  type = :outer_loop,
  η1 = nothing,
  η2 = nothing,
  active_watchdog = nothing,
)
  if type == :outer_loop
    return @sprintf(
      "%-7d%-7d%-+16.7e%-12.2e%-12.2e%-12.2e%-12.2e%-12.2e%-12.2e",
      stats.iter,
      max(solver.substats.iter, 0),
      stats.objective,
      stats.primal_feas,
      stats.dual_feas,
      solver.substats.solver_specific[:tau],
      solver.substats.solver_specific[:primal_ktol],
      solver.substats.solver_specific[:dual_ktol],
      norm(solver.x),
    )
  elseif type == :inner_loop
    trend =
      stats.iter == 0 ? "" :
      step_trend(stats.solver_specific[:rho], η1, η2, active_watchdog)
    return @sprintf(
      "      | %-7d%-7d%-+16.7e%-12.2e%-12.2e%-12.2e%-+12.2e%-6s%-12.2e%-12.2e",
      stats.iter,
      max(solver.substats.iter, 0),
      stats.objective,
      stats.primal_feas,
      stats.dual_feas,
      stats.solver_specific[:sigma],
      stats.solver_specific[:rho],
      trend,
      norm(solver.xk),
      norm(solver.s),
    )
  end
end

function conclusion_message(solver::MoreSorensenSolver, stats; type = :ms_loop)
  return "            |
                  |  Step computed with status $(stats.status) after $(stats.iter) iterations." *
         "\n      " *
         separator(type = type)
end

function conclusion_message(solver, nlp, stats; type = :outer_loop)
  if type == :outer_loop
    log = "\n"
    log *= "Number of Iterations: $(stats.iter)\n"
    log *= "\n\n"
    log *= @sprintf("Objective...........: %-+16.15e\n", stats.objective)
    log *= @sprintf("Primal Feasibility..:  %16.15e\n", stats.primal_feas)
    log *= @sprintf("Dual Feasibility....:  %16.15e\n", stats.dual_feas)
    log *= @sprintf("Complementarity.....:  %16.15e\n", solver.substats.solver_specific[:compl_error])
    log *= "\n\n"
    log *= "EXIT: $(stats.status).\n"
    return log
  elseif type == :inner_loop
    return "      |
            |  Subproblem solved with status $(stats.status) after $(stats.iter) iterations. 
            |  Reached dfeas = $(stats.dual_feas)\n      " * separator(type = type)
  end
end
