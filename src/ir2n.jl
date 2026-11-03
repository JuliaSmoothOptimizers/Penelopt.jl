import SolverCore.solve!

mutable struct PenaltyR2NSolver{
  T<:Real,
  V<:AbstractVector{T},
  ST<:AbstractOptimizationSolver,
  PB<:AbstractShiftedPenalizedProblem,
  WC<:watchdog_checkpoint,
} <: AbstractOptimizationSolver
  xk::V
  y::V
  dual_res::V
  xkn::V
  s::V
  m_fh_hist::V
  subsolver::ST
  subpb::PB
  substats::GenericExecutionStats{T,V,V,T}
  checkpoint::WC
end

function PenaltyR2NSolver(
  penalty_nlp::AbstractPenalizedProblem{T,V};
  m_monotone::Int = 12,
  linear_solver::String = "mumps",
) where {T,V}
  x0 = penalty_nlp.model.meta.x0

  xk = similar(x0)
  ∇fk = similar(x0)
  y = zeros(T, get_ncon(penalty_nlp))
  dual_res = similar(x0)
  xkn = similar(x0)
  s = similar(x0)

  m_fh_hist = fill(T(-Inf), m_monotone - 1)

  subpb = shifted(penalty_nlp, xk, ∇f = ∇fk)
  substats = GenericExecutionStats(subpb, solver_specific = Dict{Symbol,T}())
  set_solver_specific!(substats, :alpha, 0)
  subsolver = MoreSorensenSolver(subpb; solver = Symbol(linear_solver))

  checkpoint = watchdog_checkpoint(subpb; m_monotone = m_monotone)

  return PenaltyR2NSolver{T,V,typeof(subsolver),typeof(subpb),typeof(checkpoint)}(
    xk,
    y,
    dual_res,
    xkn,
    s,
    m_fh_hist,
    subsolver,
    subpb,
    substats,
    checkpoint,
  )
end

function SolverCore.reset!(solver::PenaltyR2NSolver{T}) where {T}
  fill!(solver.m_fh_hist, T(-Inf))
  reset!(solver.subsolver)
  reset!(solver.subpb)
end

SolverCore.reset!(solver::PenaltyR2NSolver, model) = SolverCore.reset!(solver)

function SolverCore.solve!(
  solver::PenaltyR2NSolver{T,V},
  reg_nlp::AbstractPenalizedProblem{T,V},
  stats::PenaltyR2NExecutionStats{T,V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  print_level::Int = 0,
  verbose::Int = 1,
  max_iter::Int = 1000,
  ms_max_iter::Int = 10,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σmin::T = eps(T)^2,
  σ0::T = eps(T),
  σk::T = σ0,
  η1::T = √√eps(T),
  η2::T = isa(reg_nlp.model, QuasiNewtonModel) ? T(0.9) : T(0.1),
  γ::T = T(3),
  watchdog_max_iter::Int = 10,
  watchdog_η0::T = √eps(T),
  tiny_step_tol::T = eps(T),
  nmax_tiny_step::Int = 2,
  is_shifted::Bool = false,
  primal_decrease::Bool = false,
  first_increase::Bool = true,

  ## MS Specific arguments
  ms_verbose::Int = 1,
  ms_accept_descent::Bool = true,
  ms_σmax::T = 1/eps(T)^(0.8),
  ms_tol::T = eps(T)^(0.6),
  ms_μα::T = T(0.1),
  ms_μσ::T = T(10),
  ms_α0::T = eps(T),
  ms_αmin1::T = eps(T)^(0.8),
  ms_αmin2::T = eps(T)^(0.6),
  ms_ηC::T = eps(T),
) where {T,V}
  reset!(stats)
  σk = max(σk, σmin)

  # Retrieve workspace
  nlp, h = reg_nlp.model, reg_nlp.h
  mk = solver.subpb
  φ, ψ = mk.model, mk.h

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  !is_shifted && shift!(mk, xk)

  ∇fk = solver.subpb.model.data.c
  xkn = solver.xkn
  s, y, dual_res = solver.s, solver.y, solver.dual_res
  m_fh_hist = solver.m_fh_hist
  watchdog_checkpoint = solver.checkpoint

  m_monotone = length(m_fh_hist) + 1

  local ξ1::T
  local ρk::T = zero(T)

  # initialize parameters
  hk = !is_shifted ? h(xk) : h.h(ψ.b)
  h0 = copy(hk)
  fk = !is_shifted ? obj(nlp, xk) : stats.solver_specific[:smooth_obj]

  # Initialize stats
  set_iter!(stats, 0)
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fk + hk)
  set_solver_specific!(stats, :smooth_obj, fk)
  set_solver_specific!(stats, :nonsmooth_obj, hk)
  set_solver_specific!(stats, :sigma, σk)
  set_solver_specific!(stats, :rho, T(0))
  m_monotone > 1 && (m_fh_hist[stats.iter%(m_monotone-1)+1] = fk + hk)
  n_tiny_step = 0

  solved = false

  set_status!(
    stats,
    get_status(
      reg_nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter - 1,
    ),
  )

  ## Logging
  if print_level > 0
    @info introduction_message(solver, nlp, stats)
    @info separator(type = :inner_loop)
    @info header_message(type = :inner_loop)
    @info separator(type = :inner_loop)
    @info log_iteration(
      solver,
      nlp,
      stats;
      type = :inner_loop,
      η1 = η1,
      η2 = η2,
      active_watchdog = is_active(watchdog_checkpoint),
    )
  end

  callback(reg_nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    # Check stopping criteria
    dual_res .= ∇fk
    mul!(dual_res, ψ.A', y, one(T), one(T))
    set_primal_residual!(stats, norm(ψ.b, Inf))
    set_dual_residual!(stats, norm(dual_res, Inf))
    solved = stats.dual_feas ≤ atol

    if stats.iter == 0
      atol += stats.dual_feas * rtol
      set_solver_specific!(stats, :dual_ktol, atol)
    end

    solved = primal_decrease ? solved && hk < h0 : solved
    solved = solved && !is_active(watchdog_checkpoint)

    if solved
      set_status!(stats, :first_order)
      done = true
      continue
    end

    # Check the watchdog
    if check_watchdog!(watchdog_checkpoint, stats, mk, xk, watchdog_max_iter, watchdog_η0)
      fallback!(mk, xk, y, watchdog_checkpoint)
      φ.data.σ *= γ^watchdog_max_iter
      σk = φ.data.σ
      hk, fk = watchdog_checkpoint.hk, watchdog_checkpoint.fk
      m_fh_hist .= watchdog_checkpoint.m_fh_hist
      deactivate!(watchdog_checkpoint)
    end


    # Compute a step 
    solver.subpb.model.data.σ = σk
    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats;
      verbose = ms_verbose,
      print_level = print_level - 1,
      max_iter = ms_max_iter,
      accept_descent = ms_accept_descent,
      σmax = ms_σmax,
      atol = ms_tol,
      μα = ms_μα,
      μσ = ms_μσ,
      α0 = ms_α0,
      αmin1 = ms_αmin1,
      αmin2 = ms_αmin2,
      ηC = ms_ηC,
    )
    get_primal_dual_sol!(s, y, solver.subsolver)
    σk = solver.subpb.model.data.σ

    # Step acceptance
    xkn .= xk .+ s
    fkn, hkn = obj(nlp, xkn), h(xkn)
    ckn = h.b
    mks = dot(∇fk, s) + ψ(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fhmax)) * 10 * eps(T)
    Δmod = fhmax - (fk + mks) + max(1, abs(fhmax)) * 10 * eps(T)

    ρk = Δmod < 0 ? 0 : Δobj / Δmod

    ## Check step assumption 3.4
    pos_def_check = Bool(solver.substats.solver_specific[:is_pos_def])
    if !pos_def_check
      # sHs = (∇fᵀs + 0.5 * sᵀ(H + σI)s) - ∇fᵀs = 0.5 * sᵀ(H + σI)s
      sHs = obj(φ, s) - dot(∇fk, s)

      # Step Assumption 3.4 in implementation paper:
      # Δmod = f + τ‖c‖ - (f + ∇fᵀs + τ‖c + Js‖) ≥ 0.5 * ηC * sᵀ(H + σI)s
      # This assumption is only necessary when (H + σI) is not p.d.
      pos_def_check = Δmod >= 1/2 * sHs * ms_ηC
    end
    ρk = pos_def_check ? ρk : 0

    if η1 ≤ ρk < Inf
      xk .= xkn

      #update functions
      fk, hk = fkn, hkn

      shift!(mk, xk, y = y, c = ckn)

    end

    if η2 ≤ ρk < Inf
      σk = max(σk / γ, σmin)
    end

    if ρk < η1 || ρk == Inf
      if first_increase && ρk < 0
        σk = max(sqrt(stats.dual_feas), σk * γ)
        first_increase = false
      elseif ρk < 0 &&
             ρk > -1 / eps(T) &&
             !is_active(watchdog_checkpoint) &&
             !isa(nlp, NullHessianModel) # Watchdog procedure

        # Check acceptance w.r.t f
        d∇fks = dot(∇fk, s)
        fρk =
          d∇fks <= 0 ?
          (fk - fkn + max(1, abs(fk)) * 10 * eps())/(
            -d∇fks + max(1, abs(fk)) * 10 * eps()
          ) : zero(T)

        if η2 ≤ fρk < Inf # Activate watchdog
          activate!(watchdog_checkpoint)
          save!(watchdog_checkpoint, mk, xk, y, stats)
          watchdog_checkpoint.m_fh_hist .= m_fh_hist
          xk .= xkn

          #update functions
          fk, hk = fkn, hkn
          ckn = h.b

          shift!(mk, xk, y = y, c = ckn)

        else
          σk = σk * γ
        end
      else
        σk = σk * γ
      end
    end

    m_monotone > 1 && (m_fh_hist[stats.iter%(m_monotone-1)+1] = fk + hk)

    # Update stats
    set_objective!(stats, fk + hk)
    set_solver_specific!(stats, :smooth_obj, fk)
    set_solver_specific!(stats, :nonsmooth_obj, hk)
    set_solver_specific!(stats, :sigma, σk)
    set_solver_specific!(stats, :rho, ρk)
    set_solver_specific!(stats, :n_fact, get_n_fact(solver.subsolver.workspace))
    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    n_tiny_step =
      all(i -> abs(s[i]) < tiny_step_tol * abs(x[i]), eachindex(s, x)) ? n_tiny_step + 1 : 0

    set_status!(
      stats,
      get_status(
        reg_nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        unbounded = fk < - 1 / eps(T),
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter - 1,
        small_step = n_tiny_step > nmax_tiny_step,
      ),
    )

    ## Log status
    if print_level > 0 && stats.iter % verbose == 0
      if stats.iter % (20 * verbose) == 0 && stats.iter > 0
        @info separator(type = :inner_loop)
        @info header_message(type = :inner_loop)
        @info separator(type = :inner_loop)
      end
      @info log_iteration(
        solver,
        nlp,
        stats;
        type = :inner_loop,
        η1 = η1,
        η2 = η2,
        active_watchdog = is_active(watchdog_checkpoint),
      )
    end

    callback(reg_nlp, solver, stats)

    done = stats.status != :unknown
  end

  if print_level > 0
    @info conclusion_message(solver, nlp, stats; type = :inner_loop)
  end

  set_solution!(stats, xk)
  set_constraint_multipliers!(stats, y)
  set_residuals!(stats, zero(eltype(xk)), stats.dual_feas)
  return stats
end

# Utilities

"""
    set_penalty!(solver::PenaltyR2NSolver, τ)

Set the penalty parameter of the subproblem handled by `solver` to `τ`.
Both the shifted model and its parent `L2PenalizedProblem` are updated.
"""
set_penalty!(solver::PenaltyR2NSolver{T}, τ::T) where {T} = set_penalty!(solver.subpb, τ)

"""
    set_multipliers!(solver::PenaltyR2NSolver, y)

Set the Lagrange multiplier estimates used by `solver` to `y`,
with the convention ∇f(x) + J(x)ᵀy = 0.
"""
function set_multipliers!(solver::PenaltyR2NSolver, y::AbstractVector)
  solver.y .= y
  return solver
end

"""
    get_constraints(solver::PenaltyR2NSolver)

Return c(x) at the point around which the subproblem of `solver` is currently shifted.
"""
get_constraints(solver::PenaltyR2NSolver) = solver.subpb.h.b

"""
    get_jacobian(solver::PenaltyR2NSolver)

Return J(x) at the point around which the subproblem of `solver` is currently shifted.
"""
get_jacobian(solver::PenaltyR2NSolver) = solver.subpb.h.A

"""
    get_gradient(solver::PenaltyR2NSolver)

Return ∇f(x) at the point around which the subproblem of `solver` is currently shifted.
"""
get_gradient(solver::PenaltyR2NSolver) = solver.subpb.model.data.c

"""
    initialize_constraints!(solver::PenaltyR2NSolver, x)

Evaluate c(x) and J(x) and store them in the subproblem of `solver`.
"""
initialize_constraints!(solver::PenaltyR2NSolver, x::AbstractVector) = shift!(solver.subpb.h, x)

"""
    shift!(solver::PenaltyR2NSolver, x; ∇f = nothing, y = nothing, J = nothing, c = nothing)

Shift the subproblem of `solver` around `x`. The quantities that are passed are not reevaluated.
"""
shift!(solver::PenaltyR2NSolver, x::AbstractVector; kwargs...) =
  shift!(solver.subpb, x; kwargs...)

"""
    subsolver_kwargs(solver::PenaltyR2NSolver; kwargs...)

Translate the keyword arguments of the outer loop into keyword arguments of the R2N `solve!`.
Options prefixed with `r2n_` are passed without their prefix, options prefixed with `ms_` are
passed as is, and so are the warm-start arguments `σk`, `primal_decrease` and `first_increase`.
"""
function subsolver_kwargs(::PenaltyR2NSolver; kwargs...)
  r2n_kwargs = map(collect(kwargs)) do (key, val)
    name = String(key)
    if startswith(name, "r2n_")
      Symbol(name[5:end]) => val
    elseif startswith(name, "ms_") || key in (:σk, :primal_decrease, :first_increase)
      key => val
    else
      throw(ArgumentError("L2Penalty: unknown keyword argument `$key`."))
    end
  end
  return (; is_shifted = true, r2n_kwargs...)
end
