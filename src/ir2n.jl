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
  compl_res_l::V
  compl_res_u::V
  xkn::V
  s::V
  s_z_l::V
  s_z_u::V
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

  barrier = find_model(LogBarrierModel, penalty_nlp.model)
  s_z_l = isnothing(barrier) ? T[] : similar(barrier.ϕ.z_l) 
  s_z_u = isnothing(barrier) ? T[] : similar(barrier.ϕ.z_u)
  compl_res_l = isnothing(barrier) ? T[] : similar(barrier.ϕ.z_l)
  compl_res_u = isnothing(barrier) ? T[] : similar(barrier.ϕ.z_u)

  return PenaltyR2NSolver{T,V,typeof(subsolver),typeof(subpb),typeof(checkpoint)}(
    xk,
    y,
    dual_res,
    compl_res_l,
    compl_res_u,
    xkn,
    s,
    s_z_l,
    s_z_u,
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
  stats::GenericExecutionStats{T,V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  compl_atol::T = √eps(T),
  print_level::Int = 0,
  verbose::Int = 0,
  max_iter::Int = 1000,
  ms_max_iter::Int = 10,
  max_time::Float64 = 30.0,
  max_eval::Int = -1,
  σk::T = eps(T)^(1 / 5),
  σmin::T = eps(T)^2,
  η1::T = √√eps(T),
  η2::T = T(0.1),
  γ::T = T(3),
  watchdog_max_iter::Int = 10,
  watchdog_η0::T = eps(T),
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

  # Retrieve workspace
  nlp, h = reg_nlp.model, reg_nlp.h
  mk, barrier = solver.subpb, get_barrier(find_model(LogBarrierModel, nlp))
  φ, ψ = mk.model, mk.h

  xk = solver.xk .= x

  # Make sure ψ has the correct shift 
  !is_shifted && shift!(mk, xk, compute_grad = compute_grad)

  ∇fk = solver.subpb.model.data.c
  xkn = solver.xkn
  s, y, dual_res = solver.s, solver.y, solver.dual_res
  compl_res_l, compl_res_u = solver.compl_res_l, solver.compl_res_u
  s_z_l, s_z_u = solver.s_z_l, solver.s_z_u
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

    compl_error = compute_mu_compl_error!(
      compl_res_l,
      compl_res_u,
      barrier,
      xk,
    )

    set_primal_residual!(stats, norm(ψ.b, Inf))
    set_dual_residual!(stats, norm(dual_res, Inf))
    solved = stats.dual_feas ≤ atol && compl_error ≤ atol

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

    get_z_l_step!(s_z_l, barrier, xk, s)
    get_z_u_step!(s_z_u, barrier, xk, s)

    truncate_to_boundary!(
      s,
      s_z_l,
      s_z_u,
      xk,
      barrier,
    )

    # Step acceptance
    xkn .= xk .+ s
    fkn, hkn = obj(nlp, xkn), h(xkn)
    ckn = h.b
    mks = dot(∇fk, s) + ψ(s)

    fhmax = m_monotone > 1 ? maximum(m_fh_hist) : fk + hk
    Δobj = fhmax - (fkn + hkn) + max(1, abs(fhmax)) * 10 * eps(T)
    Δmod = fhmax - (fk + mks) + max(1, abs(fhmax)) * 10 * eps(T)

    ρk = Δmod < 0 ? 0 : Δobj / Δmod

    if η1 ≤ ρk < Inf
      xk .= xkn

      #update bound multipliers
      update_multipliers!(barrier, s_z_l, s_z_u)

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
