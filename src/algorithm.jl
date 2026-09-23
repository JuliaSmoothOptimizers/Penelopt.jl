export L2Penalty, L2PenaltySolver, solve!

import SolverCore.solve!

mutable struct L2PenaltySolver{
  T<:Real,
  V<:AbstractVector{T},
  S<:AbstractOptimizationSolver,
  PB<:L2PenalizedProblem,
} <: AbstractOptimizationSolver
  x::V
  xn::V
  y::V
  cn::V
  dual_res::V
  s::V
  s0::V
  ∇fk::V
  temp_b::V
  y_report::V
  subsolver::S
  subpb::PB
  substats::GenericExecutionStats{T,V,V,T}
end

"""
    solver = L2PenaltySolver(nlp; r2n_m_monotone = 12, linear_solver = "mumps")

Preallocate all the memory needed to solve `nlp` with the exact ℓ₂-penalty method.
The solver can then be passed to [`solve!`](@ref) any number of times without further
allocation, as long as the problem dimensions do not change. See [Preallocation](@ref).

Unlike [`L2Penalty`](@ref), no preprocessing is applied: if `nlp` has fixed variables or
shifted constraints, see [`remove_fixed_variables`](@ref) and [`remove_constraint_shift`](@ref).

# Keyword arguments
These options determine the size of the workspace and can only be passed here, not to `solve!`:
- `r2n_m_monotone::Int = 12`: non-monotone memory of the inner (R2N) solver;
- `linear_solver::String = "mumps"`: linear solver used for step computations.

See [Options Reference](@ref) for details.
"""
function L2PenaltySolver(
  nlp::AbstractNLPModel{T,V};
  r2n_m_monotone::Int = 12,
  linear_solver::String = "mumps",
) where {T,V}
  x0 = nlp.meta.x0
  x, xn, s, s0 = similar(x0), similar(x0), similar(x0), zero(x0)
  temp_b = similar(x0, nlp.meta.ncon)
  dual_res = similar(x0)
  cn = similar(x0, nlp.meta.ncon)
  y = similar(x0, nlp.meta.ncon)
  y_report = similar(x0, nlp.meta.ncon)
  ∇fk = similar(x0)

  penalty_subproblem = L2PenalizedProblem(nlp) # f(x) + τ‖c(x)‖₂
  substats = GenericExecutionStats(penalty_subproblem, solver_specific = Dict{Symbol,T}())
  solver = PenaltyR2NSolver(
    penalty_subproblem;
    m_monotone = r2n_m_monotone,
    linear_solver = linear_solver,
  )

  set_solver_specific!(substats, :primal_ktol, T(0))
  set_solver_specific!(substats, :dual_ktol, T(0))
  set_solver_specific!(substats, :n_fact, T(0))
  set_solver_specific!(substats, :tau, T(0))
  set_solver_specific!(substats, :sigma, T(0))
  set_solver_specific!(substats, :rho, T(0))
  set_solver_specific!(substats, :smooth_obj, T(0))
  set_solver_specific!(substats, :nonsmooth_obj, T(0))
  set_solver_specific!(substats, :compl_error, T(0))

  return L2PenaltySolver(
    x,
    xn,
    y,
    cn,
    dual_res,
    s,
    s0,
    ∇fk,
    temp_b,
    y_report,
    solver,
    penalty_subproblem,
    substats,
  )
end

function SolverCore.reset!(solver::L2PenaltySolver)
  SolverCore.reset!(solver.subsolver)
end

include("logging.jl")

"""
    stats = L2Penalty(nlp; kwargs...)

Solve the equality-constrained problem

    min f(x)  s.t.  c(x) = 0

described by the `AbstractNLPModel` `nlp` (see
[NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl)) with an exact
ℓ₂-penalty method. At each outer iteration, the nonsmooth subproblem

    min f(x) + τₖ‖c(x)‖₂

is solved approximately by a regularized Newton method (R2N), and the penalty parameter τₖ is
updated. Variables may not have bounds, except fixed variables (`lvar[i] == uvar[i]`).

Fixed variables and constraint right-hand sides are handled internally, and the problem is
scaled according to the scaling options; the returned solution is expressed in terms of the original `nlp`.

# Example

    using ADNLPModels, Penelopt
    nlp = ADNLPModel(x -> (x[1] - 1)^2 + x[2]^2, [2.0, 2.0], x -> [x[1] + x[2] - 1], [0.0], [0.0])
    stats = L2Penalty(nlp; print_level = 1)

# Keyword arguments
See [Options Reference](@ref) for the full list of options, [Callbacks](@ref) for the
`callback` keyword, and [Outputs](@ref) for a description of `stats`.
To reuse memory across several solves, see [`L2PenaltySolver`](@ref) and [`solve!`](@ref).
"""
function L2Penalty(
  nlp::AbstractNLPModel{T,V};
  r2n_m_monotone::Int = 12,
  linear_solver::String = "mumps",
  qn_hessian_approximation::String = "exact",
  qn_mem::Int = 6,
  qn_scaling::Bool = true,
  qn_max_skip::Int = 2,
  μ::T = T(0.1),
  kwargs...,
) where {T<:Real,V}

  # Check problem formulation
  if !equality_constrained(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end

  # Preprocessing
  preprocessed_nlp = nlp |> remove_fixed_variables |> remove_constraint_shift |> scale_model |> add_log_barrier

  if qn_hessian_approximation == "bfgs"
    preprocessed_nlp = CompactBFGSModel(
      preprocessed_nlp;
      mem = qn_mem,
      scaling = qn_scaling,
      max_skip = qn_max_skip,
    )
  elseif qn_hessian_approximation == "null"
    preprocessed_nlp = NullHessianModel(preprocessed_nlp)
  end

  # Preallocation
  solver = L2PenaltySolver(
    preprocessed_nlp;
    r2n_m_monotone = r2n_m_monotone,
    linear_solver = linear_solver,
  )
  stats = PeneloptExecutionStats(preprocessed_nlp)

  # Solve
  solve!(solver, preprocessed_nlp, stats; kwargs...)

  # Postprocess (in case there are fixed variables)
  stats.solution = recover_full_solution(preprocessed_nlp, stats.solution)

  return stats
end

"""
    solve!(solver::L2PenaltySolver, nlp, stats; kwargs...)

Solve `nlp` with the preallocated `solver`, writing the results in `stats`
(see [`PeneloptExecutionStats`](@ref)). `solver` must have been built from a problem with
the same dimensions as `nlp`.

All keyword arguments of [`L2Penalty`](@ref) are accepted, except `r2n_m_monotone`,
`linear_solver` and the `qn_*` options, which must be set when constructing
the solver or the model. See [Options Reference](@ref).

Call `SolverCore.reset!(solver)` between two solves to discard the state kept from the previous one.
"""
function SolverCore.solve!(
  solver::L2PenaltySolver{T,V},
  nlp::AbstractNLPModel{T,V},
  stats::GenericExecutionStats{T,V,V};
  callback = (args...) -> nothing,
  x::V = nlp.meta.x0,

  ## Termination arguments
  atol::T = √eps(T),
  rtol::T = √eps(T),
  dual_inf_atol::T = zero(T),
  dual_inf_rtol::T = zero(T),
  primal_inf_atol::T = zero(T),
  primal_inf_rtol::T = zero(T),
  compl_inf_atol::T = zero(T),
  compl_inf_rtol::T = zero(T),
  max_eval::Int = -1,
  max_time::Float64 = 30.0,
  max_iter::Int = 100,
  r2n_max_iter::Int = 1000,
  ms_max_iter::Int = 10,
  μ::T = T(1e-2),
  κε::T = T(10),
  infeasible_tol::T = T(1e-3),
  infeasible_iter::Int = 2,

  ## Logging arguments
  print_level::Int = 0,
  verbose::Int = 1,
  r2n_verbose::Int = 1,
  ms_verbose::Int = 1,

  ## Outer Loop specific arguments
  τmin::T = T(1),
  τ0::T = T(1),

  ## R2N Specific arguments
  r2n_η1::T = √√eps(T),
  r2n_η2::T = isa(nlp, QuasiNewtonModel) ? T(0.9) : T(0.1),
  r2n_σmin::T = eps(T)^2,
  r2n_γ::T = T(3),
  r2n_watchdog_max_iter::Int = 10,
  r2n_watchdog_η0::T = √eps(T),
  r2n_tiny_step_tol::T = eps(T),
  r2n_nmax_tiny_step::Int = 2,

  ## MS Specific arguments
  ms_accept_descent::Bool = true,
  ms_σmax::T = 1/eps(T)^(0.8),
  ms_tol::T = eps(T)^(0.6),
  ms_μα::T = T(0.1),
  ms_μσ::T = T(10),
  ms_α0::T = eps(T),
  ms_αmin1::T = eps(T)^(0.8),
  ms_αmin2::T = eps(T)^(0.6),
  ms_ηC::T = eps(T),

  ## Scaling arguments
  nlp_scaling_method::String = isa(nlp, QuasiNewtonModel) ? "gradient-based" : "none",
  gmax::T = T(100),
) where {T,V}
  reset!(stats)

  # Check that the problem has been correctly preprocessed
  if length(nlp.meta.ifix) > 0
    error(
      "L2Penalty: The problem has fixed variables. Refer to the documentation for information on how to preprocess the problem.",
    )
  end

  if !equality_constrained(nlp) || has_bounds(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end

  # Retrieve workspace
  penalty_pb = solver.subpb # f(x) + τ‖c(x)‖₂
  mk = solver.subsolver.subpb
  φ, ψ = mk.model, mk.h

  x = solver.x .= x
  y = solver.y

  barrier = get_barrier(find_model(LogBarrierModel, nlp))

  # TODO: users should be able to pass z_l_0 and z_u_0 as keyword arguments
  # Initialize z_l, z_u
  initialize_multipliers!(barrier)
  push_to_interior!(barrier, x)

  shift!(ψ, x)
  fx = obj(nlp, x)
  hx = norm(ψ.b)

  set_iter!(stats, 0)
  rem_eval = max_eval
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fx)

  ## Compute Feasibility

  primal_feas = kkt_primal_feas!(solver)

  set_solver_specific!(solver.substats, :smooth_obj, fx)
  grad!(nlp, x, solver.∇fk)
  initialize_multipliers!(solver)
  dual_feas = least_square_dual_feas!(solver)
  solver.subsolver.y .= solver.y

  compl_feas = compute_compl_error!(
      solver.subsolver.compl_res_l,
      solver.subsolver.compl_res_u,
      barrier,
      x,
    )

  primal_tol = max(primal_inf_atol, atol) + max(primal_inf_rtol, rtol) * primal_feas
  dual_tol = max(dual_inf_atol, atol) + max(dual_inf_rtol, rtol) * dual_feas
  compl_tol = max(compl_inf_atol, atol) + max(compl_inf_rtol, rtol) * compl_feas

  primal_ktol = one(primal_tol)
  dual_ktol = min(one(dual_tol), max(μ * dual_feas, dual_tol))
  dual_krtol = T(0)
  compl_ktol = compute_compl_ktol(barrier, κε)

  set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
  set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)
  set_residuals!(stats, dual_feas, primal_feas)

  solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol

  ## Scaling
  scaling_model = find_model(ScaledModel, nlp)
  if nlp_scaling_method == "gradient-based" && scaling_model !== nothing
    update_scaling!(scaling_model, solver.∇fk, ψ.A; gmax = gmax)
  end

  ## Initialize penalty parameter
  τ = max(norm(solver.y, 1), τ0)
  set_penalty!(mk, τ)
  νsub = 1 / r2n_σmin
  set_solver_specific!(solver.substats, :tau, τ)

  ## Logging
  if print_level > 0
    @info introduction_message(solver, nlp)
    @info separator()
    @info header_message()
    @info separator()
    @info log_iteration(solver, nlp, stats)
  end

  ## Initialize Model
  shift!(mk, x, ∇f = solver.∇fk, y = y, J = ψ.A, c = ψ.b)

  infeasible = false
  not_desc = false
  primal_decrease = false
  first_increase = true

  set_status!(
    stats,
    get_status(
      nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      optimal = solved,
      infeasible = infeasible,
      not_desc = not_desc,
      max_eval = max_eval,
      max_time = max_time,
      max_iter = max_iter - 1,
    ),
  )

  callback(nlp, solver, stats)

  done = stats.status != :unknown

  while !done

    solve!(
      solver.subsolver,
      solver.subpb,
      solver.substats;
      x = x,

      ## Termination arguments
      atol = max(dual_ktol, compl_ktol),
      rtol = dual_krtol,
      # compl_atol = compl_ktol,
      max_iter = r2n_max_iter,
      ms_max_iter = ms_max_iter,
      max_time = max_time - stats.elapsed_time,
      max_eval = rem_eval,

      ## Logging arguments
      print_level = print_level - 1,
      verbose = r2n_verbose,
      ms_verbose = ms_verbose,

      ## R2N Specific arguments
      σmin = r2n_σmin,
      σk = max(1 / νsub, r2n_σmin),
      η1 = r2n_η1,
      η2 = r2n_η2,
      γ = r2n_γ,
      watchdog_max_iter = r2n_watchdog_max_iter,
      watchdog_η0 = r2n_watchdog_η0,
      tiny_step_tol = r2n_tiny_step_tol,
      nmax_tiny_step = r2n_nmax_tiny_step,
      is_shifted = true,
      primal_decrease = primal_decrease,
      first_increase = first_increase,

      ## MS Specific arguments
      ms_accept_descent = ms_accept_descent,
      ms_σmax = ms_σmax,
      ms_tol = ms_tol,
      ms_μα = ms_μα,
      ms_μσ = ms_μσ,
      ms_α0 = ms_α0,
      ms_αmin1 = ms_αmin1,
      ms_αmin2 = ms_αmin2,
      ms_ηC = ms_ηC,
    )

    if solver.substats.status == :unbounded
      τ *= 10
      set_penalty!(mk, τ)
      νsub = 1 / r2n_σmin
      shift!(mk, x, y = y)
      set_solver_specific!(solver.substats, :smooth_obj, fx)
      set_solver_specific!(solver.substats, :tau, τ)
      continue
    end

    if solver.substats.status == :not_desc
      not_desc = true
    end

    x .= solver.substats.solution
    y .= solver.subsolver.y
    fx = solver.substats.solver_specific[:smooth_obj]
    hx_prev = copy(hx)
    hx = solver.substats.solver_specific[:nonsmooth_obj]/τ
    solver.∇fk .= φ.data.c

    ## Compute feasibility 
    primal_feas = kkt_primal_feas!(solver)
    dual_feas = kkt_dual_feas!(solver)

    compl_feas = compute_compl_error!(
      solver.subsolver.compl_res_l,
      solver.subsolver.compl_res_u,
      barrier,
      x,
    )

    if primal_feas > primal_ktol || (dual_ktol ≤ dual_tol && (primal_feas > primal_tol || compl_feas > compl_tol))
      # Update penalty parameter
      τ₊ = max(τ + τmin, norm(y, 1))
      if extrapolate!(x, solver, τ₊, τ)
        shift!(mk, x, y = y, c = solver.cn)

        # Subsolver: Do not impose primal decrease
        primal_decrease = false
      else

        # Subsolver: Impose primal decrease
        primal_decrease = true
      end
      τ = τ₊
      set_penalty!(mk, τ)

      # Initialize regularization parameter
      νsub = 1 / solver.substats.solver_specific[:sigma]

      # Subsolver: Activate the aggressive regularization parameter update if sigma is too small
      first_increase = true

      # Add a relative tolerance for the subsolver
      dual_ktol = dual_tol
      set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)
      set_solver_specific!(solver.substats, :tau, τ)
      dual_krtol = μ
    else
      # Tighten tolerances
      primal_ktol = max(μ*primal_feas, primal_tol)
      dual_ktol = max(μ*dual_feas, dual_tol)
      dual_krtol = T(0)
      set_solver_specific!(solver.substats, :primal_ktol, primal_ktol)
      set_solver_specific!(solver.substats, :dual_ktol, dual_ktol)

      y .= solver.substats.multipliers

      # Initialize regularization parameter
      νsub = 1/solver.substats.solver_specific[:sigma]

      # Subsolver: Do not impose primal decrease
      primal_decrease = false

      # Subsolver: Desactivate the aggressive regularization parameter update if sigma is too small
      first_increase = false
    end

    if compl_feas > compl_tol && !isnothing(barrier)
      # Update barrier parameter
      old_μ = get_barrier(barrier)
      update_barrier!(barrier, x, compl_tol)
      set_fraction_to_boundary!(barrier)
      compl_ktol = compute_compl_ktol(barrier, κε)

      # Update objective, gradient and hessian 
      solver.substats.solver_specific[:smooth_obj] += barrier(x) * (get_barrier(barrier) - old_μ)
      add_grad!(solver.∇fk, barrier, x, α = (get_barrier(barrier) - old_μ) / get_barrier(barrier) - old_μ)
      barrier_model = find_model(LogBarrierModel, nlp)
      # Updatte Hessian
      # TODO
    end

    # Check whether the primal feasibility has decreased. If not, increase the penalty parameter more aggressively.
    if primal_feas > primal_ktol && hx_prev < hx
      τmin *= 10
    end

    set_solver_specific!(solver.substats, :compl_error, compl_feas)

    solved = dual_feas ≤ dual_tol && primal_feas ≤ primal_tol && compl_feas ≤ compl_tol

    # Infeasiblity detection
    if stats.iter % infeasible_iter == 0
      θ = compute_θ!(solver)

      infeasible =
        hx > primal_tol &&
        sqrt(max(θ, 0))/hx < infeasible_tol &&
        sqrt(max(θ, 0)) < primal_ktol
    end

    set_iter!(stats, stats.iter + 1)
    rem_eval = max_eval - neval_obj(nlp)
    set_time!(stats, time() - start_time)
    set_objective!(stats, unscale_objective(scaling_model, fx))
    set_residuals!(stats, primal_feas, dual_feas)

    unscale_multipliers!(solver.y_report, scaling_model, solver.y)
    set_constraint_multipliers!(stats, solver.y_report)

    set_solver_specific!(stats, :n_fact, solver.substats.solver_specific[:n_fact])

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = solved,
        infeasible = infeasible,
        not_desc = not_desc,
        small_step = solver.substats.status == :small_step,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter - 1,
      ),
    )

    ## Log status
    if print_level > 0 && stats.iter % verbose == 0
      if stats.iter % (20 * verbose) == 0 && stats.iter > 0
        @info separator()
        @info header_message()
        @info separator()
      end
      @info log_iteration(solver, nlp, stats)
    end

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  if print_level > 0
    @info conclusion_message(solver, nlp, stats)
  end

  set_solution!(stats, x)
  return stats
end
