# Exact ℓ₂-penalty method for the regularized equality-constrained problem
#
#   minimize f(x) + h(x)  subject to  c(x) = 0,
#
# where h is proper, lsc and prox-bounded. Each penalized subproblem
#
#   minimize f(x) + τ‖c(x)‖₂ + h(x)
#
# is solved by Envelopt, whose augmented Lagrangian subproblems
#
#   minimize f(x) + e_{μ τ‖⋅‖₂}(c(x) + μy) + h(x)
#
# are solved by a solver from RegularizedOptimization.jl (R2N by default) that
# handles h through its proximal operator.
#
# The penalty loop is ported from `Envelopt.exact_penalty_solver`.

# Envelopt only needs h to be callable and to have a prox!, which it uses to evaluate the
# Moreau envelope and to update u. For h = λ‖⋅‖₂,
#   prox_{γh}(x) = max(1 - γλ / ‖x‖₂, 0) x.
function ProximalCore.prox!(y, f::NormL2, x, γ)
  normx = norm(x)
  γλ = γ * f.lambda
  if normx > γλ
    scale = 1 - γλ / normx
    y .= scale .* x
    return f.lambda * scale * normx
  end
  fill!(y, zero(eltype(y)))
  return zero(f.lambda * normx)
end

function Penelopt.EnveloptL2PenaltySolver(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  envelopt_subsolver = R2NEnveloptSubSolver,
  envelopt_subsolver_kwargs::NamedTuple = NamedTuple(),
) where {T,V}
  nlp = reg_nlp.model
  x = copy(nlp.meta.x0)
  ncon = get_ncon(nlp)
  c = similar(x, ncon)
  y_env = fill!(similar(x, ncon), zero(T))

  # f(x) and F(x) = c(x); the actual penalty parameter is set in solve!.
  # NormL2 is mutable, so τ can later be updated in place.
  env_model = EnveloptNLPModel(
    EqualityLessModel(nlp),
    EqualityFeasibilityModel(nlp),
    NormL2(one(T));
    x0 = x,
  )

  # The subsolver is tied to env_model, so it is allocated once and reused across
  # penalty updates.
  subsolver = envelopt_subsolver(
    env_model,
    reg_nlp.h;
    selected = reg_nlp.selected,
    envelopt_subsolver_kwargs...,
  )

  return EnveloptL2PenaltySolver(x, c, y_env, env_model, subsolver)
end

SolverCore.reset!(solver::EnveloptL2PenaltySolver) = solver
SolverCore.reset!(solver::EnveloptL2PenaltySolver, model) = SolverCore.reset!(solver)

# Set the convex term of env_model to τ‖⋅‖₂. The Moreau envelope of env_model holds a
# reference to the same mutable NormL2, so nothing needs to be rebuilt.
function set_l2_penalty!(env_model::EnveloptNLPModel{T}, τ::T) where {T}
  env_model.h.lambda = τ
  return env_model
end

function dist_inf(a::AbstractVector{T}, b::AbstractVector{T}) where {T}
  d = zero(T)
  @inbounds for i in eachindex(a, b)
    d = max(d, abs(a[i] - b[i]))
  end
  return d
end

function check_regularized_problem(nlp::AbstractNLPModel)
  equality_constrained(nlp) ||
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  has_bounds(nlp) && error(
    "L2Penalty: bounds, including fixed variables, are not supported together with a regularizer; include them in the regularizer instead.",
  )
end

# Entry points: L2Penalty(reg_nlp) or L2Penalty(nlp, h; selected).
function Penelopt.L2Penalty(
  reg_nlp::AbstractRegularizedNLPModel{T,V};
  envelopt_subsolver = R2NEnveloptSubSolver,
  envelopt_subsolver_kwargs::NamedTuple = NamedTuple(),
  kwargs...,
) where {T,V}
  nlp = reg_nlp.model
  check_regularized_problem(nlp)

  # Envelopt only uses the body of the constraints, so c(x) = b must become c(x) - b = 0.
  # Fixed variables are not removed and the problem is not scaled because both would
  # change the variables seen by h.
  shifted_nlp = remove_constraint_shift(nlp)
  reg_pb =
    shifted_nlp === nlp ? reg_nlp :
    RegularizedNLPModel(shifted_nlp, reg_nlp.h, reg_nlp.selected)

  solver = EnveloptL2PenaltySolver(
    reg_pb;
    envelopt_subsolver = envelopt_subsolver,
    envelopt_subsolver_kwargs = envelopt_subsolver_kwargs,
  )
  stats = RegularizedExecutionStats(reg_pb)
  SolverCore.solve!(solver, reg_pb, stats; kwargs...)
  return stats
end

function Penelopt.L2Penalty(nlp::AbstractNLPModel, h; selected = 1:get_nvar(nlp), kwargs...)
  return Penelopt.L2Penalty(RegularizedNLPModel(nlp, h, selected); kwargs...)
end

function SolverCore.solve!(
  solver::EnveloptL2PenaltySolver{T,V},
  reg_nlp::AbstractRegularizedNLPModel{T,V},
  stats::GenericExecutionStats{T,V};
  callback = (args...) -> nothing,
  x::V = reg_nlp.model.meta.x0,

  ## Termination arguments
  atol::T = √eps(T),
  rtol::T = √eps(T),
  max_iter::Int = 100,
  max_time::Float64 = 30.0,

  ## Logging arguments
  print_level::Int = 0,

  ## Outer loop specific arguments
  τ0::T = T(10),
  τ_increase::T = T(5),
  feas_decrease::T = T(0.9),

  ## Envelopt specific arguments
  envelopt_tol0::T = T(1e-2),
  envelopt_max_outer::Int = 20,
  envelopt_warm_start::Bool = true,
) where {T,V}
  SolverCore.reset!(stats)
  start_time = time()

  nlp = reg_nlp.model
  check_regularized_problem(nlp)
  any(!iszero, get_lcon(nlp)) && error(
    "L2Penalty: The problem has constraints of the form c(x) = b with b ≠ 0; see remove_constraint_shift.",
  )

  # Retrieve workspace
  env_model, subsolver = solver.env_model, solver.subsolver
  xk, c, y_env = solver.x, solver.c, solver.y_env
  xk .= x

  # Initialization
  cons!(nlp, xk, c)
  primal_feas = norm(c, Inf)
  primal_tol = atol + rtol * primal_feas
  dual_tol = atol  # made relative after the first penalized subproblem
  subtol = envelopt_tol0
  τ = τ0

  # Moreau envelope parameter and multiplier estimate of c(x) - u = 0, carried over between
  # penalty updates when envelopt_warm_start = true. The multiplier remains a valid estimate
  # when τ increases because it lies in ∂(τ‖⋅‖₂)(u) ⊂ τ𝔹₂.
  μ_env = one(T)
  fill!(y_env, zero(T))

  set_iter!(stats, 0)
  set_time!(stats, 0.0)
  fx = obj(nlp, xk)
  hx = Envelopt.regularizer_value(subsolver, xk)
  set_objective!(stats, fx + hx)
  set_solver_specific!(stats, :smooth_obj, fx)
  set_solver_specific!(stats, :nonsmooth_obj, hx)
  set_solver_specific!(stats, :tau, τ)
  set_residuals!(stats, primal_feas, T(Inf))

  if print_level > 0
    @info @sprintf "%4s  %7s  %7s  %8s  %7s  %7s  %7s  %4s  %6s" "iter" "τ" "subtol" "f + h" "dual" "primal" "‖step‖" "AL" "inner"
    @info @sprintf "%4d  %7.1e  %7.1e  %8.1e  %7s  %7.1e  %7s  %4s  %6s" 0 τ subtol stats.objective "-" primal_feas "-" "-" "-"
  end

  set_status!(
    stats,
    get_status(
      reg_nlp,
      elapsed_time = stats.elapsed_time,
      iter = stats.iter,
      max_time = max_time,
      max_iter = max_iter - 1,
    ),
  )

  callback(reg_nlp, solver, stats)

  done = stats.status != :unknown

  while !done
    # Solve the penalized subproblem minₓ f(x) + τ‖c(x)‖₂ + h(x) from xk.
    set_l2_penalty!(env_model, τ)
    get_x0(env_model) .= xk
    env_stats, env_status, _, inner_iter = envelopt(
      env_model;
      subsolver = subsolver,
      verbose = print_level > 1,
      max_outer = envelopt_max_outer,
      dtol_min = Float64(subtol),
      ptol_min = Float64(subtol),
      indent = "  ",
      init_penalty = μ_env,
      init_multiplier = y_env,
    )

    failed = env_status == :error
    if failed
      print_level > 0 &&
        @warn "Envelopt failed to solve the penalized subproblem" env_stats.status
    else
      if envelopt_warm_start
        μ_env = env_model.μ
        y_env .= env_model.y
      end

      # Check progress towards feasibility
      step_norm = dist_inf(env_stats.solution, xk)
      xk .= env_stats.solution
      cons!(nlp, xk, c)
      primal_feas_next = norm(c, Inf)
      kkt = max(env_stats.dual_feas, env_stats.primal_feas)

      if primal_feas_next > feas_decrease * primal_feas
        τ *= τ_increase
        subtol = max(min(dual_tol, primal_tol) / 2, min(kkt, subtol / 2))
      else
        subtol = max(min(dual_tol, primal_tol) / 2, min(kkt, subtol / 5))
      end
      primal_feas = primal_feas_next

      stats.iter == 0 && (dual_tol = atol + rtol * env_stats.dual_feas)

      fx = obj(nlp, xk)
      hx = Envelopt.regularizer_value(subsolver, xk)
      set_objective!(stats, fx + hx)
      set_solver_specific!(stats, :smooth_obj, fx)
      set_solver_specific!(stats, :nonsmooth_obj, hx)
      set_solver_specific!(stats, :tau, τ)
      set_residuals!(stats, primal_feas, env_stats.dual_feas)

      # Envelopt's convention: 0 ∈ ∇f(x) + J(x)ᵀy + ∂h(x).
      set_constraint_multipliers!(stats, env_model.y)
    end

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time() - start_time)

    if print_level > 0 && !failed
      @info @sprintf "%4d  %7.1e  %7.1e  %8.1e  %7.1e  %7.1e  %7.1e  %4d  %6d" stats.iter τ subtol stats.objective stats.dual_feas primal_feas step_norm env_stats.iter inner_iter
    end

    optimal =
      !failed &&
      env_stats.dual_feas ≤ dual_tol &&
      env_stats.primal_feas ≤ primal_tol &&
      primal_feas ≤ primal_tol

    set_status!(
      stats,
      get_status(
        reg_nlp,
        elapsed_time = stats.elapsed_time,
        iter = stats.iter,
        optimal = optimal,
        exception = failed,
        max_time = max_time,
        max_iter = max_iter - 1,
      ),
    )

    callback(reg_nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, xk)
  return stats
end
