import Base.show

mutable struct MoreSorensenSolver{
  T<:Real,
  V<:AbstractVector{T},
  L<:Union{AbstractLinearOperator,AbstractMatrix},
  W,
} <: AbstractPenalizedProblemSolver
  u1::V
  u2::V
  x1::V
  x2::V
  H::L
  workspace::W
end

function MoreSorensenSolver(
  reg_nlp::AbstractShiftedPenalizedProblem{T,V};
  solver = :mumps,
) where {T,V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)
  x1 = zeros(eltype(x0), n+m)
  x2 = zeros(eltype(x0), n+m)

  # Check linear solver

  # Check for LDLFactorizations
  ldlt_loaded = !isnothing(Base.get_extension(@__MODULE__, :PeneloptLDLFactorizationsExt))
  if !ldlt_loaded && solver == :ldlt
    @warn "Penelopt.jl: LDLFactorizations extension is not loaded. Please install LDLFactorizations.jl. Switching to MUMPS..."
    solver = :mumps
  end

  # Check for HSL
  hsl_loaded = !isnothing(Base.get_extension(@__MODULE__, :PeneloptHSLExt))
  if !hsl_loaded && solver == :ma57
    @warn "Penelopt.jl: HSL extension is not loaded. Please install HSL.jl. Switching to MUMPS..."
    solver = :mumps
  end

  hsl_isfunctional = hsl_loaded && hsl_functional()
  if !hsl_isfunctional && solver == :ma57
    @warn "Penelopt.jl: HSL extension is not functional. Please check your license and make sure you have loaded HSL_jll.jl appropriately. Switching to MUMPS..."
    solver = :mumps
  end

  H = K2(
    n,
    m,
    n+m,
    n+m,
    zero(T),
    reg_nlp.model.data.σ,
    reg_nlp.h.A,
    reg_nlp.model.data.H;
    format = solver ∈ [:ma57, :mumps] ? :coo : :csc,
    int_type = solver == :mumps ? Int32 : Int64,
  )
  workspace = construct_workspace(H, u1, n, m; solver = solver)

  return MoreSorensenSolver(u1, u2, x1, x2, H, workspace)
end

"""
    factor_and_solve!(solver_workspace, u1, x1)

Factorize the system currently held in `solver_workspace`, solve it for the
right-hand side `u1`, store the resulting primal-dual solution in `x1`, and
return `(npos, nzero, nneg, status)`: the observed inertia of the factorized
system and the status reported by the linear solver.
"""
function factor_and_solve!(solver_workspace, u1, x1)
  solve_system!(solver_workspace, u1)
  get_solution!(x1, solver_workspace)
  npos, nzero, nneg = get_inertia(solver_workspace)
  status = get_status(solver_workspace)
  return npos, nzero, nneg, status
end

"""
    escalate_and_resolve!(solver, reg_nlp, stats, αmin, failure_status, opts)

Multiply the primal regularization `reg_nlp.model.data.σ` by `opts.μσ` and
either report failure — setting `stats.status` to `failure_status` — if `σ`
has reached `opts.σmax`, or restart the Moré-Sorensen iteration with the new
`σ`. `opts` is the `NamedTuple` of tuning parameters built once at the top
of `solve!` (everything but `α0`/`_restart`, which vary across restarts) and
is forwarded as-is to the recursive `solve!` call, only initializing the
dual regularization at `αmin` (the smallest value already known to work for
the current system) instead of retrying it from `0`.
"""
function escalate_and_resolve!(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats,
  αmin::T,
  failure_status::Symbol,
  opts::NamedTuple,
) where {T,V,M,H,P}
  reg_nlp.model.data.σ *= opts.μσ
  if reg_nlp.model.data.σ >= opts.σmax
    set_status!(stats, failure_status)
    opts.print_level > 0 && @info conclusion_message(solver, stats)
    return
  end
  return solve!(solver, reg_nlp, stats; opts..., α0 = αmin, _restart = true)
end

function SolverCore.solve!( #TODO add verbose and kwargs
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P},
  stats::GenericExecutionStats{T,V,V};
  x = reg_nlp.model.meta.x0,
  print_level::Int = 0,
  verbose::Int = 1,
  atol::T = eps(T)^(0.6),
  max_time::T = T(30),
  max_iter::Int = 10,
  μα::T = T(0.1),
  μσ::T = T(10),
  α0::T = eps(T),
  αmin1::T = eps(T)^(0.8),
  αmin2::T = eps(T)^(0.6),
  σmax::T = 1 / eps(T)^(0.8),
  accept_descent::Bool = true, # Whether we accept inexact steps that decrease the quadratic model.
  ηC::T = eps(T), # Cauchy decrease acceptance constant, see up_lb_is_pos_def fallback below.
  _restart::Bool = false, # Internal flag only: true when this call is a recursive continuation.
) where {T,V,M,H,P}
  if !_restart
    set_time!(stats, 0.0)
    set_iter!(stats, 0)
    if print_level > 0
      @info separator(type = :ms_loop)
      @info header_message(type = :ms_loop)
      @info separator(type = :ms_loop)
    end
  end
  start_time = time() - stats.elapsed_time

  # Bundled once so the escalation call sites below stay short: everything
  # solve! was called with, except α0/_restart which change across restarts.
  opts = (;
    x,
    print_level,
    verbose,
    atol,
    max_time,
    max_iter,
    μα,
    μσ,
    αmin1,
    αmin2,
    σmax,
    accept_descent,
    ηC,
  )

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda

  u1, u2, x1, x2 = solver.u1, solver.u2, solver.x1, solver.x2
  solver_workspace = solver.workspace

  # Create problem
  @. u1[1:n] = -reg_nlp.model.data.c
  @. u1[(n+1):(n+m)] = -reg_nlp.h.b

  α = α0
  update_workspace!(
    solver_workspace,
    reg_nlp.model.data.H,
    reg_nlp.h.A,
    reg_nlp.model.data.σ,
    α,
  )

  αmin = αmin1

  # [ H + σI Aᵀ][x] = -[∇f]
  # [   A    0 ][y] = -[c] 
  npos, nzero, nneg, status = factor_and_solve!(solver_workspace, u1, x1)

  # Get correct inertia
  # If the factorization/solver failed, it in indicates we should add a minimal regularization too.
  if nneg < m || status == :failed
    α = αmin
    set_dual_inertia!(solver_workspace, α)
    npos, nzero, nneg, status = factor_and_solve!(solver_workspace, u1, x1)

    if nneg < m || status == :failed
      αmin = αmin2
      α = αmin
      set_dual_inertia!(solver_workspace, α)
      npos, nzero, nneg, status = factor_and_solve!(solver_workspace, u1, x1)
    end
  end

  while (npos < n || status == :failed) && reg_nlp.model.data.σ <= σmax
    reg_nlp.model.data.σ *= μσ
    set_primal_inertia!(solver_workspace, reg_nlp.model.data.σ)

    # [ H + σI Aᵀ][x] = -[∇f]
    # [   A    0 ][y] = -[c] 
    npos, nzero, nneg, status = factor_and_solve!(solver_workspace, u1, x1)
  end

  # Solve with H = 0 instead.
  if reg_nlp.model.data.σ > σmax && (npos < n || status == :failed)
    isa(reg_nlp.model.data.H, CompactBFGS) ? NLPModels.reset!(reg_nlp.model.data.H) :
    reg_nlp.model.data.H.vals .= 0
    update_workspace!(solver_workspace, reg_nlp.h.A, reg_nlp.model.data.σ, α)
    # [ σI Aᵀ][x] = -[∇f]
    # [ A  0 ][y] = -[c] 
    npos, nzero, nneg, status = factor_and_solve!(solver_workspace, u1, x1)
  end

  if (npos < n || status == :failed)
    set_status!(stats, :exception)
    return
  end

  is_descent = check_descent(reg_nlp, @view x1[1:n])
  norm_x1 = norm(@view x1[(n+1):(n+m)])

  if print_level > 0 && stats.iter % verbose == 0
    @info log_ms_iteration(
      stats,
      reg_nlp.model.data.σ,
      α,
      norm_x1,
      Δ,
      npos,
      nzero,
      nneg,
      status,
      is_descent,
    )
  end

  is_pos_def = up_lb_is_pos_def(solver_workspace)
  if norm_x1 <= Δ || (is_descent && accept_descent)
    if is_pos_def ||
       (is_descent && accept_descent && check_cauchy_decrease(solver, reg_nlp; ηC = ηC))
      set_solution!(stats, @view x1[1:n])
      set_status!(stats, :first_order)

      !is_descent && set_status!(stats, :not_desc)
      set_solver_specific!(stats, :alpha, α)
      print_level > 0 && @info conclusion_message(solver, stats)

      return
    end
  end

  if !is_pos_def
    return escalate_and_resolve!(solver, reg_nlp, stats, αmin, :exception, opts)
  end

  # [ H + σI Aᵀ][x'] = -[0]
  # [   A    0 ][y'] = -[x] 
  @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
  solve_system!(solver_workspace, u2)
  get_solution!(x2, solver_workspace)

  while abs(norm_x1 - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    # α = α + (‖y‖/Δ - 1)*‖y‖²/(yᵀy')
    @views α₊ = α + norm_x1^2/dot(x1[(n+1):(n+m)], x2[(n+1):(n+m)])*(norm_x1/Δ - 1)

    α = α₊ ≤ 0 ? max(μα*α, αmin) : α₊
    set_dual_inertia!(solver_workspace, α)

    # [ H + σI  Aᵀ ][x] = -[∇f]
    # [   A    -αI ][y] = -[c] 
    solve_system!(solver_workspace, u1)
    get_solution!(x1, solver_workspace)

    # Check whether x1 decreases the model.
    is_descent = check_descent(reg_nlp, @view x1[1:n])
    norm_x1 = norm(@view x1[(n+1):(n+m)])

    if is_descent && accept_descent
      set_solution!(stats, @view x1[1:n])
      set_status!(stats, :first_order)
      set_solver_specific!(stats, :alpha, α)
      set_iter!(stats, stats.iter + 1)
      if print_level > 0 && stats.iter % verbose == 0
        @info log_ms_iteration(
          stats,
          reg_nlp.model.data.σ,
          α,
          norm_x1,
          Δ,
          npos,
          nzero,
          nneg,
          status,
          is_descent,
        )
      end
      print_level > 0 && @info conclusion_message(solver, stats)
      return
    end

    # Check whether the matrix still has the correct inertia. (We may have failed to detect earlier)
    npos, nzero, nneg = get_inertia(solver_workspace)
    if npos < n
      return escalate_and_resolve!(solver, reg_nlp, stats, αmin, :exception, opts)
    end

    # [ H + σI  Aᵀ ][x'] = -[0]
    # [   A    -αI ][y'] = -[x]
    @views @. u2[(n+1):(n+m)] = -x1[(n+1):(n+m)]
    solve_system!(solver_workspace, u2)
    get_solution!(x2, solver_workspace)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)

    if print_level > 0 && stats.iter % verbose == 0
      @info log_ms_iteration(
        stats,
        reg_nlp.model.data.σ,
        α,
        norm_x1,
        Δ,
        npos,
        nzero,
        nneg,
        status,
        is_descent,
      )
    end

    α == αmin && break
  end

  set_solution!(stats, @view x1[1:n])
  set_status!(stats, :first_order)
  set_solver_specific!(stats, :alpha, α)

  stats.iter >= max_iter && set_status!(stats, :max_iter)
  stats.elapsed_time >= max_time && set_status!(stats, :max_time)
  !check_descent(reg_nlp, @view x1[1:n]) && set_status!(stats, :not_desc)
  if !check_descent(reg_nlp, @view x1[1:n])
    return escalate_and_resolve!(solver, reg_nlp, stats, αmin, :not_desc, opts)
  end
end

function get_primal_dual_sol!(s, y, solver::MoreSorensenSolver)
  n = length(s)
  s .= @view solver.x1[1:n]
  y .= @view solver.x1[(n+1):end]
end

"""
    check_cauchy_decrease(solver::MoreSorensenSolver, reg_nlp::ShiftedL2PenalizedProblem; ηC = ...)

Checks the Cauchy decrease condition on the current primal-dual step
`solver.x1`, i.e. whether

    τₖ‖c(xₗ)‖₂ - ∇f(xₗ)ᵀsₗ - τₖ‖c(xₗ) + J(xₗ)sₗ‖₂ ≥ ηC ‖(Hₗ+σₗI)sₗ‖₂²

holds, where `sₗ = solver.x1[1:n]` is the primal step and
`yₗ = solver.x1[(n+1):(n+m)]` the associated dual step from the KKT system
just solved (`Hₗ` is the current Hessian approximation and `σₗ` the current
primal regularization parameter, `reg_nlp.model.data.σ`). If this holds,
`sₗ` is an acceptable step even though `Hₗ+σₗI` was not confirmed to be
positive definite by `up_lb_is_pos_def`.
"""
function check_cauchy_decrease(
  solver::MoreSorensenSolver{T,V},
  reg_nlp::ShiftedL2PenalizedProblem{T,V,M,H,P};
  ηC::T = T(1e-2),
) where {T,V,M,H,P}
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  x1 = solver.x1

  s = @view x1[1:n]
  y = @view x1[(n+1):(n+m)]
  ∇f = reg_nlp.model.data.c
  τ = reg_nlp.h.h.lambda

  model_decrease = τ * norm(reg_nlp.h.b) - dot(∇f, s) - reg_nlp.h(s)

  Hs = reg_nlp.model.data.v
  mul!(Hs, reg_nlp.h.A', y)
  @. Hs += ∇f
  # ‖(Hₗ+σₗI)sₗ‖₂² = ‖-∇f(xₗ) - J(xₗ)ᵀyₗ‖₂² = ‖Hs‖₂² regardless of sign.

  return model_decrease >= ηC * dot(Hs, Hs)
end

function SolverCore.reset!(solver::MoreSorensenSolver{T}) where {T}
  reset!(solver.workspace)
end
