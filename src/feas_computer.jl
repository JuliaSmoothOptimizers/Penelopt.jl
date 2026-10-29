function compute_θ!(solver::L2PenaltySolver{T}) where {T}
  # Computes a model decrease for the feasbility problem minₓ ‖c(x)‖₂

  ## Retrieve workspace
  r2n_solver, r2n_stats = solver.subsolver, solver.substats

  ms_problem, ms_solver, ms_stats =
    r2n_solver.subpb, r2n_solver.subsolver, r2n_solver.substats
  ls_workspace = ms_solver.workspace
  u1, x1 = ms_solver.u1, ms_solver.x1
  nlp, ψ = ms_problem.model, ms_problem.h
  τ = ψ.h.lambda
  n, m = nlp.meta.nvar, length(ψ.b)

  # Step 1: Compute
  # ( I     Jᵀ )( s ) = - ( 0 )
  # ( J    -αI )( y ) = - ( c )
  @. u1[1:n] = 0
  @. u1[(n+1):(n+m)] = - ψ.b

  α = one(T)
  update_workspace!(ls_workspace, ψ.A, one(T), α)

  solve_system!(ls_workspace, u1)
  get_solution!(x1, ls_workspace)

  # With a CompactBFGSK2, solve_system! automatically performs the Sherman-Morrison update.
  # x1 is the preallocated space for the solution of the linear system with a block identity.
  # Refer to the CompactBFGSK2 code.
  if isa(ls_workspace.H, CompactBFGSK2Matrix)
    # ( I     Jᵀ )( s ) = - ( 0 )
    # ( J    -αI )( y ) = - ( c )
    x1 .= ls_workspace.H.x1
  end

  s, y = view(x1, 1:n), view(x1, (n+1):(n+m))

  # Step 2: Compute θ = (ψ(0; x) - ψ(s; x)) / (τ * ‖y‖₂)
  norm_y = norm(y, 2)
  iszero(norm_y) && return zero(T)

  θ = (ψ(solver.s0) - ψ(s)) / (τ * norm_y)

  set_solver_specific!(r2n_stats, :n_fact, get_n_fact(ls_workspace))

  return θ
end

function kkt_primal_feas!(solver::L2PenaltySolver{T}) where {T}
  return norm(solver.subsolver.subpb.h.b, Inf)
end

function initialize_multipliers!(solver::L2PenaltySolver{T}) where {T}

  ## Retrieve workspace
  r2n_solver, r2n_stats = solver.subsolver, solver.substats

  ms_problem, ms_solver, ms_stats =
    r2n_solver.subpb, r2n_solver.subsolver, r2n_solver.substats
  ls_workspace = ms_solver.workspace
  u1, x1 = ms_solver.u1, ms_solver.x1
  nlp, ψ = ms_problem.model, ms_problem.h
  n, m = nlp.meta.nvar, length(ψ.b)

  norm_∇f = norm(solver.∇fk)
  # Step 1: Compute
  # ( I     Jᵀ )( s ) = - ( ∇f )
  # ( J     0I )( y ) = - ( 0  )
  # If ∇f != 0, otherwise, solve
  # ( I     Jᵀ )( s ) = - ( 0 )
  # ( J     0I )( y ) = - ( c )
  @. u1[1:n] = - solver.∇fk
  @. u1[(n+1):(n+m)] = 0
  iszero(norm_∇f) && (@. u1[(n+1):(n+m)] = - ψ.b)

  α = eps(T)
  update_workspace!(ls_workspace, ψ.A, one(T), α)

  solve_system!(ls_workspace, u1)
  get_solution!(x1, ls_workspace)

  # With a CompactBFGSK2, solve_system! automatically performs the Sherman-Morrison update.
  # x1 is the preallocated space for the solution of the linear system with a block identity.
  # Refer to the CompactBFGSK2 code.
  if isa(ls_workspace.H, CompactBFGSK2Matrix)
    # ( I     Jᵀ )( s ) = - ( 0 )
    # ( J    -αI )( y ) = - ( c )
    x1 .= ls_workspace.H.x1
  end

  # Step 2: Check Jacobian full row rank and recompute if necessary
  status = get_status(ls_workspace)
  if status != :success
    α = eps(T)^(0.8)
    update_workspace!(ls_workspace, ψ.A, one(T), α)
    solve_system!(ls_workspace, u1)
    get_solution!(x1, ls_workspace)

    if isa(ls_workspace.H, CompactBFGSK2Matrix)
      # ( I     Jᵀ )( s ) = - ( 0 )
      # ( J    -αI )( y ) = - ( c )
      x1 .= ls_workspace.H.x1
    end
  end

  status = get_status(ls_workspace)
  if status != :success
    α = eps(T)^(0.6)
    update_workspace!(ls_workspace, ψ.A, one(T), α)
    solve_system!(ls_workspace, u1)
    get_solution!(x1, ls_workspace)

    if isa(ls_workspace.H, CompactBFGSK2Matrix)
      # ( I     Jᵀ )( s ) = - ( 0 )
      # ( J    -αI )( y ) = - ( c )
      x1 .= ls_workspace.H.x1
    end
  end

  # Step 3: Extract the solution
  s, y = view(x1, 1:n), view(x1, (n+1):(n+m))
  solver.y .= y

  # Set factorization counters
  set_solver_specific!(r2n_stats, :n_fact, get_n_fact(ls_workspace))
end

function kkt_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  return solver.substats.dual_feas
end

function least_square_dual_feas!(solver::L2PenaltySolver{T}) where {T}
  dual_res, y = solver.dual_res, solver.y
  g, J = solver.∇fk, solver.subsolver.subpb.h.A

  dual_res .= g
  mul!(dual_res, J', y, one(T), -one(T))
  return norm(dual_res, Inf)
end
