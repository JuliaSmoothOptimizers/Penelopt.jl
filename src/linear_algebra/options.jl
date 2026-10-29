mutable struct PenaltyLinearSolverOptions{T1<:Real,T2<:Real}
  n_itref::Int
  n_precise::Int
  increase_pivtol_threshold::T1
  decrease_pivtol_threshold::T1
  max_pivtol::T2
  min_pivtol::T2
  decrease_pivtol_factor::T2
end

function PenaltyLinearSolverOptions(; 
  n_itref::Int = 5, 
  n_precise::Int = 20, 
  increase_pivtol_threshold::T1 = sqrt(eps(Float64)), 
  decrease_pivtol_threshold::T1 = 100 * eps(Float64), 
  max_pivtol::T2 = 1e-2, 
  min_pivtol::T2 = eps(Float64), 
  decrease_pivtol_factor::T2 = 0.1
) where {T1<:Real, T2<:Real}
  return PenaltyLinearSolverOptions{T1,T2}(
    n_itref, 
    n_precise, 
    increase_pivtol_threshold, 
    decrease_pivtol_threshold, 
    max_pivtol, 
    min_pivtol, 
    decrease_pivtol_factor
  )
end 

function set_params!(workspace::PenaltyDirectWorkspace; 
  n_itref::Int = 5, 
  n_precise::Int = 20, 
  increase_pivtol_threshold::T1 = sqrt(eps(Float64)), 
  decrease_pivtol_threshold::T1 = 100 * eps(Float64), 
  max_pivtol::T2 = 1e-2, 
  min_pivtol::T2 = eps(Float64), 
  decrease_pivtol_factor::T2 = 0.1
) where {T1<:Real, T2<:Real}
  workspace.options.n_itref = n_itref
  workspace.options.n_precise = n_precise
  workspace.options.increase_pivtol_threshold = increase_pivtol_threshold
  workspace.options.decrease_pivtol_threshold = decrease_pivtol_threshold
  workspace.options.max_pivtol = max_pivtol
  workspace.options.min_pivtol = min_pivtol
  workspace.options.decrease_pivtol_factor = decrease_pivtol_factor
end

function update_pivtol!(workspace::PenaltyDirectWorkspace)

  !factorized(workspace) && return

  opt = workspace.options
  relative_error = relative_error!(workspace)

  if relative_error > opt.increase_pivtol_threshold
    increase_pivtol!(workspace)
    force_iterative_refinement!(workspace)
    workspace.status = :failed
    set_n_precise_solve!(workspace, 0)

  elseif relative_error < opt.decrease_pivtol_threshold
    set_n_precise_solve!(workspace, get_n_precise_solve(workspace) + 1)
    if workspace._n_precise > opt.n_precise
      decrease_pivtol!(workspace)
      relax_iterative_refinement!(workspace)
    end

  else
    set_n_precise_solve!(workspace, 0)
  end
end

function increase_pivtol!(workspace::PenaltyDirectWorkspace)
  opt = workspace.options
  set_pivtol!(workspace, opt.max_pivtol)
end

function decrease_pivtol!(workspace::PenaltyDirectWorkspace)
  opt = workspace.options
  set_pivtol!(workspace, max(opt.min_pivtol, get_pivtol(workspace) * opt.decrease_pivtol_factor))
end

function set_pivtol!(workspace::PenaltyDirectWorkspace, pivtol)
  return
end

function get_pivtol(workspace::PenaltyDirectWorkspace)
  return 0.0
end

function relative_error!(workspace::PenaltyDirectWorkspace)
  return 0.0
end

function force_iterative_refinement!(workspace::PenaltyDirectWorkspace)
  return
end

function relax_iterative_refinement!(workspace::PenaltyDirectWorkspace)
  return
end

function set_n_precise_solve!(workspace::PenaltyDirectWorkspace, n::Int)
  return
end

function get_n_precise_solve(workspace::PenaltyDirectWorkspace)
  return
end