abstract type PenaltyWorkspace end
abstract type PenaltyDirectWorkspace <: PenaltyWorkspace end

function construct_workspace(H::M, u1::V, n::Int, m::Int; solver = :mumps) where {M,V}
  if solver == :ldlt
    return construct_ldlt_workspace(H, u1, n, m)
  elseif solver == :ma57
    return construct_ma57_workspace(H, u1, n, m)
  elseif solver == :mumps
    return construct_mumps_workspace(H, u1, n, m)
  else
    error(
      "Penelopt.jl: unknown linear solver `$(solver)`. Supported values are \"mumps\", \"ldlt\" and \"ma57\".",
    )
  end
end

function update_workspace!(
  workspace::PenaltyDirectWorkspace,
  B::M1,
  A::M2,
  σ::T,
  α::T;
) where {T, M1, M2}
  K = workspace.H
  initialize_primal_regularization!(K, σ)
  set_dual_regularization!(K, α)
  set_jacobian!(K, A)
  set_hessian!(K, B)
  workspace.factorized = false
end

function update_workspace!(
  workspace::PenaltyDirectWorkspace,
  A::M,
  σ::T,
  α::T;
) where {T, M}
  K = workspace.H
  reset_hessian!(K)
  set_jacobian!(K, A)
  set_primal_regularization!(K, σ)
  set_dual_regularization!(K, α)
  workspace.factorized = false
end

function update_workspace!(
  workspace::PenaltyDirectWorkspace,
  B::M1,
  A::M2,
  σ::T,
  α::T;
) where {T, M1<:CompactBFGS{T}, M2}
  K = workspace.H
  initialize_primal_regularization!(K, σ + B.ξ)
  set_jacobian!(K, A)
  set_dual_regularization!(K, α)
  K.σ = σ
end

# HSL Misc.: Checks whether the user has the license in the extension.
const _HSL_AVAILABLE = Ref(false)
hsl_functional() = _HSL_AVAILABLE[]
_set_hsl_available(flag::Bool) = (_HSL_AVAILABLE[] = flag)

function construct_ma57_workspace(H, u1, n, m)
  error("MA57 not available. Load PeneloptHSLExt.")
end

abstract type AbstractHSLWorkspace <: PenaltyDirectWorkspace end

# LDLFactorizations Misc.
function construct_ldlt_workspace(H, u1, n, m)
  error("LDLFactorizations not available. Load PeneloptLDLFactorizationsExt.")
end

abstract type AbstractLDLTWorkspace <: PenaltyDirectWorkspace end

# MUMPS Misc.
abstract type AbstractMUMPSWorkspace <: PenaltyDirectWorkspace end

get_n_fact(workspace::PenaltyDirectWorkspace) = workspace._n_fact

function set_n_fact!(workspace::PenaltyDirectWorkspace, n::Int)
  workspace._n_fact = n
end

function SolverCore.reset!(workspace::PenaltyWorkspace)
  set_n_fact!(workspace, 0)
end