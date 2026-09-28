"""
    remove_constraint_shift.jl

Standalone preprocessing step that reformulates equality constraints
written as `c_i(x) = v_i` (i.e. `lcon[i] == ucon[i] == v_i != 0`) into the
standard form `c_i(x) - v_i = 0` expected by solvers such as
ExactPenalty.jl (see https://github.com/JuliaSmoothOptimizers/Penelopt.jl/issues/186).

Inequality constraints, and equality constraints whose target is already
`0`, are left untouched.

    julia> nlp = MyNLPModel(...)   # some c_i(x) = v_i, v_i != 0
    julia> nlp_shifted = remove_constraint_shift(nlp)
    julia> stats = my_solver(nlp_shifted)   # sees c(x) - v = 0 everywhere

    # to recover the value of the *original* constraint function at a point:
    julia> c_orig = cons(nlp_shifted, stats.solution) .+ constraint_shift(nlp_shifted)

Composes cleanly with `remove_fixed_variables.jl` and `scale_model.jl`;
apply in whatever order is convenient (shifting does not change the
Jacobian/Hessian sparsity or values, only the constraint *values* and
`lcon`/`ucon`).
"""

export ShiftedConstraintModel, remove_constraint_shift, constraint_shift

"""
    ShiftedConstraintModel(nlp, shift) <: AbstractNLPModel{T,S}

Wraps `nlp` and exposes `c(x) - shift` in place of `c(x)`, with `lcon`/`ucon`
adjusted accordingly. Do not construct directly; use
[`remove_constraint_shift`](@ref).
"""
struct ShiftedConstraintModel{
  T,
  S<:AbstractVector{T},
  M<:AbstractNLPModel{T,S},
  Meta<:AbstractNLPModelMeta{T,S},
} <: AbstractNLPModel{T,S}
  meta::Meta
  counters::Counters
  model::M
  shift::S  # length ncon(nlp); 0 for constraints that did not need shifting
end

get_model(nlp::ShiftedConstraintModel) = nlp.model

"""
    remove_constraint_shift(nlp::AbstractNLPModel)

Detect equality constraints of the form `c_i(x) = v_i` with `v_i != 0`
(`lcon[i] == ucon[i] != 0`) and reformulate them as `c_i(x) - v_i = 0`,
by wrapping `nlp` in a model whose `cons!` subtracts `v_i` and whose
`lcon[i]`/`ucon[i]` become `0`. Inequality constraints and equality
constraints already at `0` are left untouched.

The Jacobian and Hessian are unaffected by this reformulation (subtracting
a constant does not change derivatives), so they are delegated to `nlp`
unchanged.

If no constraint needs shifting (including if `nlp` is unconstrained),
`nlp` itself is returned unchanged.

See [`constraint_shift`](@ref) to recover the original constraint values,
and https://github.com/JuliaSmoothOptimizers/Penelopt.jl/issues/186 for context.
"""
function remove_constraint_shift(nlp::AbstractNLPModel{T,S}) where {T,S}
  ncon = get_ncon(nlp)
  if ncon == 0
    return nlp
  end

  lcon = get_lcon(nlp)
  ucon = get_ucon(nlp)
  is_equality = lcon .== ucon
  needs_shift = is_equality .& (lcon .!= 0)

  if !any(needs_shift)
    return nlp
  end

  shift = zeros(T, ncon)
  shift[needs_shift] .= lcon[needs_shift]

  new_lcon = copy(lcon)
  new_ucon = copy(ucon)
  new_lcon[needs_shift] .= 0
  new_ucon[needs_shift] .= 0

  meta = NLPModelMeta(
    get_nvar(nlp);
    x0 = get_x0(nlp),
    lvar = get_lvar(nlp),
    uvar = get_uvar(nlp),
    ncon = ncon,
    y0 = get_y0(nlp),
    lcon = new_lcon,
    ucon = new_ucon,
    nnzj = get_nnzj(nlp.meta),
    nnzh = get_nnzh(nlp.meta),
    lin = nlp.meta.lin,
    minimize = nlp.meta.minimize,
    islp = false,
    name = string(nlp.meta.name, " (constraint shift removed)"),
  )

  return ShiftedConstraintModel(meta, Counters(), nlp, shift)
end

"""
    constraint_shift(nlp)

Return the shift vector `v` such that the *original* constraints satisfy
`c_orig(x) = c(x) + v`, where `c(x)` is what `nlp` itself now reports.
Returns a vector of zeros for a plain `AbstractNLPModel` (i.e. one that
was left untouched by [`remove_constraint_shift`](@ref)).
"""
constraint_shift(nlp::ShiftedConstraintModel) = nlp.shift
constraint_shift(nlp::AbstractNLPModel{T}) where {T} = zeros(T, get_ncon(nlp))

# ------------------------------------------------------------------------
# NLPModels API
#
# Only `cons!`/`cons` and the bounds (already handled in `meta`) change.
# Objective, Jacobian and Hessian are delegated to `nlp` unchanged, since
# a constant shift of the constraints does not affect any derivative.
# ------------------------------------------------------------------------

function NLPModels.obj(nlp::ShiftedConstraintModel, x::AbstractVector)
  NLPModels.increment!(nlp, :neval_obj)
  return NLPModels.obj(nlp.model, x)
end

function NLPModels.grad!(nlp::ShiftedConstraintModel, x::AbstractVector, g::AbstractVector)
  NLPModels.increment!(nlp, :neval_grad)
  NLPModels.grad!(nlp.model, x, g)
  return g
end

function NLPModels.cons!(nlp::ShiftedConstraintModel, x::AbstractVector, c::AbstractVector)
  NLPModels.increment!(nlp, :neval_cons)
  NLPModels.cons!(nlp.model, x, c)
  c .-= nlp.shift
  return c
end

function NLPModels.jac_structure!(
  nlp::ShiftedConstraintModel,
  rows::AbstractVector,
  cols::AbstractVector,
)
  NLPModels.jac_structure!(nlp.model, rows, cols)
  return rows, cols
end

function NLPModels.jac_coord!(
  nlp::ShiftedConstraintModel,
  x::AbstractVector,
  vals::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jac)
  NLPModels.jac_coord!(nlp.model, x, vals)
  return vals
end

function NLPModels.jprod!(
  nlp::ShiftedConstraintModel,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jprod)
  NLPModels.jprod!(nlp.model, x, v, Jv)
  return Jv
end

function NLPModels.jtprod!(
  nlp::ShiftedConstraintModel,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  NLPModels.increment!(nlp, :neval_jtprod)
  NLPModels.jtprod!(nlp.model, x, v, Jtv)
  return Jtv
end

function NLPModels.hess_structure!(
  nlp::ShiftedConstraintModel,
  rows::AbstractVector,
  cols::AbstractVector,
)
  NLPModels.hess_structure!(nlp.model, rows, cols)
  return rows, cols
end

function NLPModels.hess_coord!(
  nlp::ShiftedConstraintModel{T},
  x::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  NLPModels.hess_coord!(nlp.model, x, vals; obj_weight = obj_weight)
  return vals
end

function NLPModels.hess_coord!(
  nlp::ShiftedConstraintModel{T},
  x::AbstractVector,
  y::AbstractVector,
  vals::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hess)
  NLPModels.hess_coord!(nlp.model, x, y, vals; obj_weight = obj_weight)
  return vals
end

function NLPModels.hprod!(
  nlp::ShiftedConstraintModel{T},
  x::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hprod)
  NLPModels.hprod!(nlp.model, x, v, Hv; obj_weight = obj_weight)
  return Hv
end

function NLPModels.hprod!(
  nlp::ShiftedConstraintModel{T},
  x::AbstractVector,
  y::AbstractVector,
  v::AbstractVector,
  Hv::AbstractVector;
  obj_weight = one(T),
) where {T}
  NLPModels.increment!(nlp, :neval_hprod)
  NLPModels.hprod!(nlp.model, x, y, v, Hv; obj_weight = obj_weight)
  return Hv
end
