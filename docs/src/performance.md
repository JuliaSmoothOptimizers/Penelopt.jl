# Performance

## Preallocation

When calling our solver, `Penelopt.jl` starts with an *allocation* phase where all of the vectors, matrices, and sub-solvers needed to run the algorithm are built.
The workspace is allocated in a [struct](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) called `L2PenaltySolver`.
Our solver then proceeds to solve the provided `nlp`, in an allocation-free function named `solve!`.

In particular, this means that once a solver has been constructed, it can be reused across several calls to `solve!` without allocating any new memory, provided the problem dimensions ($n$ variables, $m$ constraints) stay the same.
This is particularly useful:

* when solving the same problem repeatedly from different starting points (e.g., in a multistart strategy);
* when solving a sequence of closely related problems (e.g., inside a loop, for bilevel or parametric optimization);
* when you want to avoid the garbage collector getting in the way of a tight benchmarking loop.

The convenience call
```{julia}
julia> stats = L2Penalty(nlp)
```
internally does exactly the following:
```{julia}
julia> solver = L2PenaltySolver(nlp)
julia> stats = PeneloptExecutionStats(nlp)
julia> solve!(solver, nlp, stats)
```
Separating these steps is the key to avoiding allocations: `L2PenaltySolver(nlp)` performs *all* the allocations up front (workspaces, factorization buffers, sub-solver states, etc.), and every subsequent call to `solve!` reuses that memory in place.

```{julia}
julia> using Penelopt

# Allocate the solver and the stats object once.
julia> solver = L2PenaltySolver(nlp)
julia> stats = PeneloptExecutionStats(nlp)

# Solve as many times as needed;
julia> solve!(solver, nlp, stats)
julia> solve!(solver, nlp, stats; x = x1)
julia> solve!(solver, nlp, stats; x = x2)
```

!!! warning "Resetting internal state"
    Some internal quantities of the solver (such as quasi-Newton
    approximations, the watchdog checkpoint, and factorization counters)
    persist between calls to `solve!`. Therefore, we strongly recommend calling [`SolverCore.reset!`](https://github.com/JuliaSmoothOptimizers/SolverCore.jl)
    between calls to `solve!`.
    ```{julia}
    julia> solve!(solver, nlp, stats)
    julia> reset!(solver)
    julia> solve!(solver, nlp, stats; x = x1)
    julia> ...
    ```

!!! warning "Keyword arguments"
    Some [options](options.md) require allocations and are therefore passed to the solver structure construction call.
    The following options **need** to be passed to the solver constructor (passing them to `solve!` will cause failure).
    * `r2n_m_monotone::Int = 12`;
    * `linear_solver::Sring = "mumps"`.
    For clarity, if you want to modify these options, while preallocating the workspace, you should do
    ```{julia}
    julia> solver = L2PenaltySolver(nlp, r2n_m_monotone = 6, linear_solver = "ldlt")
    julia> ...
    ```

### The `L2PenaltySolver` structure

The fields for the `L2PenaltySolver` are:

| Field | Type | Description |
|:------|:-----|:-------------|
| `x` | `V` | current outer iterate $x_k$ |
| `xn` | `V` | (*internal*) trial/next iterate buffer |
| `y` | `V` | current Lagrange multiplier estimate $y_k$ |
| `cn` | `V` | (*internal*) buffer for the constraint values at a trial point |
| `dual_res` | `V` | buffer for the dual residual $\nabla f(x_k) + J(x_k)^Ty_k$ |
| `s` | `V` | (*internal*) buffer for the computed step |
| `s0` | `V` | (*internal*) buffer used when computing least-squares multipliers |
| `∇fk` | `V` | buffer for $\nabla f(x_k)$ |
| `temp_b` | `V` | (*internal*) scratch buffer of length $m$, used e.g. by the least-squares multiplier computation |
| `subsolver` | `PenaltyR2NSolver` | the R2N (inner-loop) solver structure used to (approximately) minimize the current penalized subproblem $f(x) + \tau_k\lVert c(x) \rVert_2$ |
| `subpb` | `L2PenalizedProblem` | the penalized subproblem $f(x) + \tau_k\lVert c(x)\rVert_2$ associated with `nlp` |
| `substats` | `GenericExecutionStats` | the statistics object updated by `subsolver` at every R2N iteration |

We mark some fields as *internal* which means that we discourage users to read/modify these in [callbacks](callbacks.md) as they are only used for internal computations.

Here, `T` is the scalar type and `V` is the vector type of the problem (by
default `T == Float64` and `V == Vector{Float64}`; see the [note on
parametric types](options.md) on the options page).

### Nested solver structures

Because the algorithm is organized in nested loops (see the [options
terminology](options.md#Terminology)), `solver.subsolver` and
`solver.subsolver.subsolver` expose the state of the inner loops.

* `solver.subsolver::PenaltyR2NSolver` holds the state needed to minimize a
  *single* penalized subproblem $f(x) + \tau\lVert c(x)\rVert_2$ for a fixed
  $\tau$: the current inner iterate `xk`, its multiplier estimate `y`, the
  non-monotone objective history `m_fh_hist`, the watchdog `checkpoint`,
  and its own `subsolver::MoreSorensenSolver` and `subpb::ShiftedL2PenalizedProblem`.
* `solver.subsolver.subsolver::MoreSorensenSolver` holds the buffers needed
  to compute a single step by (approximately) solving the trust-region
  subproblem via the Moré–Sorensen method: the right-hand side/solution
  buffers `u1`, `u2`, `x1`, `x2`, the assembled KKT-like matrix `H`, and the
  `workspace` used by the chosen [linear solver](options.md#Linear-Solver)
  (e.g., `PenaltyLDLTWorkspace`, `PenaltyMUMPSWorkspace`, ...).

All three levels can be constructed once (`L2PenaltySolver`,
`PenaltyR2NSolver`, or `MoreSorensenSolver`, respectively) and reused across
solves, following the same two-call pattern described above.

## Fixed Variables

If some variables in your problem are fixed (i.e., some of your constraints are $x_i = c_i$ for some constants $c_i$), Penelopt.jl automatically removes these internally and solves a reduced problem.

```@example fixed
using CUTEst, Penelopt

# NLP model with fixed variables
nlp = CUTEstModel("AIRCRFTA")

# You can check that a problem has fixed variables by accessing the `meta` of the NLP model
length(nlp.meta.ifix) > 0
```
The reformulation is done automatically when calling the solver, and the solution is then mapped back to the original problem.
However, if you wish to use a [preallocated solver](performance.md#preallocation) you are responsible for constructing the reduced problem yourself.

The example below shows how to construct a preallocated solver for a problem with fixed variables.
```@example fixed

# For the reformulation, it suffices to call the following function.
nlp_no_fixed = remove_fixed_variables(nlp)

# Now you can construct a preallocated solver for the reduced problem.
solver = L2PenaltySolver(nlp_no_fixed)
stats = PeneloptExecutionStats(nlp_no_fixed)

solve!(solver, nlp_no_fixed, stats)

finalize(nlp) # hide

# To retrieve the solution back you can do
solution_full = recover_full_solution(nlp_no_fixed, stats.solution)
```

## Shifted Constraints

If some of your equality constraints are written as $c_i(x) = v_i$ for some constant $v_i \neq 0$, Penelopt.jl automatically reformulates them as $c_i(x) - v_i = 0$ internally.

```@example shifted
using ADNLPModels, Penelopt

# NLP model with a shifted constraint c(x) = 5
nlp = ADNLPModel(x -> sum(x .^ 2), ones(5), x -> [sum(x .^ 3)], [5.0], [5.0])

# You can check whether a problem has shifted constraints from its `meta`
any(!iszero, nlp.meta.lcon) || any(!iszero, nlp.meta.ucon)
```
The reformulation is done automatically when calling the solver. If you wish to use a [preallocated solver](performance.md#preallocation) you are responsible for constructing the reformulated problem yourself.

```@example shifted
nlp_no_shift = remove_constraint_shift(nlp)

solver = L2PenaltySolver(nlp_no_shift)
stats = PeneloptExecutionStats(nlp_no_shift)

solve!(solver, nlp_no_shift, stats)
```

If your problem has both fixed variables and shifted constraints, compose the two functions; the order does not matter.
```julia
nlp_preprocessed = nlp |> remove_fixed_variables |> remove_constraint_shift
```

## Quasi-Newton Approximations

If the Hessian of the Lagrangian of your nonlinear programming problem is dense, ill-conditionned, expensive to compute, or inaccessible, you may be interested in replacing it with a quasi-Newton approximation.

`Penelopt.jl` offers you the possibility to run the optimization process with a [Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) approximation.
You can simply pass a keyword argument when calling the solver (see the [options](options.md) page):

```@example bfgs
using CUTEst, Penelopt

# Construct your NLP model
nlp = CUTEstModel("HS6")

stats = L2Penalty(nlp; print_level = 1, qn_hessian_approximation = "bfgs")

finalize(nlp) # hide
```

!!! warning "Default options"
    Some default [options](options.md) have a different value when using our solver with a quasi-Newton approximation.
    Those options are
    * `r2n_η2::T = 0.1` which default to `0.9` with a quasi-Newton approximation.

!!! tip "L-BFGS options"
    You can pass the following additional keyword arguments to customize your approximation.
    Those are
    * `qn_mem::Int = 6`: memory parameter for the limited-memory approximation.
    * `qn_scaling::Bool = true`: whether we scale $B_0 = \gamma I$ with $\gamma = y^Ty / s^T y$, where $s$ is the step and $y$ is the difference of the two last gradients of the Lagrangian.
    * `qn_max_skip::Int = 2` (*advanced*): if we skipped a pair $(s, y)$ more than `max_skip` times in a row, we reset the approximation.

If you wish to use a quasi-Newton approximation together with the [preallocation](performance.md#preallocation) feature you should do the following instead:

```{julia}
julia> using CUTEst, Penelopt

# Construct your NLP model
julia> nlp = CUTEstModel("HS6")

# Construct a "quasi-Newton" NLP structure first
julia> nlp_bfgs = CompactBFGSModel(nlp; mem = 6, scaling = true, max_skip = 2)

# The solver will automatically use the quasi-Newton approximation now
julia> solver = L2PenaltySolver(nlp_bfgs)
julia> stats = PeneloptExecutionStats(nlp_bfgs)

# Solve as many times as needed;
julia> solve!(solver, nlp_bfgs, stats)
```

!!! danger "Quasi-Newton approximations and preprocessing"
    Apply `remove_fixed_variables` and `remove_constraint_shift` before constructing a quasi-Newton approximation, not after.
    The quasi-Newton approximation is built for whatever problem you pass it; it will not be valid for the original, unpreprocessed problem.