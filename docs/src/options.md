# Options

Options are passed to the solver as [keyword arguments](https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments).
For clarity, if you want to pass an `option` to the solver with some `value`, simply write
```{julia}
# Interface your problem with the NLPModels.jl API
julia> nlp = ...

# Install
julia> using Pkg
julia> Pkg.add("Penelopt")
julia> using Penelopt

# Solve with a specified option
julia> L2Penalty(nlp, option = value)
```
Each option has a type and a default value.
Therefore, following the Julia syntax, we present each parameter as 
```
  parameter::type = default_value
```

!!! note "Parametric Types"
    Some option types are [parametric](https://docs.julialang.org/en/v1/manual/types/#Parametric-Types).
    The [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API that we use to represent the nonlinear programming problem allow users to choose for a [floating point format](https://en.wikipedia.org/wiki/Computer_number_format) when making the representation.
    In Penelopt.jl, the (parametric) scalar type of the problem is `T` and the (parametric) vector type is `V`.
    By default, you can consider that `T == Float64` and `V == Vector{Float64}`.

    To acount for the parametric nature of the problem, default values are often a function of `eps(T)` which represents the [machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) of `T`. Again, by default you can consider that
    ```
      eps(T) = eps(Float64) ≈ 1e-16
    ```

Options that should be modified by expert users only are marked as *advanced*.

## Terminology

!!! note "Solver Structure"
    Our solver is organized with multiple nested loops.
    Each of these loops have their own specific options.
    This section tries to give a concise presentation of each of these.
    Just **be aware** that, unless explicitly stated otherwise, options interact with the outermost loop.
    On the other hand, most *advanced* options are the options for the lower level loops.
    
    We try to clearly disembiguate on which level each option acts. 
    Therefore, if you want to use some *advanced* options, you are strongly advised to read this section.

Our algorithm solves
```math
    \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ f(x) \quad \textup{subject to} \ c(x) = 0.
```
We replace this problem with a **sequence** of unconstrained *penalized* problems
```math
    \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ f(x) + \tau_k \| c(x) \|_2.
```
This sequence is the outermost loop, we call it the *penalty loop* or *outer loop*.
To solve this unconstrained problem, we call an **iterative subsolver** called *R2N*.
The process of solving one penalized problem is called the *R2N loop* or *inner loop*.

Basically, the *R2N* solver starts with an initial iterate $x_0 \in \mathbb{R}^n$.
Then, it computes a *step* $s_0$ and updates $x_1 := x_0 + s_0$.
It then proceeds with $x_2 := x_1 + s_1$, etc. until it converges to a solution of the *penalized* problem.

To compute a *step*, we use an **iterative subsolver** based on the method of Moré and Sorensen.
The process of computing a step for *R2N* is called the *Moré-Sorensen loop* or *MS loop*.

Finally, the Moré-Sorensen solver requires solving multiple *linear systems*.
Therefore, it calls a *linear solver*, such as MUMPS.

Schematically, the structure looks like this.
```
┌──────────────────┐
│ Penelopt     │
└──────────────────┘
          │
          ▼
Penalty loop
          │
          ▼
R2N loop                     (minimizes one penalized objective)
          │
          ▼
Moré–Sorensen loop           (computes a step)
          │
          ▼
Linear solver                (solves a linear system)
```

!!! info "Options Terminology"
    * An option that acts on the R2N loop is prefixed with `r2n`. For example, `r2n_max_iter` represents the maximum number of iterations for the R2N loop.
    * An option that acts on the MS loop is prefixed with `ms`.
    * An option that acts on the linear solver is prefixed with `ls`.

# Options Reference

## Termination

We declare an iterate $(x_k, y_k)$ optimal when it satisfies
```math
\lVert \nabla f(x_k) + J(x_k)^T y_k \rVert_{\infty} \leq \epsilon_D
\quad \text{and} \quad
\lVert c(x_k) \rVert_{\infty} \leq \epsilon_P.
```
We define $\epsilon_P$ and $\epsilon_D$ as 
```math
\begin{align*}
  \epsilon_P &= \max(\epsilon_P^a, \epsilon^a) + \max(\epsilon_P^r, \epsilon^r) \lVert c(x_0) \rVert_{\infty}, \\
  \epsilon_D &= \max(\epsilon_D^a, \epsilon^a) + \max(\epsilon_D^r, \epsilon^r) \lVert \nabla f(x_0) + J(x_0)^T y_0 \rVert_{\infty}.
\end{align*}
```
* `atol::T = √eps(T)`: $\epsilon^a$, desired convergence tolerance (absolute).
  > Absolute tolerance for primal and dual residuals. When `T == Float64`, the default value is $\approx 10^{-8}$.

* `rtol::T = √eps(T)`: $\epsilon^r$, desired convergence tolerance (relative).
  > Relative tolerance for primal and dual residuals. When `T == Float64`, the default value is $\approx 10^{-8}$.

* `dual_inf_atol::T = 0`: $\epsilon^a_D$, desired convergence tolerance (absolute) for dual infeasibility.
  
* `dual_inf_rtol::T = 0`: $\epsilon^r_D$, desired convergence tolerance (relative) for dual infeasibility.

* `primal_inf_atol::T = 0`: $\epsilon^a_P$, desired convergence tolerance (absolute) for primal infeasibility.

* `primal_inf_rtol::T = 0`: $\epsilon^r_P$, desired convergence tolerance (relative) for primal infeasibility.

* `μ::T = 0.01` (*advanced*): Decrease factor for penalized problem accuracy.
  > If $\epsilon_D^k$ is the accuracy level for the current penalized problem, then when the algorithm increases the accuracy for the subproblem, it performs the update $\epsilon_D^{k+1} = \max(μ\epsilon_D^{k},\epsilon_D)$. Smaller values mean that subproblems are solved to higher accuracy more aggressively. This is $\mu_{\epsilon}$ in the implementation paper.

* `infeasible_tol::T = 0.001` (*advanced*): local infeasibility tolerance.
  > When an optimality measure of $\min_x \|c(x)\|_2$ arround the current iterate is detected to be smaller (relative to this tolerance) than the current primal residual, the problem is declared infeasible. Larger values can cause false positives while smaller values cause false negatives. This is $\epsilon_I$ in the implementation paper.

* `infeasible_iter::Int = 2` (*advanced*): local infeasibility detection frequency.
  > Frequency (in *outer-loop* iterations) at which we estimate locally infeasibility. For performance, if you are positive that your model is feasible, you can set this parameter to a very large value. 

* `max_eval::Int = -1`: Maximum number of objective function evaluation.
  > The solver stops when the number of objective function evaluations exceeds `max_eval`. A negative value means unlimited.

* `max_time::Float64 = 30.0`: Maximum number of CPU seconds.
  > The solver stops when the number of CPU seconds exceeds `max_time`.

* `max_iter::Int = 100`: Maximum number of iterations.
  > The solver stops when the number of iterations exceeds `max_iter`.

* `r2n_max_iter::Int = 1000` (*advanced*): Maximum number of *r2n-loop* iterations.
  > Each run of the *r2n-loop* stops when the number of iterations exceeds `r2n_max_iter`.

* `ms_max_iter::Int = 10` (*advanced*): Maximum number of *ms-loop* iterations.
  > Each run of the *ms-loop* stops when the number of iterations exceeds `ms_max_iter`.

## Initialization

* `x::V = nlp.meta.x0`: Initial point for the optimizer.

## Logging
We refer to the [outputs](outputs.md#console-output) section for an explanation of the logger output.

* `verbose_level::Int = 0`: Output verbosity level.
    >   The larger this value the more detailed is the output. The valid range is `0 ≤ print_level ≤ 4`. **Warning**: large values can potentially print a lot of information, we recommend to start with a value of `1`. The value of `print_level` corresponds to the depth of the loop that will be printed. That is 
    >   - `print_level = 1`: Prints the information relative to the *penalty loop*,
    >   - `print_level = 2`: Prints the information relative to the *r2n loop*,
    >   - `print_level = 3`: Prints the information relative to the *ms loop*.

 * `verbose::Int = 1`: Frequency (in iterations) at which information is printed.
    > `verbose = 1` prints every iteration (of the outer loop), `verbose = 10` prints every ten iteration (of the outer loop). If `print_level < 1`, this parameter is ignored.

 * `r2n_verbose::Int = 1`: Frequency (in iterations) at which information is printed.
    > `r2n_verbose = 1` prints every iteration (of the r2n loop), `r2n_verbose = 10` prints every ten iteration (of the r2n loop). If `print_level < 2`, this parameter is ignored.
 
 * `ms_verbose::Int = 1`: Frequency (in iterations) at which information is printed.
   > `ms_verbose = 1` prints every iteration (of the ms loop), `ms_verbose = 10` prints every ten iteration (of the ms loop). If `print_level < 3`, this parameter is ignored.

## Customization

 * `callback::Function = (args...) -> nothing`: callback function.
   > We have a dedicated [section](callbacks.md) for this option.

## Outer Loop Specific

 * `τ0::T = 1.0` (*advanced*): initial penalty parameter.
    > The first penalized problem is solved with a penalty parameter $\max(τ_0, \lVert y^{LS}_0 \rVert_1)$. This is $\tau_0$ in the implementation paper.

 * `τmin::T = 1.0` (*advanced*): penalty parameter minimimum increase.
    > The penalty parameter is increased by at least `τmin`, i.e. $\tau_{k+1} \geq \tau_k + \tau_{\text{min}}$. This is $\tau_{\text{min}}$ in the implementation paper.

## R2N Specific

 * `r2n_η1::T = √√eps(T)` (*advanced*): Armijo sufficient decrease threshold. 
    > Steps are accepted when the ratio of the actual decrease and the first-order decrease is larger than `r2n_η1`. This is $\eta_1$ in the implementation paper. When `T == Float64`, the default value is $\approx 10^{-4}$.

 * `r2n_η2::T = 0.1` (*advanced*): strong Armijo sufficient decrease threshold.
    > In addition to being accepted, when the ratio of the actual decrease and the first-order decrease is larger than `r2n_η2`, the quadratic regularization parameter is decreased by some constant factor (see `r2n_γ`). This is $\eta_2$ in the implementation paper. Note that `r2n_η2 ≥ r2n_η1` should hold. When using quasi-Newton approximations, the default value becomes `r2n_η2::T = 0.9`.

 * `r2n_γ::T = 3` (*advanced*): decrease/increase factor for quadratic regularization parameter.
    > When a step is rejected, the quadratic regularization paramater $\sigma_l$ is increased by a factor `r2n_γ`, when a step is "strongly" accepted (see `r2n_η2`), the quadratic regularization parameter is decreased by a factor `1/r2n_γ`. Note that `r2n_γ > 1` should hold.

 * `r2n_σmin::T = eps(T)^2` (*advanced*): minimum quadratic regularization parameter.
    > The quadratic regularization parameter is bounded from below by `r2n_σmin`. When `T == Float64`, the default value is $\approx 10^{-32}$. This value should be very small.

 * `r2n_m_monotone::Int = 12` (*advanced*): non-monotone memory parameter.
    > When computing the ratio of the actual decrease and the first-order decrease (see `r2n_η1`), the decrease is computed with respect to the maximum of the last `r2n_m_monotone` values of the objective.

 * `r2n_watchdog_max_iter::Int = 10` (*advanced*): maximum number of watchdog iterations.
    > When the watchdog technique is activated, the solver accepts a step regardless of `r2n_η1` and proceeds for at most `r2n_watchdog_max_iter` before retreating to the watchdog activation iterate. When the watchdog is active, if a sufficient decrease (see `r2n_watchdog_η0`) in either the dual infeasibility or the objective is attained, the watchdog is deactivated and the current iterate is retained. See the implementation paper for more details.
  
 * `r2n_watchdog_η0::T = √eps(T)` (*advanced*): watchdog deactivation threshold.
    > The watchdog is deactivated when a sufficient decrease condition in either the dual infeasibility or the objective is attained. See the implementation paper for more details. When `T == Float64`, the default value is $\approx 10^{-8}$.

 * `r2n_tiny_step_tol::T = eps(T)` (*advanced*): tolerance for detecting numerically insignificant steps.
    > When a step $s$ for some iterate $x$ is such that $|s_i|/|x_i|$ is smaller than `r2n_tiny_step_tol` for all $1 \leq i \leq n$ for a fixed number of consective iterations (see `r2n_nmax_tiny_step`), the inner loop returns with a corresponding exit message. When `T == Float64`, the default value is $\approx 10^{-16}$.

 * `r2n_nmax_tiny_step::Int = 2` (*advanced*): number of consecutive tiny steps allowed (see `r2n_tiny_step_tol`).

## MS Specific

 * `ms_accept_descent::Bool = true` (*advanced*): *secular* equation Newton's method truncation parameter.
     > When `ms_accept_descent` is set to `true`, the Newton's method applied to the *secular* equation is truncated when a simple decrease in the quadratic model of the objective has been reached. Setting this parameter to `false` can significantly increase the number of matrix factorizations, therefore we do not recommend it. For details on the *secular* equation and Newton's method, we refer to the implementation paper.

 * `ms_σmax::T = 1/eps(T)^(0.8)`, (*advanced*): maximum value for the regularization parameter.
     > When inertia corrections are performed within the ms loop, we increase the quadratic regularization parameter `σ` by a factor `ms_μσ` (see `ms_μσ`) until either the inertia is $(n, 0, m)$ or `σ` reaches `ms_σmax`. In the latter case, the solver proceeds by setting the Hessian to $0$. When `T == Float64`, the default value is $\approx 10^{16}$.

 * `ms_tol::T = eps(T)^(0.6)`, (*advanced*): tolerance for the MS loop (absolute).
     > Tolerance for Newton's method applied to the *secular* equation. For completeness, the MS loop terminates when $| \|y\|_2 - Δ | \leq \text{ms\_tol}$. When `T == Float64`, the default value is $\approx 10^{-10}$.

 * `ms_μα::T = 0.1`, (*advanced*): dual inertia decrease factor.
     > When the Newton's method applied to the *secular* equation produces a negative value of `α`, the iterate is reduced instead by a factor `ms_μα`. This is $\mu_{\alpha}$ in the implementation paper.

 * `ms_μσ::T = T(10)`, (*advanced*): primal inertia increase factor.
     > When inertia corrections are performed within the ms loop, we increase the quadratic regularization parameter `σ` by a factor `ms_μσ` until either the inertia is $(n, 0, m)$ or `σ` reaches `ms_σmax` (see `ms_σmax`). This is $\mu_{\sigma}$ in the implementation paper.

 * `ms_α0::T = eps(T)`, (*advanced*): initial value for the MS method.
      > We highly recommend to keep this value as the default. Accepted values are `ms_α0 ∈ [0, ms_αmin1)` (see `ms_αmin1`). When `T == Float64`, the default value is $\approx 10^{-16}$.

 * `ms_αmin1::T = eps(T)^(0.8)`, (*advanced*): first minimum value for the MS method.
      > When a rank defficient Jacobian is detected, the MS method is restarted with `α = ms_αmin1`. Accepted values are `ms_αmin1 ∈ (ms_α0, ms_αmin2)` (see `ms_α0` and `ms_αmin2`). This is $\alpha_{\text{trial}}^1$ in the implementation paper. When `T == Float64`, the default value is $\approx 10^{-13}$.

 * `ms_αmin2::T = eps(T)^(0.6)`, (*advanced*): second minimum value for the MS method.
      > When a rank defficient Jacobian is detected, and the linear solver failed with `α = ms_αmin2`, the MS method is restarted with `α = ms_αmin2`. This is $\alpha_{\text{trial}}^2$ in the implementation paper. When `T == Float64`, the default value is $\approx 10^{-10}$.

 * `ms_ηC::T = eps(T)`, (*advanced*): Cauchy decrease acceptance constant.
      > When $H + \sigma I$ is not positive definite, the first-order model decrease should be larger than `ms_ηC` times the optimality measure squared. Refer to the implementation paper for more information. When `T == Float64`, the default value is $\approx 10^{-16}$.

## Linear Solver

 * `linear_solver::Sring = "mumps"`: Linear solver library used for step computations.
    > Determines which linear algebra package is to be used for the solution of the linear systems.
    >
    > Possible values:
    > * mumps: use the [MUMPS](https://mumps-solver.org/index.php?page=doc) package (default).
    > * ldlt: use the [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl) package.
    > * ma57: use the HSL routine MA57.
    >
    > **NOTE**: Except for the default, you need to **load** corresponding packages to use each option.
    > * ldlt: Load [LDLFactorizations.jl](https://github.com/JuliaSmoothOptimizers/LDLFactorizations.jl).
    > * ma57: Load [HSL.jl](https://github.com/JuliaSmoothOptimizers/HSL.jl).

## Scaling

Scaling replaces the original NLP
```math
   \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ f(x) \quad \textup{subject to} \ c(x) = 0.
```
with
```math
   \underset{x \in \mathbb{R}^n}{\textup{minimize}} \ d_f f(x) \quad \textup{subject to} \ d_c c(x) = 0.
```
for some $$d_f > 0$$ and $$d_c \in R^m_0$$.

 * `nlp_scaling_method::String = "none"`: method used to rescale the objective and constraints before solving.
    > Determines how the scaling factors $d_f$ (objective) and $d_c$ (constraints) are computed.
    >
    > Possible values:
    > * gradient-based: pick $d_f$ and $d_c$ so that the scaled gradient and Jacobian rows have infinity norm at most `gmax` (default).
    > * none: leave the problem unscaled ($d_f = 1$, $d_c = 1$).
    >
    > Scaling is on by default for quasi-Newton approximations.

 * `gmax::T = T(100)`: target infinity norm for the scaled gradient and Jacobian rows.
    > If `nlp_scaling_method != "gradient-based"`, this parameter is ignored.

## quasi-Newton Approximations

 * `qn_hessian_approximation::String = "exact"`: Quasi-Newton approximation for the Hessian of the Lagrangian.
    > Determines which quasi-Newton approximation is to be used for the Hessian of the Lagrangian.
    >
    > Possible values:
    > * exact: use the exact Hessian of the Lagrangian (default).
    > * bfgs: use a limited-memory BFGS approximation.
    > * null: use a zero approximation (does not work well).

 * `qn_mem::Int = 6`: memory parameter for the limited-memory approximation.
    > When using a limited-memory BFGS approximation, this parameter determines how many pairs of $(s, y)$ are stored to build the approximation. If `qn_hessian_approximation::String != "bfgs"`, this parameter is ignored.

 * `qn_scaling::Bool = true`: scaling parameter for the limited-memory approximation.
    > Whether we scale $B_0 = \gamma I$ with $\gamma = y^Ty / s^T y$, where $s$ is the step and $y$ is the difference of the two last gradients of the Lagrangian. If `qn_hessian_approximation::String != "bfgs"`, this parameter is ignored.

 * `qn_max_skip::Int = 2` (*advanced*): if a pair $(s, y)$ is skipped more than `max_skip` times in a row, we reset the approximation.
    > If `qn_hessian_approximation::String != "bfgs"`, this parameter is ignored.