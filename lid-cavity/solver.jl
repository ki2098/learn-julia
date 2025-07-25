using IncompleteLU
using Statistics
import LinearSolve

include("cfd.jl")
include("eq.jl")
include("bc.jl")

function field_time_integral!(
    uold, vold, u, v, div_U, lid_u,
    p, b,
    Re, dx, dt,
    sz, gc;
    max_A_diag=1.0
)
    uold .= u
    vold .= v
    
    field_pseudo_U!(uold, vold, u, v, 1.0/Re, dx, dt, sz, gc)
    field_pressure_eq_rhs!(u, v, b, dx, dt, max_A_diag, sz, gc)
    linsolve.b = vec(b)
    solution = LinearSolve.solve!(linsolve)
    p .= reshape(solution.u, size(b))
    field_update_U_by_grad_p!(u, v, p, dx, dt, sz, gc)
    apply_U_bc!(u, v, lid_u, sz, gc)
    avg_mag_div_U = field_div_U!(u, v, div_U, dx, sz, gc)

    return avg_mag_div_U
end