module CuCavity

include("bc.jl")
include("cfd.jl")
include("eq.jl")

struct Solve
    u
    v
    ut
    vt
    uu
    vv
    div_U
    u_lid
    p
    A
    b
    r
    max_diag
    x
    y
    dx
    dy
    Re
    dt
    max_step
    sz
    gc
end

nthread_2d = (16, 16)

function init(L, division, u_lid, Re, T, dt)
    gc = 2
    sz = (division + 2*gc, division + 2*gc)
    dx = dy = L/division
    u = CUDA.zeros(sz)
    v = CUDA.zeros(sz)
    ut = CUDA.zeros(sz)
    vt = CUDA.zeros(sz)
    uu = CUDA.zeros(sz)
    vv = CUDA.zeros(sz)
    div_U = CUDA.zeros(sz)
    p = CUDA.zeros(sz)
    A, b, r, max_diag = gpu_init_pressure_eq(dx, dy, sz, gc, nthread_2d)
    x = [(i - gc - 0.5)*dx for i=1:sz[1]]
    y = [(j - gc - 0.5)*dy for j=1:sz[2]]
    max_step::Int = T/dt
    return Solve(
        u, v, ut, vt, uu, vv, div_U, u_lid,
        p, A, b, r, max_diag,
        x, y, dx, dy,
        Re, dt, max_step,
        sz, gc
    )
end

function time_integral!(solve::Solve)
    solve.ut .= solve.u
    solve.vt .= solve.v

    gpu_predict_U!(
        solve.ut, solve.vt, solve.u, solve.v, solve.uu, solve.vv,
        solve.dx, solve.dy, solve.dt, 1/solve.Re,
        solve.sz, solve.gc, nthread_2d
    )

    gpu_pressure_eq_b!(
        solve.uu, solve.vv, solve.b,
        solve.dx, solve.dy, solve.dt, solve.max_diag,
        solve.sz, solve.gc, nthread_2d
    )

    gpu_sor!(
        solve.A, solve.p, solve.b, solve.r,
        1.5, solve.sz, solve.gc, 1e-6, 1000, nthread_2d
    )

    gpu_pbc!(
        solve.p, solve.sz, solve.gc
    )

    gpu_update_U!(
        solve.u, solve.v, solve.uu, solve.vv, solve.p,
        solve.dx, solve.dy, solve.dt,
        solve.sz, solve.gc, nthread_2d
    )

    gpu_Ubc!(
        solve.u, solve.v, solve.u_lid, solve.sz, solve.gc
    )

    gpu_UUbc!(
        solve.uu, solve.vv, solve.sz, solve.gc
    )

    div_err = gpu_div_U!(
        solve.uu, solve.vv, solve.div_U,
        solve.dx, solve.dy,
        solve.sz, solve.gc, nthread_2d
    )

    return div_err
end

end
