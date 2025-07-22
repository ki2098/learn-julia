using IncompleteLU
using ArgParse
using Plots

include("solver.jl")

cavity_size = 1.0
lid_u = 1.0
n = 128
gc = 2
dx = cavity_size/n
sz = (n + 2*gc, n + 2*gc)
Re = 10000.0

x_coords = [dx*(i - gc - 0.5) - 0.5*cavity_size for i in 1:sz[1]]
y_coords = [dx*(j - gc - 0.5) - 0.5*cavity_size for j in 1:sz[2]]

u = zeros(sz)
v = zeros(sz)
uold = zeros(sz)
vold = zeros(sz)
div_U = zeros(sz)
p = zeros(sz)

A, b, max_A_diag = init_linear_eq(dx, sz, gc)

T = 100
dt = 1e-3
cfl = dt*lid_u/dx
max_step::Int = T/dt

println("cell count = ($(sz[1]) $(sz[1]))")
println("guide cell = $gc")
println("dx = $dx")
println("dt = $dt")
println("initial cfl = $cfl")
println("Re = $Re")
println("total steps = $max_step")
println("max A diag = $max_A_diag")
println()

s = ArgParseSettings()
@add_arg_table s begin
    "--dry"
        help = "print params but not don't run time integral"
        action = :store_true
end

parsed_args = parse_args(ARGS, s)

if parsed_args["dry"]
    exit()
end

apply_U_bc!(u, v, lid_u, sz, gc)
for step = 1:max_step
    mag_div = field_time_integral!(
        uold, vold, u, v, div_U, lid_u,
        p, b,
        Re, dx, dt,
        sz, gc,
        max_A_diag = max_A_diag
    )
    print("\rstep = $step, mag div = $mag_div")
end
println()

