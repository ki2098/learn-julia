include("solve.jl")

using Printf
using .CuCavity

cd(@__DIR__)

L = 1
Division = 256
uLid = 1
Re = 10000
T = 100
dt = 1e-3

sol = CuCavity.init(L, Division, uLid, Re, T, dt)

outputInterval = Int(1/sol.dt)

for step = 1:sol.max_step
    lsIt, lsErr, divErr = CuCavity.time_integral!(sol)
    # print("\rstep=$step, |div U|=$divErr, LS=($lsIt, $lsErr)")
    @printf(
        "\rstep=%7i, |div U|=%.5e, LS iter=%4i, LS err=%.5e", 
        step, divErr, lsIt, lsErr
    )
    flush(stdout)
    if step % outputInterval == 0
        num::Int = step/outputInterval
        CuCavity.write_csv("data/result.csv.$num", sol)
    end
end
println()
