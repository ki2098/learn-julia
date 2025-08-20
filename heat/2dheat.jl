using Plots
using SparseArrays
using LinearAlgebra
import LinearSolve as LS

L = 1
N = 200
gc = 1
sz = (N + 2*gc, N + 2*gc)
dx = L/N

x_coords = [dx*(i - gc - 0.5) for i in gc+1:sz[1]-gc]
y_coords = [dx*(j - gc - 0.5) for j in gc+1:sz[2]-gc]

cell_count = sz[1]*sz[2]

A = spdiagm(0 => ones(cell_count))
b = zeros(sz)

map_id = LinearIndices(sz)

for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
    Ac = - 4/(dx^2)
    Ae = Aw = An = As = 1/(dx^2)
        
    idc = map_id[i, j]
    ide = map_id[i + 1, j]
    idw = map_id[i - 1, j]
    idn = map_id[i, j + 1]
    ids = map_id[i, j - 1]
    A[idc, idc] = Ac
    A[idc, ide] = Ae
    A[idc, idw] = Aw
    A[idc, idn] = An
    A[idc, ids] = As
end

T_top = 100.0
# top boundary, Tc = 2*T_top - Ts
for i in gc + 1:sz[1] - gc
    j = sz[2] - gc + 1
    idc = map_id[i, j]
    ids = map_id[i, j - 1]
    A[idc, idc] = 1.0
    A[idc, ids] = 1.0
    b[i, j] = 2*T_top
end

T_right = 0.0
# right boundary, Tc = 2*T_right - Tw
for j in gc + 1:sz[2] - gc
    i = sz[1] - gc + 1
    idc = map_id[i, j]
    idw = map_id[i - 1, j]
    A[idc, idc] = 1.0
    A[idc, idw] = 1.0
    b[i, j] = 2*T_right
end

T_left = 0
# left boundary, Tc = 2*T_left - Te
for j in gc + 1:sz[2] - gc
    i = gc
    idc = map_id[i, j]
    ide = map_id[i + 1, j]
    A[idc, idc] = 1.0
    A[idc, ide] = 1.0
    b[i, j] = 2*T_left
end

# bottom boundary, Tc = Tn
for i in gc + 1:sz[1] - gc
    j = gc
    idc = map_id[i, j]
    idn = map_id[i, j + 1]
    A[idc, idc] = 1.0
    A[idc, idn] = - 1.0
end

println("A, b ready")
flush(stdout)

Pl = Diagonal(A)
prob = LS.LinearProblem(A, vec(b))
@time sol = LS.solve(prob, LS.KrylovJL_BICGSTAB(), verbose=true, abstol=1e-6*cell_count, Pl=Pl)
T = reshape(sol.u, sz)

heatmap(x_coords, y_coords, transpose(T[gc+1:end-gc,gc+1:end-gc]))