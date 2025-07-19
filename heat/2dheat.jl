using Plots
using SparseArrays
using LinearAlgebra
using IterativeSolvers
using IncompleteLU

L = 1
N = 50
gc = 1
sz = (N + 2*gc, N + 2*gc)
dx = L/N

x_coords = [dx*(i - gc - 0.5) for i in 1:sz[1]]
y_coords = [dx*(j - gc - 0.5) for j in 1:sz[2]]

cell_count = sz[1]*sz[2]

A = spdiagm(0 => ones(cell_count))
b = zeros(cell_count)

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

T_top = 100
# top boundary, Tc = 2*T_top - Ts
for i in gc + 1:sz[1] - gc
    j = sz[2] - gc + 1
    idc = map_id[i, j]
    ids = map_id[i, j - 1]
    A[idc, idc] = 1
    A[idc, ids] = 1
    b[idc] = 2*T_top
end

T_right = 20
# right boundary, Tc = 2*T_right - Tw
for j in gc + 1:sz[2] - gc
    i = sz[1] - gc + 1
    idc = map_id[i, j]
    idw = map_id[i - 1, j]
    A[idc, idc] = 1
    A[idc, idw] = 1
    b[idc] = 2*T_right
end

T_left = 20
# left boundary, Tc = 2*T_left - Te
for j in gc + 1:sz[2] - gc
    i = gc
    idc = map_id[i, j]
    ide = map_id[i + 1, j]
    A[idc, idc] = 1
    A[idc, ide] = 1
    b[idc] = 2*T_left
end

# bottom boundary, Tc = Tn
for i in gc + 1:sz[1] - gc
    j = gc
    idc = map_id[i, j]
    idn = map_id[i, j + 1]
    A[idc, idc] = 1
    A[idc, idn] = - 1
end

Pl_ilu = ilu(A)

T, ch = bicgstabl(A, b, Pl=Pl_ilu, log=true, verbose=true, abstol=1e-9, reltol=0)
# T = A\b

heatmap(x_coords, y_coords, transpose(T))