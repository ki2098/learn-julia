using SparseArrays
using LinearAlgebra
using IterativeSolvers

const L = 1.0
const Division = 10
const CellCount = Division + 1
const Dx = L/Division

A = spdiagm(0 => ones(CellCount))

for i in 2:CellCount - 1
    Ac = -2/(Dx^2)
    Ae = Aw = 1/(Dx^2)
    A[i, i] = Ac
    A[i, i + 1] = Ae
    A[i, i - 1] = Aw
end

A[1, 1] = 1
A[CellCount, CellCount] = 1

b = zeros(CellCount)
b[1] = 20
b[CellCount] = 100

x = zeros(CellCount)
# x[1] = 0
# x[CellCount] = 100

t, ch = bicgstabl(A, b, max_mv_products=100, reltol=1e-6, log=true, verbose=true)