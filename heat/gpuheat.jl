using Plots
using SparseArrays
using LinearAlgebra
using CUDA

L = 1
Division = 1000
N = Division + 1
dx = L/Division

nthreads=128
nblocks=cld(N, nthreads)

A = zeros(N, 3)
b = zeros(N)

Tl = 0
Tr = 100

for i = 2:N - 1
    Ae = Aw = 1/(dx^2)
    Ac = - (Ae + Aw)
    A[i, 1] = Ac
    A[i, 2] = Ae
    A[i, 3] = Aw
end
A[1, 1] = 1
A[N, 1] = 1
b[1] = Tl
b[N] = Tr

max_diag = maximum(abs.(A[:, 1]))
A ./= max_diag
b ./= max_diag

Ad = CuArray(A)
bd = CuArray(b)
Td = CUDA.zeros(N)
rd = CUDA.zeros(N)

function get_point_res(A, x, b, i)
    Ac = A[i, 1]
    Ae = A[i, 2]
    Aw = A[i, 3]

    xc = x[i]
    xe = xw = 0.0

    if Ae != 0.0
        xe = x[i + 1]
    end

    if Aw != 0.0
        xw = x[i - 1]
    end

    r = b[i] - (Ac*xc + Ae*xe + Aw*xw)

    return r
end

function gpu_colored_sor_sweep!(A, x, b, ω, color)
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = gridDim().x*blockDim().x
    for i = index:stride:length(x)
        if i%2 == color
            r = get_point_res(A, x, b, i)
            x[i] += ω*r/A[i,1]
        end
    end
end

function gpu_res!(A, x, b, r)
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = gridDim().x*blockDim().x
    for i = index:stride:length(x)
        r[i] = get_point_res(A, x, b, i)
    end
end

function gpu_solve_sor!(A, x, b, r, ω)
    it = 0
    while true
        
        @cuda threads=nthreads blocks=nblocks gpu_colored_sor_sweep!(A, x, b, ω, 0)

        @cuda threads=nthreads blocks=nblocks gpu_colored_sor_sweep!(A, x, b, ω, 1)

        @cuda threads=nthreads blocks=nblocks gpu_res!(A, x, b, r)
        
        err = norm(r)/sqrt(length(r))

        it += 1
        print("\r$it, $err")
        if err <= 1e-6
            break
        end
    end
    println()
end

gpu_solve_sor!(Ad, Td, bd, rd, 1.5)

T = Array(Td)
x = [dx*(i - 1) for i=1:N]
p = plot(x, T)
savefig(p, "T.png")