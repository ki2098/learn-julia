using CUDA
using Test
using Base.Threads
using BenchmarkTools

N = 2^20

xd = CUDA.fill(1.0, N)
yd = CUDA.fill(2.0, N)

function gpu_add2!(y, x)
    index = (blockIdx().x - 1)*blockDim().x + threadIdx().x
    stride = gridDim().x*blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

threads=256
blocks=cld(N, threads)

@btime begin
CUDA.@sync begin
    @cuda threads=threads blocks=blocks gpu_add2!(yd, xd)
end
end