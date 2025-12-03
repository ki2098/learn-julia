using MPI
using CUDA

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

devices = CUDA.devices()
n_gpu = length(devices)
gpu_id = rank%n_gpu

CUDA.device!(gpu_id)

gpu_id = CUDA.device(CUDA.current_context())

print("$rank/$size -> $gpu_id\n")

MPI.Barrier(comm)

MPI.Finalize()