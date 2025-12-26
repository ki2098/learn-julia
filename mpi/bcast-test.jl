import MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
root = 0

N = 5

if rank == root
    A = [i for i=1:N]
else
    A = nothing
end

print("rank = $rank, A = $A\n")

MPI.Barrier(comm)

A = MPI.bcast(A, root, comm)

print("rank = $rank, A = $A\n")

MPI.Finalize()