#!/bin/sh

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=8
#PJM -L elapse=00:30:00
#PJM -j
#PJM -o test-run.log

module load nvidia nvompi julia

julia cuda-hpc-setup.jl 12.2

julia --project -e 'using MPIPreferences; MPIPreferences.use_jll_binary("OpenMPI_jll"); using Pkg; Pkg.instantiate()'

mpiexec -n 8 julia --project foo.jl