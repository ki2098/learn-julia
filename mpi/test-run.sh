#!/usr/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM --mpi proc=4
#PJM -L elapse=00:30:00
#PJM -j
#PJM -o test-run.log

module load nvidia nvompi julia

julia cuda-hpc-setup.jl 12.2

mpiexec -n $PJM_MPI_PROCS -map-by ppr:$PJM_PROC_BY_NODE:node julia --project foo.jl