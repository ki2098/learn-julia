#!/usr/bin/bash

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -o test-run.log

module load nvidia nvompi julia

julia cuda-hpc-setup.jl 12.2

mpiexec -n 8 julia --project foo.jl