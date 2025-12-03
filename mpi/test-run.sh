#!/bin/sh

#PJM -L rscgrp=b-batch
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -j
#PJM -o test-run.log

module load julia

# julia --project -e 'using CUDA; CUDA.set_runtime_version!(v"12.2")'

~/.julia/bin/mpiexecjl -n 8 julia --project foo.jl