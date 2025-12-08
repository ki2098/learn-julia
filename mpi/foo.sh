#!/usr/bin/bash
julia foo.jl > out.${OMPI_COMM_WORLD_RANK}.txt
