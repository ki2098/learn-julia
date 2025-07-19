using SparseArrays
using LinearAlgebra

function build_pressure_A(dx, sz, gc)
    cell_count = sz[1]*sz[2]
    A = spdiagm(0 => ones(cell_count))
    map_id = LinearIndices((sz[1], sz[2]))
    
    # inner regin, (pe + pw + pn + ps - 4*pc)/(dx^2) = b
    for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
        Ac = - 4/(dx^2)
        Ae = Aw = An = As = 1/(dx^2)
        
        idc = map_id[i, j]
        ide = map_id[i + 1, j]
        idw = map_id[i - 1, j]
        idn = map_id[i, j + 1]
        ids = map_id[i, j - 1]
        A[idc, idc] = Ac
        A[idc, ide] = Ae
        A[idc, idw] = Aw
        A[idc, idn] = An
        A[idc, ids] = As
    end

    # upper outer layer, pc - ps = 0
    for i in gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        idc = map_id[i, j]
        ids = map_id[i, j - 1]
        A[idc, idc] = 1
        A[idc, ids] = - 1
    end

    # lower outer layer, pc - pn = 0
    for i in gc + 1:sz[1] - gc
        j = gc
        idc = map_id[i, j]
        idn = map_id[i, j + 1]
        A[idc, idc] = 1
        A[idc, idn] = - 1
    end

    # right outer layer, pc - pw = 0
    for j in gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        idc = map_id[i, j]
        idw = map_id[i - 1, j]
        A[idc, idc] = 1
        A[idc, idw] = - 1
    end

    # left outer layer, pc - pe = 0
    for j in gc + 1:sz[2] - gc
        i = gc
        idc = map_id[i, j]
        ide = map_id[i + 1, j]
        A[idc, idc] = 1
        A[idc, ide] = - 1
    end

    return A
end

