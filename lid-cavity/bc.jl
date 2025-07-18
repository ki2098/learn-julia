function apply_U_bc(U, u_lid, sz, gc)
    Ulid = [u_lid, 0]

    # upper outer layer, Uc = 2*Ulid - Us
    for i in gc + 1:sz[1] - gc
        j = sz[2] - gc + 1
        U[i, j] = 2*Ulid - U[i, j - 1]
    end

    # lower outer layer, Uc = - Un
    for i in gc + 1:sz[1] - gc
        j = gc
        U[i, j] = - U[i, j + 1]
    end

    # right outer layer, Uc = - Uw
    for j in gc + 1:sz[2] - gc
        i = sz[1] - gc + 1
        U[i, j] = - U[i - 1, j]
    end

    # left outer layer, Uc = - Ue
    for j in gc + 1:sz[2] - gc
        i = gc
        U[i, j] = - U[i + 1, j]
    end
end