using LinearAlgebra

function compute_utopia_convection(ww, w, c, e, ee, u, dx)
    return (u*(- ee + 8*e - 8*w + ww) + abs(u)*(ee - 4*e + 6*c - 4*w + ww))/(12*dx)
end

function compute_cell_convection(U, f, dx, i, j)
    uc  = U[i, j, 1]
    vc  = U[i, j, 2]
    fc  = f[i, j]
    fe  = f[i + 1, j]
    fee = f[i + 2, j]
    fw  = f[i - 1, j]
    fww = f[i - 2, j]
    fn  = f[i, j + 1]
    fnn = f[i, j + 2]
    fs  = f[i, j - 1]
    fss = f[i, j - 2]
    return 
        compute_utopia_convection(fww, fw, fc, fe, fee, uc, dx) 
    +   compute_utopia_convection(fss, fs, fc, fn, fnn, vc, dx)
end

function compute_cell_diffusion(viscosity, f, dx, i, j)
    fc = f[i, j]
    fe = f[i + 1, j]
    fw = f[i - 1, j]
    fn = f[i, j + 1]
    fs = f[i, j - 1]
    return viscosity*(fe + fw + fn + fs - 4*fc)/(dx^2)
end

function compute_cell_pseudo_U(U, viscosity, dx, dt, i, j)
    convection = compute_cell_convection(U, U, dx, i, j)
    diffusion = compute_cell_diffusion(viscosity, U, dx, i, j)
    return U[i, j] + dt*(- convection + diffusion)
end

function compute_pseudo_U(U, Ut, viscosity, dx, dt, sz, gc)
    for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
        Ut[i, j] = compute_cell_pseudo_U(U, viscosity, dx, dt, i, j)
    end
end

function compute_cell_U_div(U, dx, i, j)
    ue = U[i + 1, j, 1]
    uw = U[i - 1, j, 1]
    vn = U[i, j + 1, 2]
    vs = U[i, j - 1, 2]
    return (ue - uw + vn - vs)/(2*dx)
end

function compute_pressure_rhs(U, b, dx, dt, scale, sz, gc)
    for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
        b[i, j] = compute_cell_U_div(U, dx, i, j)
    end
    b./(dt*scale)
end

function project_cell_pressure_grad(U, p, dx, dt, i, j)
    dpdx = (p[i + 1, j] - p[i - 1, j])/dx
    dpdy = (p[i, j + 1] - p[i, j - 1])/dx
    U[i, j, 1] -= dt*dpdx
    U[i, j, 2] -= dt*dpdy
end

function project_pressure_grad(U, p, dx, dt, sz, gc)
    for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
        project_cell_pressure_grad(U, p, dx, dt, i, j)
    end
end

function compute_U_div(U, divU, dx, sz, gc)
    for i in gc + 1:sz[1] - gc, j in gc + 1:sz[2] - gc
        divU[i, j] = compute_cell_U_div(U, dx, i, j)
    end
    mag_divU = norm(divU)
    inner_cell_count = (sz[1] - 2*gc)*(sz[2] - 2*gc)
    return mag_divU/sqrt(inner_cell_count)
end