using LinearAlgebra
using CUDA

function utopia_convection(fww, fw, fc, fe, fee, u, dx)
    return (u*(- fee + 8*fe - 8*fw + fww) + abs(u)*(fee - 4*fe + 6*fc - 4*fw + fww))/(12*dx)
end

function cell_convection(u, v, f, dx, i, j)
    fc  = f[i, j]
    fe  = f[i + 1, j]
    fee = f[i + 2, j]
    fw  = f[i - 1, j]
    fww = f[i - 2, j]
    fn  = f[i, j + 1]
    fnn = f[i, j + 2]
    fs  = f[i, j - 1]
    fss = f[i, j - 2]
    return utopia_convection(fww, fw, fc, fe, fee, u, dx) + utopia_convection(fss, fs, fc, fn, fnn, v, dx)
end

function cell_diffusion(f, μ, dx, i, j)
    fc = f[i, j]
    fe = f[i + 1, j]
    fw = f[i - 1, j]
    fn = f[i, j + 1]
    fs = f[i, j - 1]
    return μ*(fe + fw + fn + fs - 4*fc)/(dx^2)
end

function cell_pseudo_U(u, v, μ, dx, dt, i, j)
    uc = u[i, j]
    vc = v[i, j]
    pseudo_u = uc + dt*(- cell_convection(uc, vc, u, dx, i, j) + cell_diffusion(u, μ, dx, i, j))
    pseudo_v = vc + dt*(- cell_convection(uc, vc, v, dx, i, j) + cell_diffusion(v, μ, dx, i, j))
    return pseudo_u, pseudo_v
end

function pseudo_U!(uold, vold, u, v, μ, dx, dt, sz, gc)
    i = threadIdx().x + (blockIdx().x - 1)*blockDim().x
    j = threadIdx().y + (blockIdx().y - 1)*blockDim().y
    if gc < i <= sz[1] - gc && gc < j <= sz[2] - gc
        ua, va = cell_pseudo_U(uold, vold, μ, dx, dt, i, j)
        u[i, j] = ua
        v[i, j] = va
    end
end

function cell_div_U(u, v, dx, i, j)
    ue = u[i + 1, j]
    uw = u[i - 1, j]
    vn = v[i, j + 1]
    vs = v[i, j - 1]
    return (ue - uw + vn - vs)/(2*dx)
end

function pressure_eq_rhs!(u, v, b, dx, dt, max_diag, sz, gc)
    i = threadIdx().x + (blockIdx().x - 1)*blockDim().x
    j = threadIdx().y + (blockIdx().y - 1)*blockDim().y
    if gc < i <= sz[1] - gc && gc < j <= sz[2] - gc
        b[i, j] = cell_div_U(u, v, dx, i, j)/(dt*max_diag)
    end
end

function cell_grad_p(p, dx, i, j)
    pe = p[i + 1, j]
    pw = p[i - 1, j]
    pn = p[i, j + 1]
    ps = p[i, j - 1]
    return (pe - pw)/(2*dx), (pn - ps)/(2*dx)
end

function update_U_by_gradp!(u, v, p, dx, dt, sz, gc)
    i = threadIdx().x + (blockIdx().x - 1)*blockDim().x
    j = threadIdx().y + (blockIdx().y - 1)*blockDim().y
    if gc < i <= sz[1] - gc && gc < j <= sz[2] - gc
        
    end
end

