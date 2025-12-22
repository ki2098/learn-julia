using Distributions
using Plots

μ = 1
σ = 0.1
n = 10000
check_points = [100, 1000, 5000, 10000]

d = Normal(μ, σ)
x = rand(d, n)

println(d)

μ = 0
M = 0
for i=1:n
    δ1 = x[i] - μ
    global μ += δ1/i
    δ2 = x[i] - μ
    global M += δ1*δ2

    if i in check_points
        local σ = sqrt(M/(i-1))
        println("i=$i μ=$μ σ=$σ")
    end
end

histogram(x; bins=100)
savefig("data/normal-histogram-$n.png")
