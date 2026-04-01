# MMDeweighting

## Installation
Open Julia REPL and proceed to the project directory
```Julia
using Pkg
Pkg.activate(pwd())
Pkg.instantiate()
```

## Quick Start for Quantile Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, Random
Random.seed!(123)
# generate data
p = 100
n = 10p
covM = [0.7^abs(i-j) for i in 1:p, j in 1:p]
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d, n))
β_true = [1; 0.1ones(p)]
d = TDist(1.5)
truth =  X * β_true[2:end] .+ β_true[1]
y = truth + rand(d, n) .- Statistics.quantile(d,0.5)
X = [ones(n) X]
μ = max(0.05, ((log(n) + p) / n)^0.4)
#trigger compilation
@time (beta, obj, iter, μ) = FastQR(X, y, 0.5; μ_max=μ, μ_min=μ, decay_iters=2, tol=1e-6, verbose=false)
#actual excution time
@time (beta, obj, iter, μ) = FastQR(X, y, 0.5; μ_max=μ, μ_min=μ, decay_iters=2, tol=1e-6, verbose=false)
# check estimation error
println("Estimation error: ", norm(beta - β_true))
# compare with least squares
beta_ls = X \ y
println("LS estimation error: ", norm(beta_ls - β_true))
```

## Quick Start for Sparse Quantile Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, Random
Random.seed!(123)
# generate data
n = 500
p = 250
q = 0.7
covM = [0.7^abs(i-j) for i in 1:p, j in 1:p]
d = MvNormal(zeros(p), covM)
X = [ones(n) rand(d, n)']
β_true = [[4, 0, 1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.4, 0, -1.6, 0, -1.8]; zeros(p - 20)]
truth = X * β_true
# heavy tailed error
d = TDist(1.5)
y = truth .+ (0.5 .* X[:, end] .+ 1) .* (rand(d, n) .- Statistics.quantile(d, q))
μ = max(0.05, sqrt(0.25) * (log(p) / n)^0.25)
λs = exp10.(range(1, -4, length=30))
ks = [1:1:50;]
# trigger compilation
@time best_λ, best_β, mean_errors = CV_SQR(X,y,q,λs; μ_max=μ, μ_min=μ, nfold=5)
# actual computation time: sparse quantile regression using hard thresholding
@time best_λ, best_β_ht, mean_errors = CV_SQR(X,y,q,λs; μ_max=μ, μ_min=μ, nfold=5)
# actual computation time: sparse quantile regression using proximal distance principle 
@time best_k, best_β_pd, mean_errors = CV_SQR(X,y,q,ks; μ=μ, nfold=5)
# check sparsity and estimation error
println("True sparsity: ", sum(β_true .!= 0))
println("Estimated sparsity (HT): ", sum(best_β_ht .!= 0))
println("Estimation error (HT): ", norm(best_β_ht - β_true))
println("Estimated sparsity (PD): ", sum(best_β_pd .!= 0))
println("Estimation error (PD): ", norm(best_β_pd - β_true))
```

## Quick Start for L2E Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, Random
Random.seed!(123)
# generate data
p = 100
n = 10p
covM = [0.7^abs(i-j) for i in 1:p, j in 1:p]
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d, n))
β_true = [1; 0.1ones(p)]
truth = X * β_true[2:end] .+ β_true[1]
d = Normal(0, 1)
y = truth + rand(d, n)
y[1:Int(0.1n)] .+= 10
X[Int(0.1n + 1):Int(0.2n), 2] .+= 10
X = [ones(n) X]
β0 = zeros(p + 1)
τ0 = 1 / mean(abs.(y .- mean(y)))
# trigger compilation
@time beta2, τ2, obj2, iter2 = FastL2E(X, y, β0, τ0; MM=true, tol=1e-6, verbose=false)
# actual computation time: L2E regression using weighted least squares
@time beta1, τ1, obj1, iter1 = FastL2E(X, y, β0, τ0; MM=false, tol=1e-6, verbose=false)
# actual computation time: L2E regression using double majorization
@time beta2, τ2, obj2, iter2 = FastL2E(X, y, β0, τ0; MM=true, tol=1e-6, verbose=false)
# check estimation error
println("Estimation error (WLS): ", norm(beta1 - β_true))
println("Estimation error (MM): ", norm(beta2 - β_true))
# compare with least squares (affected by outliers)
beta_ls = X \ y
println("LS estimation error: ", norm(beta_ls - β_true))
```

## Quick Start for L2E Isotonic Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, DelimitedFiles, Random
#data = readdlm("data/december.csv", ',', Any, '\n') 
data = readdlm("data/december.csv", ',', Float64)
y = data[:, 2]
β0 = y
τ0 = 1.0
# trigger compilation
@time β1, τ1, obj1, w1 = L2EIsotonic(y, β0, τ0; MM=true, verbose=false)
# actual computation time: L2E isotonic regression using weighted least squares
@time β1, τ1, obj1, w1 = L2EIsotonic(y, β0, τ0; MM=false, verbose=false)
# actual computation time: L2E isotonic regression using double majorization
@time β2, τ2, obj2, w2 = L2EIsotonic(y, β0, τ0; MM=true, verbose=false)
```

## Quick Start for Multinomial Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, Random
Random.seed!(123)
# generate data
p = 30
n = 1000p
q = 10
covM = [0.7^abs(i-j) for i in 1:p, j in 1:p]
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d, n))
B_true = 0.2rand(p, q-1)
ηs = X * B_true .+ 1
row_sums_buffer = zeros(n)
P = zeros(n,q)
compute_probs_reduced!(P, ηs, row_sums_buffer)
Y = zeros(n, q)
X = [ones(n) X]
for i in 1:n
    probs = P[i, :]
    d = Multinomial(1, probs)
    Y[i, :] .= rand(d)
end
# trigger compilation
@elapsed B, obj = FastMultinomial(X, Y; verbose=false)
# actual computation time
@elapsed B, obj = FastMultinomial(X, Y; verbose=false)
# check estimation error
B_true_full = [zeros(1, q-1); B_true]
println("Estimation error: ", norm(B - B_true_full))
```

## Quick Start for Low-rank Multinomial Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, Random
Random.seed!(123)
# generate data
p = 100
n = 1000
q = 10
covM = [0.7^abs(i-j) for i in 1:p, j in 1:p]
d = MvNormal(zeros(p), covM)
X_train = Transpose(rand(d, n))
X_train = [ones(n) X_train]
B_true = 3rand(p + 1, q-1)
ηs = X_train * B_true
row_sums_buffer = zeros(n)
P = zeros(n,q)
compute_probs_reduced!(P, ηs, row_sums_buffer)
Y = zeros(n,q)
for i in 1:n
    probs = P[i, :]
    Y[i,:] .= rand(Multinomial(1, probs))
end
# generate validation set
X_val = Transpose(rand(d, n))
X_val = [ones(n) X_val]
ηs_val = X_val * B_true
P_val = zeros(n,q)
compute_probs_reduced!(P_val, ηs_val, row_sums_buffer)
Y_val = zeros(n,q)
for i in 1:n
    Y_val[i,:] .= rand(Multinomial(1, P_val[i, :]))
end
λs = exp10.(range(0, -4, length=50))
#trigger compilation
@time Bs, _ = LowrankMultinomial(X_train, Y, λs; μ_max=1.0, μ_min=0.01, verbose=false);
#actual computation time for solution path
@time Bs, _ = LowrankMultinomial(X_train, Y, λs; μ_max=1.0, μ_min=0.01, verbose=false);
# compare validation log-likelihood across λs
ll_val = zeros(50)
for i in 1:50
    compute_probs_reduced!(P_val, X_val * Bs[:,:,i], row_sums_buffer)
    ll_val[i] = sum(Y_val .* log.(P_val .+ 1e-12))
end
best_idx = argmax(ll_val)
println("Best λ: ", λs[best_idx])
println("Estimation error at best λ: ", norm(Bs[:,:,best_idx] - B_true))
# compare with regular multinomial regression
B_full, _ = FastMultinomial(X_train, Y; verbose=false)
compute_probs_reduced!(P_val, X_val * B_full, row_sums_buffer)
ll_full = sum(Y_val .* log.(P_val .+ 1e-12))
println("Validation log-likelihood (Low-rank): ", ll_val[best_idx])
println("Validation log-likelihood (Full): ", ll_full)
```
