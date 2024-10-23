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
using MMDeweighting, Distributions, LinearAlgebra, Statistics
# generate data
p = 100
n = 10*p
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d,n))
β = 0.1 .* ones(p)
d = TDist(1.5)
truth =  X*β.+ 1
y = truth + rand(d, n) .- Statistics.quantile(d,0.5)
X = [ones(n,1) X]
h = max(0.05,((log(n)+p)/n)^0.4)
#trigger compilation
@time (beta, obj, iter, h) = FastQR(X, y, 0.5;tol=1e-6, h=h, verbose=false)
#actual excution time
@time (beta, obj, iter, h) = FastQR(X, y, 0.5;tol=1e-6, h=h, verbose=false)
```

## Quick Start for Sparse Quantile Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics
# generate data
n = 250
p = 500
τ = 0.7
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = rand(d, n)
X = [ones(n) X']
β = [[4, 0, 1.8, 0, 1.6, 0, 1.4, 0, 1.2, 0, 1, 0, -1, 0, -1.2, 0, -1.4, 0, -1.6, 0, -1.8]; zeros(p-20)]
truth =  X*β
# heavy tailed error
d = TDist(1.5)
y = truth + (0.5.*X[:,end].+1) .*(rand(d, n) .- Statistics.quantile(d,τ))
h = max(0.05,sqrt(0.25)*(log(p)/n)^0.25)
λs = exp10.(range(1, -4, length=30))
ks = [1:1:50;]
# trigger compilation
@time best_λ, best_β, mean_errors = CV_SQR(X,y,τ,λs,h=h, nfold=5)
# actual computation time: sparse quantile regression using proximal distance principle 
@time best_k, best_β_pd, mean_errors = CV_SQR(X,y,τ,ks,h=h, nfold=5)
# actual computation time: sparse quantile regression using hard thresholding
@time best_λ, best_β_ht, mean_errors = CV_SQR(X,y,τ,λs,h=h, nfold=5)
```

## Quick Start for L2E regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics
p = 100
n = 10*p
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d,n))
β = 0.1 .* ones(p)
truth =  X*β .+ 1
d = Normal(0,1)
y = truth + rand(d, n) 
y[1:Int(0.1*n)] .+= 10
X[Int(0.1*n+1):Int(0.2*n),2] .+= 10
X = [ones(n,1) X]
β = [1;β]
β0 = zeros(p+1)
τ0 = 1/mean(abs.(y .- mean(y)))
# trigger compilation
@time beta2, τ2, obj2 = FastL2E(X,y,β0,τ0; MM=true,tol=1e-6,verbose=false)
# actual computation time: L2E regression using weighted least squares
@time beta1, τ1, obj1 = FastL2E(X,y,β0,τ0; MM=false,tol=1e-6,verbose=false)
# actual computation time: L2E regression using double majorization
@time beta2, τ2, obj2 = FastL2E(X,y,β0,τ0; MM=true,tol=1e-6,verbose=false)
```

## Quick Start for L2E Isotonic regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics, DelimitedFiles
data = readdlm("data/december.csv", ',', Any, '\n') 
y = data[1:end,2]
y = convert(Vector{Float64}, y)
β0 = y
τ0 = 1.0
# trigger compilation
@time β1, τ1, obj1, w1 = L2EIsotonic(y,β0,τ0;MM=true,verbose=false)
# actual computation time: L2E isotonic regression using weighted least squares
@time β1, τ1, obj1, w1 = L2EIsotonic(y,β0,τ0;MM=false,verbose=false)
# actual computation time: L2E isotonic regression using double majorization
@time β2, τ2, obj2, w2 = L2EIsotonic(y,β0,τ0;MM=true,verbose=false)
```

## Quick Start for Logistic Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics
p = 30
n = 1000*p
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d,n))
β= 0.1 .* ones(p)
d = Normal(0,1)
truth =  X*β .+ 1
y = zeros(n)
for i = 1:n
    u = 1/(1+exp(-truth[i]))
    if rand() < u
        y[i] = 1.0 
    end
end
X = [ones(n,1) X]
# trigger compilation
@time beta, obj = FastLogistic(X,y;verbose=false)
# actual computation time
@time beta, obj = FastLogistic(X,y;verbose=false)
```

## Quick Start for Multinomial Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics
p = 30
n = 1000*p
q = 10
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d,n))
B = 0.2*rand(p,q)
ηs = zeros(n,q)
mul!(ηs,X,B)
ηs .+= 1
row_sums_buffer = zeros(n)
P = zeros(n,q)
compute_probs!(P,ηs, row_sums_buffer)
Y = zeros(n,q)
X = [ones(n) X]
for i in 1:n
    probs = P[i, :]
    d = Multinomial(1, probs)
    Y[i,:] .= rand(d)
end
# trigger compilation
@elapsed B, obj = FastMultinomial(X,Y;verbose=false)
# actual computation time
@elapsed B, obj = FastMultinomial(X,Y;verbose=false)
```

## Quick Start for Low-rank Multinomial Regression
```julia
using MMDeweighting, Distributions, LinearAlgebra, Statistics
p = 100
n = 1000
q = 10
covM = ones(p,p)
for i in 1:p
    for j in 1:p
        covM[i,j] = 0.7^abs(i-j)
    end
end
d = MvNormal(zeros(p), covM)
X = Transpose(rand(d,n))
X = [ones(n) X]
B = 3*rand(p+1,q)
B_proj = similar(B)
proj_rank!(B_proj, B, 3)
ηs = X*B_proj
row_sums_buffer = zeros(n)
P = zeros(n,q)
compute_probs!(P,ηs, row_sums_buffer)
Y = zeros(n,q)
for i in 1:n
    probs = P[i, :]
    d = Multinomial(1, probs)
    Y[i,:] .= rand(d)
end
λs = exp10.(range(0, -4, length=50))
#trigger compilation
@time Bs_snn1 = LowrankMultinomial(X,Y,λs;μ=1.0,verbose=false);
#actual computation time for solution path (μ=1.0)
@time Bs_snn1 = LowrankMultinomial(X,Y,λs;μ=1.0,verbose=false);
#actual computation time for solution path (μ=0.01)
@time Bs_snn2 = LowrankMultinomial(X,Y,λs;μ=0.01,verbose=false);
```