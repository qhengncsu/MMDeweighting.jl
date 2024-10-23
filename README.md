# MMDeweighting

## Installation
Open Julia REPL and proceed to the project directory
```Julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start for Quantile Regression
```julia
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
# trigger compilation: sparse quantile regression using proximal distance principle 
@time best_k, best_β1, mean_errors = CV_SQR(X,y,τ,[1:1:50;],h=max(0.05,sqrt(0.25)*(log(p)/n)^0.25), nfold=5)
# actual computation time: sparse quantile regression using proximal distance principle 
@time best_k, best_β1, mean_errors = CV_SQR(X,y,τ,[1:1:50;],h=max(0.05,sqrt(0.25)*(log(p)/n)^0.25), nfold=5)
# trigger compilation: sparse quantile regression using hard thresholding
@time best_λ, best_β2, mean_errors = CV_SQR(X,y,τ,exp10.(range(1, -4, length=30)),h=max(0.05,sqrt(0.25)*(log(p)/n)^0.25), nfold=5)
# actual computation time: sparse quantile regression using hard thresholding
@time best_λ, best_β2, mean_errors = CV_SQR(X,y,τ,exp10.(range(1, -4, length=30)),h=max(0.05,sqrt(0.25)*(log(p)/n)^0.25), nfold=5)
```