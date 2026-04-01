function loglikelihood(ys::Vector, ηs::Vector)
    obj = zero(eltype(ηs))
    for i in eachindex(ys)
        obj += ys[i]*ηs[i] - log1p(exp(ηs[i]))
    end
    obj
end

function FastLogistic(X::Matrix, y::Vector; tol::Real=1e-6, verbose=true)
    ((n, p), max_iters) = (size(X), 1000)
    ηs, r = similar(y), similar(y)
    XTX, XTy = X'X, X'y
    L = cholesky!(Symmetric(XTX))
    β = zeros(p)
    direction, grad = similar(β), similar(β) # work vectors
    mul!(ηs, X, β) # ηs = X * β
    obj, old_obj = -loglikelihood(y, ηs), Inf
    (γ, δ) = (copy(β), copy(β))
    nesterov = 0 
    for iteration = 1:max_iters
        nesterov  += 1
        @. β = γ + ((nesterov  - 1)/(nesterov + 2)) * (γ - δ)
        @. δ = γ # Nesterov acceleration
        mul!(ηs, X, β) # ηs = X * β
        obj = -loglikelihood(y, ηs)
        if verbose
            println(iteration," ",obj," ",nesterov)
        end
        if old_obj < obj 
            nesterov = 0 
        end
        for i = 1:n
            r[i] = y[i] - 1/(1+exp(-ηs[i]))
        end
        mul!(grad, transpose(X), r)
        ldiv!(direction, L, grad)
        @. β = β + 4* direction
        if abs(old_obj - obj) < tol*(abs(old_obj)+1)
            return (β, obj, iteration)
        else
            @. γ = β
            old_obj = obj
        end 
    end
    return (β, obj, max_iters)
end

function loglikelihood(Y::Matrix, P::Matrix)
    T = eltype(P)
    obj = zero(T)
    @inbounds for i in axes(Y, 1)
        for j in axes(Y, 2)
            if Y[i,j] == one(T)
                obj += log(P[i,j])
            end
        end
    end
    obj
end

function compute_probs_reduced!(P::Matrix{T}, ηs::Matrix{T}, row_sums::Vector{T}) where T <: AbstractFloat
    n, q = size(ηs)
    # q+1 classes: first q have linear predictors, last one has η=0
    @inbounds for i in 1:n
        # find row max (including the reference category η=0)
        max_η = zero(T)
        for j in 1:q
            max_η = max(max_η, ηs[i,j])
        end
        # compute exp(η - max_η) for each class
        row_sums[i] = exp(-max_η)  # reference category: exp(0 - max_η)
        for j in 1:q
            P[i,j] = exp(ηs[i,j] - max_η)
            row_sums[i] += P[i,j]
        end
        # normalize
        inv_sum = inv(row_sums[i])
        for j in 1:q
            P[i,j] *= inv_sum
        end
        P[i, q+1] = exp(-max_η) * inv_sum
    end
end

function FastMultinomial(X::Matrix, Y::Matrix;
    tol            = 1e-6,
    max_iters      = 1000,
    strategy       = "nesterov",
    doubling_start = 20,
    verbose        = true,
)
    strategy in ("original", "nesterov", "step-doubling") ||
        throw(ArgumentError("strategy must be \"original\", \"nesterov\", or \"step-doubling\", got \"$strategy\""))
    ((n, p), c) = (size(X), size(Y, 2))
    q = c - 1
    ηs              = zeros(n, q)
    r               = zeros(n, q)
    P               = similar(Y)
    row_sums_buffer = zeros(n)
    XTX = X'X
    XTY = X'Y[:, 1:q]
    L   = cholesky!(Symmetric(XTX))
    B         = zeros(p, q)
    direction = similar(B)
    grad      = similar(B)
    E       = 2 .* (Diagonal(ones(q)) + ones(q) * ones(q)')
    γ, δ    = copy(B), copy(B)
    nesterov = 0
    obj, old_obj = 0.0, Inf
    for iter in 1:max_iters
        if strategy === "nesterov"
            nesterov += 1
            @. B = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
            @. δ = γ
        else
            @. B = γ
            @. δ = γ
        end
        mul!(ηs, X, B)
        compute_probs_reduced!(P, ηs, row_sums_buffer)
        obj = -loglikelihood(Y, P)
        if verbose
            if strategy === "original"
                phase = "original"
            elseif strategy === "nesterov"
                phase = @sprintf("nesterov k=%d", nesterov)
            else
                phase = iter >= doubling_start ? "step-doubling" : "plain"
            end
            @printf("%4d  obj=%.6g  %s\n", iter, obj, phase)
        end
        if old_obj < obj && strategy === "nesterov"
            nesterov = 0
        end
        if abs(obj - old_obj) < tol * (abs(old_obj) + 1)
            verbose && @printf("Converged at iter %d\n", iter)
            return (B, obj, iter)
        end
        @. r = Y[:, 1:q] - P[:, 1:q]
        mul!(grad, transpose(X), r)
        ldiv!(XTY, L, grad)
        mul!(direction, XTY, E)
        step = (strategy === "step-doubling" && iter >= doubling_start) ? 2 : 1
        @. B = B + step * direction
        @. γ = B
        old_obj = obj
    end
    verbose && @printf("Max iters (%d) reached\n", max_iters)
    return (B, obj, max_iters)
end