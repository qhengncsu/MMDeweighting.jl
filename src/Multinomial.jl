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

function compute_probs!(P::Matrix, ηs::Matrix, row_sums::Vector)
    n, q = size(ηs)
    @inbounds for i in 1:n, j in 1:q
        P[i,j] = exp(ηs[i,j])
    end
    fill!(row_sums, zero(eltype(P)))
    @inbounds for i in 1:n, j in 1:q
        row_sums[i] += P[i,j]
    end
    
    @inbounds for i in 1:n
        inv_sum = inv(row_sums[i])
        for j in 1:q
            P[i,j] *= inv_sum
        end
    end
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

function FastMultinomial(X::Matrix, Y::Matrix; tol::Real=1e-6, verbose=true)
    ((n, p), q, max_iters) = (size(X), size(Y,2), 1000)
    ηs, r, P = similar(Y), similar(Y), similar(Y)
    row_sums_buffer = zeros(n)
    XTX, XTY = X'X , X'Y
    L = cholesky!(Symmetric(XTX))
    B = zeros(p,q)
    direction, grad = similar(B), similar(B)
    mul!(ηs, X, B) # ηs = X * β
    obj, old_obj = 0.0, Inf
    (γ, δ) = (copy(B), copy(B))
    nesterov = 0 
    E = 2 .*(Diagonal(ones(q)) + ones(q)*Transpose(ones(q)))
    for iteration = 1:max_iters
        nesterov  += 1
        @. B = γ + ((nesterov  - 1)/(nesterov + 2)) * (γ - δ)
        @. δ = γ # Nesterov acceleration
        mul!(ηs, X, B) # ηs = X * β
        compute_probs!(P,ηs,row_sums_buffer)
        obj = -loglikelihood(Y, P)
        if verbose
            println(iteration," ",obj," ",nesterov)
        end
        if old_obj < obj 
            nesterov = 0 
        end
        @. r = Y - P
        mul!(grad, transpose(X), r)
        ldiv!(XTY, L, grad)
        mul!(direction,XTY, E)
        @. B = B + direction
        if abs(obj-old_obj) < tol*(abs(old_obj)+1)
            return (B, obj, iteration)
        else
            @. γ = B
            old_obj = obj
        end
    end
    return (B, obj, max_iters)
end