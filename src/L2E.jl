function L2ELoss(r::Vector, τ::Real)
    n = length(r)
    l = zero(eltype(r))
    for i in eachindex(r)
        l +=  - τ/n *sqrt(2/pi) *exp(-τ^2*r[i]^2/2)
    end
    l += τ/(2*sqrt(pi))
    l
end

function update_τ(τ::Real,r::Vector,w::Vector;tol::Real=1e-6, max_steps=10, verbose = true, max_iters = 100)
    obj = L2ELoss(r, τ)
    new_obj = Inf
    n = length(r)
    η = log(τ)
    iter = 0
    grad = Inf
    while iter < max_iters
        iter += 1
        weights!(w, r, τ)
        grad = τ/(2*sqrt(pi)) - τ/n * sqrt(2/pi) * sum(w) + τ^3/n * sqrt(2/pi) * sum(w .* (r.^2))
        if abs(grad)<tol
            break
        end
        halving_steps = 0
        for step = 1:max_steps 
            η = η - grad
            new_obj = L2ELoss(r, exp(η))
            if new_obj < obj || step == max_steps
                halving_steps = step - 1
                break
            else
                η = η + grad
                grad /= 2
            end
        end
        obj = new_obj
        τ = exp(η)
        if verbose
            println(iter," ",obj," ",halving_steps)
        end
    end
    return τ,obj
end
    

function weights!(w::Vector, r::Vector, τ::Real)
    for i in eachindex(r)
        w[i] = exp(-τ^2*r[i]^2/2)
    end
end

function FastL2E(X::Matrix, y::Vector, β0::Vector, τ0::Real; tol::Real=1e-6, MM=true, verbose=true)
    ((n, p), T) = (size(X), eltype(X))
    max_iters = 1000
    r, w, ytilde, eta = similar(y), similar(y), similar(y), similar(y)
    XTX, XTy = X'X, X'y
    WX = similar(X)
    L = cholesky!(Symmetric(XTX))
    β = copy(β0)
    τ = τ0
    obj = 0
    rel_change = Inf
    outer_iter = 0
    while rel_change > tol
        outer_iter += 1
        obj, old_obj = 0.0, Inf
        (γ, δ) = (copy(β), copy(β))
        nesterov = 0
        for iter = 1:max_iters
            nesterov += 1
            @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
            @. δ = γ # Nesterov acceleration
            mul!(eta,X,β)
            @. r = y - eta
            obj = L2ELoss(r, τ)
            if verbose
                println(iter,"  ",obj,"  ",nesterov)
            end
            weights!(w, r, τ)
            if MM
                @. ytilde = y*w + eta*(1-w)
                mul!(XTy, transpose(X), ytilde)
                ldiv!(β, L, XTy)
            else
                mul!(WX,Diagonal(w),X)
                mul!(XTX, Transpose(X),WX)
                mul!(XTy, Transpose(WX), y)
                β .= XTX\XTy
            end
            if old_obj < obj 
                nesterov = 0 
            end # restart Nesterov
            if abs(old_obj - obj) < tol * (abs(old_obj)+1) && iter > 1
                break
            else
                @. γ = β
                old_obj = obj
            end
        end
        τ, new_obj = update_τ(τ,r,w,verbose=verbose)
        rel_change = abs(new_obj-obj)/(abs(obj)+1)
        obj = new_obj
    end
    return β, τ, obj, outer_iter
end