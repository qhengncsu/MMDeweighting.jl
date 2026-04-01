function L2ELoss(r::Vector{T}, τ::T) where T <: AbstractFloat
    n = length(r)
    l = zero(T)
    @inbounds @simd for i in eachindex(r)
        l += -τ/n * sqrt(T(2/pi)) * exp(-τ^2 * r[i]^2 / 2)
    end
    l += τ / (2 * sqrt(T(pi)))
    return l
end

function update_τ(τ::Real, r::Vector, w::Vector;
    tol       = 1e-6,
    max_steps = 10,
    max_iters = 100,
    verbose   = true,
)
    T   = eltype(r)
    obj = L2ELoss(r, T(τ))
    n   = length(r)
    η   = log(τ)
    iter = 0
    grad = typemax(T)
    while iter < max_iters
        iter += 1
        weights!(w, r, T(τ))
        grad = τ/(2*sqrt(pi)) - τ/n * sqrt(2/pi) * sum(w) +
               τ^3/n * sqrt(2/pi) * sum(w .* (r.^2))
        if abs(grad) < tol
            verbose && @printf("  update_τ converged at iter %d  ‖∇‖=%.4g\n", iter, abs(grad))
            break
        end
        halving_steps = 0
        new_obj = obj
        for step in 1:max_steps
            η_new   = η - grad
            new_obj = L2ELoss(r, T(exp(η_new)))
            if new_obj < obj || step == max_steps
                η             = η_new
                halving_steps = step - 1
                break
            else
                grad /= 2
            end
        end
        obj = new_obj
        τ   = exp(η)
        verbose && @printf("  update_τ iter %d  obj=%.6g  τ=%.4g  halving=%d\n",
                           iter, obj, τ, halving_steps)
    end
    return T(τ), obj
end

function weights!(w::Vector{T}, r::Vector{T}, τ::T) where T <: AbstractFloat
    @inbounds @simd for i in eachindex(r)
        w[i] = exp(-τ^2 * r[i]^2 / 2)
    end
end

function FastL2E(X::Matrix, y::Vector, β0::Vector, τ0::Real;
    tol     = 1e-6,
    MM      = true,
    verbose = true,
)
    ((n, p), T) = (size(X), eltype(X))
    max_iters = 1000
    r, w, ytilde, eta = similar(y), similar(y), similar(y), similar(y)
    XTX, XTy = X'X, X'y
    WX = similar(X)
    L  = cholesky!(Symmetric(XTX))
    β  = copy(β0)
    τ  = T(τ0)
    obj        = zero(T)
    rel_change = typemax(T)
    outer_iter = 0
    while rel_change > tol
        outer_iter += 1
        obj, old_obj = zero(T), typemax(T)
        γ, δ     = copy(β), copy(β)
        nesterov = 0
        for iter in 1:max_iters
            nesterov += 1
            @. β = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
            @. δ = γ
            mul!(eta, X, β)
            @. r = y - eta
            obj = L2ELoss(r, τ)
            if verbose
                @printf("%4d  obj=%.6g  τ=%.4g  nesterov k=%d\n", iter, obj, τ, nesterov)
            end
            if MM
                weights!(w, r, τ)
                @. ytilde = y*w + eta*(1 - w)
                mul!(XTy, transpose(X), ytilde)
                ldiv!(β, L, XTy)
            else
                weights!(w, r, τ)
                mul!(WX, Diagonal(w), X)
                mul!(XTX, transpose(X), WX)
                mul!(XTy, transpose(WX), y)
                β .= XTX \ XTy
            end
            if old_obj < obj
                nesterov = 0
            end
            if abs(old_obj - obj) < tol * (abs(old_obj) + 1) && iter > 1
                verbose && @printf("Converged at iter %d\n", iter)
                break
            end
            @. γ = β
            old_obj = obj
        end
        τ, new_obj = update_τ(τ, r, w; tol=tol, verbose=verbose)
        rel_change = abs(new_obj - obj) / (abs(obj) + 1)
        obj = new_obj
        verbose && @printf("Outer iter %d  obj=%.6g  τ=%.4g  rel_change=%.4g\n",
                           outer_iter, obj, τ, rel_change)
    end
    verbose && @printf("Converged at outer iter %d\n", outer_iter)
    return β, τ, obj, outer_iter
end