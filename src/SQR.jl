# hard thresholding
function l0_threshold!(β_proj::Vector, β::Vector, λ::Real; intercept=true)
    copy!(β_proj, β)
    if intercept
        for i in 2:length(β)
            if abs(β[i]) < sqrt(2*λ)
                β_proj[i] = 0
            end
        end
        return sum(β_proj[2:end] .!= 0)
    else
        for i in 1:length(β)
            if abs(β[i]) < sqrt(2*λ)
                β_proj[i] = 0
            end
        end
        return sum(β_proj .!= 0)
    end
end

# projection onto l0 norm ball
function l0_proj!(β_proj::Vector, β::Vector, k::Integer, order_buffer::Vector; intercept=true)
    if k == 0
        copy!(β_proj, β)
        return
    end
    fill!(β_proj, zero(β[1]))
    order_buffer_nj = @view order_buffer[2:end]
    if intercept
        β_proj[1] = β[1]
        β_ni = @view β[2:end]
        β_proj_ni = @view β_proj[2:end]
        sortperm!(order_buffer_nj, β_ni, rev=true, by=abs)
        for i in 1:k
            index = order_buffer_nj[i]
            β_proj_ni[index] = β_ni[index]
        end
    else
        sortperm!(order_buffer, β, rev=true, by=abs)
        for i in 1:k
            index = order_buffer[i]
            β_proj[index] = β[index]
        end
    end
end

# solve (MX'X + ρI)β = RHS via Woodbury if p > n
function lin_solve_MM!(β::Vector, RHS::Vector, M::Real, ρ::Real, e::Vector, U::Matrix,
                       X::Matrix, buffer1::Vector, buffer2::Vector, buffer3::Vector)
    (n, p) = size(X)
    if n >= p
        mul!(buffer1, Transpose(U), RHS)
        @. buffer1 /= (M*e + ρ)
        mul!(β, U, buffer1)
    else
        mul!(buffer2, X, RHS)
        mul!(buffer3, Transpose(U), buffer2)
        @. buffer3 /= (1 + M/ρ*e)
        mul!(buffer2, U, buffer3)
        mul!(β, Transpose(X), buffer2)
        axpby!(1/ρ, RHS, -M/ρ^2, β)
    end
end

# -----------------------------------------------------------------------
# SparseQR — λ (penalty) version
# -----------------------------------------------------------------------
function SparseQR(X::AbstractMatrix, y::Vector, q::Real, λs::Vector;
    μ_max          = 10.0,
    μ_min          = 0.25,
    α              = 0.01,
    decay_iters    = 100,
    strategy       = "nesterov",
    tol_grad       = 1e-3,
    intercept      = true,
    verbose        = true,
)
    strategy in ("original", "nesterov") ||
        throw(ArgumentError("strategy must be \"original\" or \"nesterov\", got \"$strategy\""))
    ((n, p), T) = (size(X), eltype(X))
    max_iters = 1000
    μ_max, μ_min = T(μ_max), T(μ_min)
    α = T(α)
    r, s = zeros(T, n), zeros(T, n)
    G = n >= p ? X'X : X*X'
    XTy = Transpose(X) * y
    e, U = eigen(G)
    β = zeros(p)
    β_mm      = similar(β)
    β_proj    = similar(β)
    β_proj_mm = similar(β)
    buffer_p  = similar(β)
    grad      = similar(β)
    γ, δ = copy(β), copy(β)
    βs = zeros(p, length(λs))

    for i in 1:length(λs)
        λ = λs[i]
        copy!(γ, β); copy!(δ, β)
        nesterov = 0
        obj, old_obj = zero(T), typemax(T)

        for iter = 1:max_iters
            μ = (i == 1 && iter <= decay_iters) ?
                μ_max * (μ_min / μ_max)^(T(iter - 1) / T(decay_iters - 1)) :
                μ_min

            if strategy === "nesterov"
                nesterov += 1
                @. β = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
                @. δ = γ
            else
                nesterov = 0
                @. β = γ
                @. δ = γ
            end

            mul!(copy!(r, y), X, β, T(-1), T(1))
            obj = QuantileLoss(r, q, μ_min)
            l0_β_proj = l0_threshold!(β_proj, β, α; intercept=intercept)
            obj += λ * l0_β_proj + 0.5*λ/α * norm(β - β_proj, 2)^2

            if verbose
                phase = strategy === "nesterov" ?
                        @sprintf("nesterov k=%d", nesterov) : "original"
                @printf("%4d  obj=%.6g  μ=%.4g  α=%.4g  %s\n", iter, obj, μ, α, phase)
            end

            if old_obj < obj && strategy === "nesterov" &&
               (i == 1 && iter > decay_iters || i > 1)
                nesterov = 0
            end

            @inbounds @simd for j in eachindex(s)
                s[j] = (q - T(0.5)) + T(0.5) * sign(r[j]) * min(abs(r[j]) / μ_min, one(T))
            end
            mul!(grad, Transpose(X), s)
            @. grad = -grad / n + (λ/α) * (β - β_proj)

            if (i == 1 && iter > decay_iters || i > 1) && norm(grad) < tol_grad
                verbose && @printf("Converged at iter %d  ‖∇‖=%.4g\n", iter, norm(grad))
                break
            end

            l0_threshold!(β_proj_mm, β, α; intercept=intercept)
            MoreauProx!(s, r, μ, q)
            @. r = (y - s) / (2*μ*n)
            mul!(XTy, Transpose(X), r)
            @. XTy += λ/α * β_proj_mm
            lin_solve_MM!(β_mm, XTy, 1/(2*μ*n), λ/α, e, U, X, buffer_p, r, s)
            @. γ = β_mm
            old_obj = obj
        end

        verbose && @printf("λ=%.4g  solved\n", λ)
        βs[:, i] .= β_proj
    end
    return βs, λs
end

# -----------------------------------------------------------------------
# SparseQR — k (cardinality) version
# -----------------------------------------------------------------------
function SparseQR(X::AbstractMatrix, y::Vector, q::Real, ks::Vector{Int};
    μ              = 0.25,
    strategy       = "nesterov",
    tol_grad       = 1e-2,
    intercept      = true,
    verbose        = true,
)
    strategy in ("plain", "nesterov", "doubling") ||
        throw(ArgumentError("strategy must be \"plain\", \"nesterov\", or \"doubling\", got \"$strategy\""))
    ((n, p), T) = (size(X), eltype(X))
    max_iters = 1000
    μ = T(μ)
    r, s = zeros(T, n), zeros(T, n)
    order_buffer = Array{Int}(undef, p)
    G = n >= p ? X'X : X*X'
    XTy = Transpose(X) * y
    e, U = eigen(G)
    β = zeros(p)
    β_mm   = similar(β)
    β_proj = similar(β)
    buffer_p = similar(β)
    grad = similar(β)
    γ, δ = copy(β), copy(β)
    βs = zeros(p, length(ks))

    for i in 1:length(ks)
        ρ = 1.0
        k = ks[i]
        iter_total = 0

        while (ρ == 1) | (norm(β_proj .- β, 2) > 1e-4)
            copy!(γ, β); copy!(δ, β)
            nesterov = 0
            obj, old_obj = zero(T), typemax(T)

            for iter = 1:max_iters
                iter_total += 1

                if strategy === "nesterov"
                    nesterov += 1
                    @. β = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
                    @. δ = γ
                else
                    nesterov = 0
                    @. β = γ
                    @. δ = γ
                end

                mul!(copy!(r, y), X, β, T(-1), T(1))
                obj = QuantileLoss(r, q, μ)
                l0_proj!(β_proj, β, k, order_buffer; intercept=intercept)
                obj += 0.5*ρ * norm(β - β_proj, 2)^2

                if verbose
                    if strategy === "nesterov"
                        phase = @sprintf("nesterov k=%d", nesterov)
                    elseif strategy === "doubling"
                        phase = "doubling"
                    else
                        phase = "plain"
                    end
                    @printf("%4d  obj=%.6g  μ=%.4g  ρ=%.4g  %s\n", iter, obj, μ, ρ, phase)
                end

                if old_obj < obj && strategy === "nesterov"
                    nesterov = 0
                end

                @inbounds @simd for j in eachindex(s)
                    s[j] = (q - T(0.5)) + T(0.5) * sign(r[j]) * min(abs(r[j]) / μ, one(T))
                end
                mul!(grad, Transpose(X), s)
                @. grad = -grad / n + ρ * (β - β_proj)

                if norm(grad) < tol_grad
                    verbose && @printf("Converged at iter %d  ‖∇‖=%.4g\n", iter, norm(grad))
                    break
                end

                MoreauProx!(s, r, μ, q)
                @. r = (y - s) / (2*μ*n)
                mul!(XTy, Transpose(X), r)
                @. XTy += ρ * β_proj
                lin_solve_MM!(β_mm, XTy, 1/(2*μ*n), ρ, e, U, X, buffer_p, r, s)
                step = strategy === "doubling" ? T(1.5) : T(1)
                @. γ = β + step * (β_mm - β)
                old_obj = obj
            end

            ρ *= 1.2
        end

        verbose && @printf("k=%d  solved\n", k)
        βs[:, i] .= β_proj
    end
    return βs, ks
end

# -----------------------------------------------------------------------
# CV_SQR — λ version
# -----------------------------------------------------------------------
function CV_SQR(X::AbstractMatrix, y::Vector, q::Real, λs::Vector;
    nfold       = 10,
    intercept   = true,
    μ_max       = 10.0,
    μ_min       = 0.25,
    α           = 0.01,
    decay_iters = 100,
    strategy    = "nesterov",
    tol_grad    = 1e-3,
    verbose     = false,
)
    n, p = size(X)

    if intercept
        μ_X = mean(X[:, 2:end], dims=1)
        σ_X = std(X[:, 2:end],  dims=1)
        σ_X[σ_X .== 0] .= 1.0
        X_std = [X[:, 1] (X[:, 2:end] .- μ_X) ./ σ_X]
    else
        μ_X = mean(X, dims=1)
        σ_X = std(X,  dims=1)
        σ_X[σ_X .== 0] .= 1.0
        X_std = (X .- μ_X) ./ σ_X
    end

    indices     = randperm(n)
    fold_size   = div(n, nfold)
    fold_errors = zeros(nfold, length(λs))

    for fold_idx in 1:nfold
        val_start = (fold_idx - 1) * fold_size + 1
        val_end   = fold_idx == nfold ? n : fold_idx * fold_size
        val_idx   = indices[val_start:val_end]
        train_idx = setdiff(indices, val_idx)
        X_tr  = X_std[train_idx, :]
        y_tr  = y[train_idx]
        X_val = X_std[val_idx,   :]
        y_val = y[val_idx]
        βs, _ = SparseQR(X_tr, y_tr, q, λs;
            μ_max=μ_max, μ_min=μ_min, α=α,
            decay_iters=decay_iters, strategy=strategy,
            tol_grad=tol_grad, intercept=intercept, verbose=verbose)
        for i in 1:length(λs)
            fold_errors[fold_idx, i] = QuantileLoss(y_val .- X_val*βs[:,i], q, 0.001)
        end
    end

    mean_errors  = vec(mean(fold_errors, dims=1))
    best_idx     = argmin(mean_errors)
    best_λ       = λs[best_idx]

    βs_full, _ = SparseQR(X_std, y, q, λs;
        μ_max=μ_max, μ_min=μ_min, α=α,
        decay_iters=decay_iters, strategy=strategy,
        tol_grad=tol_grad, intercept=intercept, verbose=verbose)
    β_std = vec(βs_full[:, best_idx])

    if intercept
        β_orig        = similar(β_std)
        β_orig[2:end] = β_std[2:end] ./ vec(σ_X)
        β_orig[1]     = β_std[1] - dot(β_orig[2:end], vec(μ_X))
    else
        β_orig = β_std ./ vec(σ_X)
    end

    return best_λ, β_orig, mean_errors
end

# -----------------------------------------------------------------------
# CV_SQR — k version
# -----------------------------------------------------------------------
function CV_SQR(X::AbstractMatrix, y::Vector, q::Real, ks::Vector{Int};
    nfold     = 10,
    intercept = true,
    μ         = 0.25,
    strategy  = "nesterov",
    tol_grad  = 1e-2,
    verbose   = false,
)
    n, p = size(X)

    if intercept
        μ_X = mean(X[:, 2:end], dims=1)
        σ_X = std(X[:, 2:end],  dims=1)
        σ_X[σ_X .== 0] .= 1.0
        X_std = [X[:, 1] (X[:, 2:end] .- μ_X) ./ σ_X]
    else
        μ_X = mean(X, dims=1)
        σ_X = std(X,  dims=1)
        σ_X[σ_X .== 0] .= 1.0
        X_std = (X .- μ_X) ./ σ_X
    end

    indices     = randperm(n)
    fold_size   = div(n, nfold)
    fold_errors = zeros(nfold, length(ks))

    for fold_idx in 1:nfold
        val_start = (fold_idx - 1) * fold_size + 1
        val_end   = fold_idx == nfold ? n : fold_idx * fold_size
        val_idx   = indices[val_start:val_end]
        train_idx = setdiff(indices, val_idx)
        X_tr  = X_std[train_idx, :]
        y_tr  = y[train_idx]
        X_val = X_std[val_idx,   :]
        y_val = y[val_idx]
        βs, _ = SparseQR(X_tr, y_tr, q, ks;
            μ=μ, strategy=strategy,
            tol_grad=tol_grad, intercept=intercept, verbose=verbose)
        for i in 1:length(ks)
            fold_errors[fold_idx, i] = QuantileLoss(y_val .- X_val*βs[:,i], q, 0.001)
        end
    end

    mean_errors  = vec(mean(fold_errors, dims=1))
    best_idx     = argmin(mean_errors)
    best_k       = ks[best_idx]

    βs_full, _ = SparseQR(X_std, y, q, ks;
        μ=μ, strategy=strategy,
        tol_grad=tol_grad, intercept=intercept, verbose=verbose)
    β_std = vec(βs_full[:, best_idx])

    if intercept
        β_orig        = similar(β_std)
        β_orig[2:end] = β_std[2:end] ./ vec(σ_X)
        β_orig[1]     = β_std[1] - dot(β_orig[2:end], vec(μ_X))
    else
        β_orig = β_std ./ vec(σ_X)
    end

    return best_k, β_orig, mean_errors
end
