# hard thresholding
function l0_threshold!(β_proj::Vector,β::Vector,λ::Real;intercept=true)
    p = length(β)
    copy!(β_proj,β)
    if intercept
        for i in 2:length(β)
            if abs(β[i]) < sqrt(2*λ)
                β_proj[i] = 0
            end
        end
        return sum(β_proj[2:end].!=0)
    else
        for i in 1:length(β)
            if abs(β[i]) < sqrt(2*λ)
                β_proj[i] = 0
            end
        end
        return sum(β_proj.!=0)
    end
end

# projection onto l0 norm ball
function l0_proj!(β_proj::Vector,β::Vector,k::Integer, order_buffer::Vector;intercept=true)
    if k==0
        copy!(β_proj,β)
        return
    end
    p = length(β)
    fill!(β_proj, zero(β[1]))
    order_buffer_nj = @view order_buffer[2:end]
    if intercept
        β_proj[1] = β[1]
        β_ni = @view β[2:end]
        β_proj_ni = @view β_proj[2:end]
        sortperm!(order_buffer_nj, β_ni, rev = true, by=abs)
        for i in 1:k
            index = order_buffer_nj[i]
            β_proj_ni[index] = β_ni[index]
        end
    else
        sortperm!(order_buffer, β, rev = true, by=abs)
        for i in 1:k
            index = order[i]
            β_proj[index] = β[index]
        end
    end
end

# solve (MX'X + ρI)β = RHS, invoking the woodbury formula should p>n  
function lin_solve_MM!(β::Vector,RHS::Vector,M::Real, ρ::Real, e::Vector, U::Matrix, 
                         X::Matrix, buffer1::Vector, buffer2::Vector, buffer3::Vector)
                        # buffer1 of length p, buffer2, buffer3 of length n
    (n, p)=size(X)
    if n >= p 
        mul!(buffer1,Transpose(U),RHS)
        @. buffer1 /= (M*e+ρ)
        mul!(β,U,buffer1)
    else
        mul!(buffer2, X, RHS)
        mul!(buffer3, Transpose(U), buffer2)
        @. buffer3 /= (1+M/ρ*e)
        mul!(buffer2, U, buffer3)
        mul!(β,Transpose(X),buffer2)
        axpby!(1/ρ,RHS,-M/ρ^2,β)
    end
end

# sparse quantile regression with hard thresholding
function SparseQR(X::Matrix, y::Vector, τ::Real, λs::Vector; h=0.25, tol=1e-6, verbose=true, μ=0.01)
    ((n, p), T) = (size(X), eltype(X)) # cases, predictors, precision
    (h, max_iters) = (T(h), 3000)
    (r, s) = (zeros(T, n), zeros(T, n))
    order_buffer = Array{Int}(undef, p)
    if n>=p
        G = X'X
    else
        G = X*X'
    end
    XTy = Transpose(X) * y
    (e,U) = eigen(G)
    β = zeros(p)
    β_proj = similar(β) 
    buffer_p = similar(β) 
    (γ, δ) = (copy(β), copy(β))
    βs = zeros(p,length(λs))
    for i in 1:length(λs)
        λ = λs[i]
        copy!(γ,β)
        copy!(δ,β)
        nesterov = 0
        (obj, old_obj) = (zero(T), Inf)
        for iter = 1:max_iters
            nesterov += 1
            @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
            @. δ = γ
            mul!(copy!(r, y), X, β, T(-1), T(1)) # r = y - X * β
            obj = QuantileLoss(r, τ, h)
            l0_β_proj = l0_threshold!(β_proj,β,μ)
            obj += λ* l0_β_proj + 0.5*λ/μ*norm(β-β_proj,2)^2
            if verbose
                println(iter,"  ",obj,"  ", h, " ",nesterov)
            end
            MoreauProx!(s, r, h, τ)
            @. r = (y - s)/(2*h*n)
            mul!(XTy, Transpose(X), r)
            @. XTy += λ/μ* β_proj
            lin_solve_MM!(β,XTy ,1/(2*h*n),λ/μ,e,U,X,buffer_p,r,s)
            if old_obj < obj
                nesterov = 0
            end
            if abs(obj-old_obj)<tol*(abs(obj)+1)
                break
            else
                @. γ = β
                old_obj = obj
            end
        end
        if verbose
          println("λ=",λ," Solved!")
        end
        βs[:,i] .=  β_proj
    end
    return βs, λs
end


# sparse quantile regression under proximal distance principle
function SparseQR(X::Matrix, y::Vector, τ::Real, ks::Vector{Int}; h=0.25, tol=1e-6, verbose=true)
    ((n, p), T) = (size(X), eltype(X)) # cases, predictors, precision
    (h, max_iters) = (T(h), 5000)
    (r, s) = (zeros(T, n), zeros(T, n))
    order_buffer = Array{Int}(undef, p)
    if n>=p
        G = X'X
    else
        G = X*X'
    end
    XTy = Transpose(X) * y
    (e,U) = eigen(G)
    β = zeros(p)
    β_proj = similar(β) 
    buffer_p = similar(β) 
    (γ, δ) = (copy(β), copy(β))
    βs = zeros(p,length(ks))
    for i in 1:length(ks)
        ρ = 1.0
        k = ks[i]
        #fill!(β, 0.0)
        while (ρ==1) | (norm(β_proj.-β,2)>1e-4)
            copy!(γ,β)
            copy!(δ,β)
            nesterov = 0
            (obj, old_obj) = (zero(T), Inf)
            for iter = 1:max_iters
                nesterov += 1
                @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
                @. δ = γ
                mul!(copy!(r, y), X, β, T(-1), T(1)) # r = y - X * β
                obj = QuantileLoss(r, τ, h)
                l0_proj!(β_proj,β,k,order_buffer)
                obj += 0.5*ρ*norm(β-β_proj,2)^2
                if verbose
                    println(iter,"  ",obj,"  ", h, " ",nesterov," ",ρ)
                end
                MoreauProx!(s, r, h, τ)
                @. r = (y - s)/(2*h*n)
                mul!(XTy, Transpose(X), r)
                @. XTy += ρ* β_proj
                lin_solve_MM!(β,XTy ,1/(2*h*n),ρ,e,U,X,buffer_p,r,s)
                if old_obj < obj
                    nesterov = 0
                end
                if abs(obj-old_obj)<tol*(abs(obj)+1)
                    break
                else
                    @. γ = β
                    old_obj = obj
                end
            end
            ρ = ρ*1.2
        end
        if verbose
          println("k=",k," Solved!")
        end
        βs[:,i] .=  β_proj
    end
    return βs, ks
end

# cross validation for sparse quantile regression with proximal distance principle
function CV_SQR(X::Matrix, y::Vector, τ::Real, ks::Vector{Int}; nfold=10, h=0.25, tol=1e-6)
    n = size(X, 1)
    indices = collect(1:n)
    Random.shuffle!(indices)
    cv_error = zeros(length(ks))
    fold_errors = zeros(nfold,length(ks))
    fold_size = div(n, nfold)
    for fold_idx in 1:nfold
        val_start = (fold_idx - 1) * fold_size + 1
        val_end = fold_idx == nfold ? n : fold_idx * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = setdiff(indices, val_indices)
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_val = X[val_indices, :]
        y_val = y[val_indices]
        βs,_ = SparseQR(X_train, y_train, τ, ks; h=h, tol=tol, verbose=false)
        for i in 1:length(ks)
            y_pred = X_val*βs[:,i]
            fold_errors[fold_idx,i] =  QuantileLoss(y_val .- y_pred, τ, h)
        end
    end
    mean_errors = vec(mean(fold_errors,dims=1))
    best_k_index = argmin(mean_errors)
    best_k = ks[best_k_index]
    βs, _ = SparseQR(X, y, τ, ks; h=h, tol=tol, verbose=false)
    best_β = vec(βs[:,best_k_index])
    return best_k, best_β, mean_errors
end

# cross validation for sparse quantile regression with hard thresholding
function CV_SQR(X::Matrix, y::Vector, τ::Real, λs::Vector; nfold=10, h=0.25, tol=1e-6)
    n = size(X, 1)
    indices = collect(1:n)
    Random.shuffle!(indices)
    cv_error = zeros(length(λs))
    fold_errors = zeros(nfold,length(λs))
    fold_size = div(n, nfold)
    for fold_idx in 1:nfold
        val_start = (fold_idx - 1) * fold_size + 1
        val_end = fold_idx == nfold ? n : fold_idx * fold_size
        val_indices = indices[val_start:val_end]
        train_indices = setdiff(indices, val_indices)
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_val = X[val_indices, :]
        y_val = y[val_indices]
        βs,_ = SparseQR(X_train, y_train, τ, λs; h=h, tol=tol, verbose=false)
        for i in 1:length(λs)
            y_pred = X_val*βs[:,i]
            fold_errors[fold_idx,i] =  QuantileLoss(y_val .- y_pred, τ, h)
        end
    end
    mean_errors = vec(mean(fold_errors,dims=1))
    best_λ_index = argmin(mean_errors)
    best_λ = λs[best_λ_index]
    βs, _ = SparseQR(X, y, τ, λs; h=h, tol=tol, verbose=false)
    best_β = vec(βs[:,best_λ_index])
    return best_λ, best_β, mean_errors
end