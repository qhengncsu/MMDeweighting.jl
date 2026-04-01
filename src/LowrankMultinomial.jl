function solve_sylvester!(B::Matrix,RHS::Matrix,M::Real,e1::Vector,U::Matrix,ρ::Real,e2::Vector,V::Matrix,Ve2,
                           buffer1::Matrix, buffer2::Matrix)
    (p, q)=size(B)
    mul!(buffer1,Transpose(U),RHS)
    mul!(buffer2,buffer1,Ve2)
    for j in 1:q
        for i in 1:p
            buffer2[i,j] = buffer2[i,j]/(e1[i]*M+e2[j]*ρ)
        end
    end
    mul!(buffer1,U,buffer2)
    mul!(B,buffer1,Transpose(V))
end
    
function proj_rank!(B_proj::Matrix, B::Matrix, rank::Int; intercept=true) 
    if intercept
        F = svd(B[2:end,:],alg=LinearAlgebra.QRIteration())
    else
        F = svd(B,alg=LinearAlgebra.QRIteration())
    end
    F.S[(rank+1):end] .= 0
    if intercept
        # first row of B is the 1xm intercept term
        B_proj[1,:] .= B[1,:]
        B_proj[2:end,:] .= F.U * Diagonal(F.S) * F.Vt
    else
        B_proj .= F.U * Diagonal(F.S) * F.Vt
    end
end

function svt!(B_prox::Matrix, B::Matrix, λ::Real; intercept=true) 
    if intercept
        F = svd(B[2:end,:],alg=LinearAlgebra.QRIteration())
    else
        F = svd(B,alg=LinearAlgebra.QRIteration())
    end
    for i in eachindex(F.S)
        if F.S[i] < λ
            F.S[i] = 0
        else
            F.S[i] -= λ
        end
    end
    if intercept
        # first row of B is the 1xm intercept term
        B_prox[1, :] .= B[1, :]
        B_prox[2:end,:] .= F.U * Diagonal(F.S) * F.Vt
    else
        B_prox .= F.U * Diagonal(F.S) * F.Vt
    end
    return sum(F.S)
end
        
function prox_rank!(B_prox::Matrix, B::Matrix, μ::Real; intercept=true)
    if intercept
        F = svd(B[2:end,:],alg=LinearAlgebra.QRIteration())
    else
        F = svd(B,alg=LinearAlgebra.QRIteration())
    end
    for i in 1:length(F.S)
        if F.S[i] < sqrt(2*μ)
            F.S[i] = 0
        end
    end
    if intercept
        # first row of B is the 1xm intercept term
        B_prox[1,:] .= B[1,:]
        B_prox[2:end,:] .= F.U * Diagonal(F.S) * F.Vt
    else
        B_prox .= F.U * Diagonal(F.S) * F.Vt
    end
    return sum(F.S.!=0)
end
                
function LowrankMultinomial(X::Matrix, Y::Matrix, λs::Vector;
    tol            = 1e-4,
    μ_max          = 1.0,
    μ_min          = 0.01,
    decay_iters    = 100,
    hard           = false,
    strategy       = "nesterov",
    doubling_start = 21,
    verbose        = true,
)
    strategy in ("original", "nesterov", "step-doubling") ||
        throw(ArgumentError("strategy must be \"original\", \"nesterov\", or \"step-doubling\", got \"$strategy\""))
    ((n, p), c, max_iters) = (size(X), size(Y, 2), 3000)
    q    = c - 1
    ηs   = zeros(n, q)
    r    = zeros(n, q)
    e1, U = eigen(X'X)
    E     = 2 .* (Diagonal(ones(q)) + ones(q) * ones(q)')
    e2, V = eigen(E)
    Ve2   = V * Diagonal(e2)
    B         = zeros(p, q)
    direction = similar(B)
    grad      = similar(B)
    B_prox    = similar(B)
    B_prox_mm = similar(B)
    buffer1   = similar(B)
    buffer2   = similar(B)
    row_sums_buffer = zeros(n)
    Bs  = zeros(p, q, length(λs))
    P   = zeros(n, c)
    γ, δ = copy(B), copy(B)
    for j in 1:length(λs)
        λ = λs[j]
        copy!(γ, B); copy!(δ, B)
        nesterov = 0
        obj, old_obj = 0.0, Inf
        for iter in 1:max_iters
            μ = (j == 1 && iter <= decay_iters) ?
                μ_max * (μ_min / μ_max)^((iter - 1) / (decay_iters - 1)) :
                μ_min
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
            obj = -loglikelihood(Y, P) / n
            if !hard
                B_prox_nn = svt!(B_prox, B, μ_min)
                obj += λ * B_prox_nn + 0.5*λ/μ_min * norm(B - B_prox, 2)^2
            else
                rank = prox_rank!(B_prox, B, μ_min)
                obj += λ * rank + 0.5*λ/μ_min * norm(B - B_prox, 2)^2
            end
            if verbose
                if strategy === "nesterov"
                    phase = @sprintf("nesterov k=%d", nesterov)
                elseif strategy === "step-doubling"
                    phase = iter >= doubling_start ? "step-doubling" : "plain"
                else
                    phase = "original"
                end
                @printf("%4d  obj=%.6g  μ=%.4g  %s\n", iter, obj, μ, phase)
            end
            if old_obj < obj && strategy === "nesterov" &&
               (j == 1 && iter > decay_iters || j > 1)
                nesterov = 0
            end
            @. r = Y[:, 1:q] - P[:, 1:q]
            mul!(grad, transpose(X), -r / n)
            if !hard
                svt!(B_prox, B, μ_min)
            else
                prox_rank!(B_prox, B, μ_min)
            end
            @. grad += λ/μ_min * (B - B_prox)
            if (j == 1 && iter > decay_iters || j > 1) && norm(grad) < tol
                verbose && @printf("Converged at iter %d  ‖∇‖=%.4g\n", iter, norm(grad))
                break
            end
            if !hard
                svt!(B_prox_mm, B, μ)
            else
                prox_rank!(B_prox_mm, B, μ)
            end
            mul!(grad, transpose(X), -r / n)
            @. grad += λ/μ * (B - B_prox_mm)
            solve_sylvester!(direction, grad, 1/n, e1, U, λ/μ, e2, V, Ve2, buffer1, buffer2)
            @. B = B - direction
            step = (strategy === "step-doubling" && iter >= doubling_start) ? 2.0 : 1.0
            @. γ = B - (step - 1) * direction
            old_obj = obj
        end
        verbose && @printf("λ=%.4g  solved\n", λ)
        Bs[:, :, j] .= B_prox
    end
    return Bs, λs
end