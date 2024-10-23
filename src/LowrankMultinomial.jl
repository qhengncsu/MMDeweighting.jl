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
                            tol::Real=1e-4, verbose=true,μ=0.01, hard=false) 
    ((n, p), q, max_iters) = (size(X), size(Y, 2),3000)
    ηs, r = zeros(n,q), zeros(n,q)
    (e1,U) = eigen(X'X)
    E = 2 .*(Diagonal(ones(q)) + ones(q)*Transpose(ones(q)))
    (e2,V) = eigen(E)
    Ve2 = V*Diagonal(e2)
    B = zeros(p,q)
    direction, grad = similar(B),similar(B)
    B_prox, buffer1, buffer2 = similar(B),similar(B),similar(B)
    row_sums_buffer = zeros(n)
    Bs = zeros(p,q,length(λs))
    B_prox_nn = 0.0
    P = zeros(n,q)
    (γ, δ) = (copy(B), copy(B))
    for j in 1:length(λs)
        λ = λs[j]
        nesterov = 0
        (obj, old_obj) = (0.0, Inf)
        #if hard
            #fill!(B, 0.0)
        #end
        (γ, δ) = (copy(B), copy(B))
        for iteration = 1:max_iters
            nesterov += 1
            @. B = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
            @. δ = γ # Nesterov acceleration
            mul!(ηs, X, B) # ηs = X * B
            compute_probs!(P,ηs,row_sums_buffer)
            obj = -loglikelihood(Y, P)/n
            if !hard
                B_prox_nn = svt!(B_prox, B, μ)
                obj += λ*B_prox_nn + 0.5*λ/μ*norm(B - B_prox,2)^2
            else
                rank = prox_rank!(B_prox, B, μ)
                obj += λ*rank + 0.5*λ/μ*norm(B - B_prox,2)^2
            end            
            if verbose
                println(iteration," ",obj," ",nesterov)
            end
            @. r = Y - P
            mul!(grad, transpose(X), -r/n)
            @. grad += λ/μ*(B - B_prox)
            solve_sylvester!(direction,grad,1/n,e1,U,λ/μ,e2,V,Ve2,buffer1,buffer2)
            @. B = B - direction
            if old_obj < obj 
                nesterov = 0
            end
            if norm(grad,2)<tol
                break
            else
                @. γ = B
                old_obj = obj
            end
        end
        if verbose
            println("λ= ",λ," finished!")
        end
        Bs[:,:,j] .=  B
    end
    return Bs, λs
end