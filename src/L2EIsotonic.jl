function L2EIsotonic(y::Vector, β0::Vector, τ0::Real; tol::Real=1e-6, MM=true, verbose=true)
    max_iters = 1000
    β = copy(β0)
    τ = τ0
    n = length(y)
    D = zeros(n-1,n)
    Dβ= zeros(n-1)
    PCDβ= zeros(n-1)
    for i in 1:(n-1)
        D[i, i] = -1
        D[i, i+1] = 1
    end
    DtD = Transpose(D)*D
    (e,U) = eigen(DtD)
    r, w, ytilde, RHS = similar(y), similar(y), similar(y), similar(y)
    obj = 1.0
    for j in 1:10
        ρ = 1.0
        while (ρ==1) | (norm(Dβ.-PCDβ,2)>1e-4)
            (γ, δ) = (copy(β), copy(β))
            nesterov = 0
            obj, old_obj = 0.0, Inf
            for iter = 1:max_iters
                nesterov += 1
                @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
                @. δ = γ # Nesterov acceleration
                @. r = y - β
                obj = L2ELoss(r, τ)
                mul!(Dβ,D,β)
                for i in 1:(n-1)
                    if Dβ[i]>0
                        PCDβ[i] = Dβ[i]
                    else
                        PCDβ[i] = 0
                    end
                end
                obj += 0.5*ρ*norm(Dβ-PCDβ,2)^2
                weights!(w, r, τ)
                if MM
                    @. ytilde = y*w + β*(1-w)
                    mul!(RHS,Transpose(D),ρ*PCDβ)
                    @. RHS += ytilde * τ^3/n * sqrt(2/pi)
                    mul!(ytilde,Transpose(U),RHS)
                    @. ytilde /= (ρ*e+τ^3/n * sqrt(2/pi))
                    mul!(β,U,ytilde)
                else
                    @. ytilde = y*w
                    mul!(RHS,Transpose(D),ρ*PCDβ)
                    @. RHS += ytilde * τ^3/n * sqrt(2/pi)
                    β .= (ρ*DtD+τ^3/n * sqrt(2/pi)*diagm(w))\RHS
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
            ρ = ρ*1.2
        end
        @. r = y - β
        weights!(w, r, τ)
        τ, obj = update_τ(τ,r,w,verbose=false)
        if verbose
            println(j,"  ",obj," tau = ",τ)
        end
    end
    return β, τ, obj, w
end