#smoothed quantile loss
function QuantileLoss(r::Vector, τ::Real, h::Real)
  l = zero(eltype(r))
  taumhalf = τ-0.5
  hover2 = h/2
  oneovertwoh = 1/2*h
  for i in eachindex(r)
    abs_ri = abs(r[i])
    l += 0.5*ifelse(abs_ri>h,abs_ri,hover2+r[i]^2*oneovertwoh)+taumhalf*r[i]
  end
  l/length(r)
end

function MoreauProx!(s, r, h, τ)
  twotaum1timesh = (2*τ-1)*h
  for i in eachindex(s)
    s[i] = ifelse(r[i] > h, r[i] - h, 0) + ifelse(r[i] < - h, r[i] + h, 0) - twotaum1timesh
  end
end

# nonsparse quantile regression
function FastQR(X::Matrix, y::Vector, τ::Real; h=0.25, tol=1e-6, verbose=true)
  ((n, p), T) = (size(X), eltype(X)) # cases, predictors, float precision
  (h, max_iters) = (T(h), 3000)
  (r, s) = (zeros(T, n), zeros(T, n)) # work vectors
  XTy = Transpose(X) * y
  XTX = Transpose(X) * X
  L = cholesky!(XTX)
  β = L\XTy # least squares initialization
  (obj, old_obj) = (zero(T), Inf)
  (γ, δ) = (copy(β), copy(β))
  nesterov = 0 
  iter = 0
  for iter = 1:max_iters
    nesterov = nesterov + 1
    @. β = γ + ((nesterov - 1)/(nesterov + 2)) * (γ - δ)
    @. δ = γ # Nesterov acceleration
    mul!(copy!(r, y), X, β, T(-1), T(1)) # r = y - X * β
    obj = QuantileLoss(r, τ, h)
    if verbose
      println(iter,"  ",obj,"  ", h, " ",k)
    end
    MoreauProx!(s, r, h, τ) # MM shift
    @. r = y - s #shift the response
    mul!(XTy, Transpose(X), r)
    ldiv!(β, L, XTy)
    if old_obj < obj 
      nesterov = 0 
    end # restart Nesterov momentum
    if abs(obj-old_obj)<tol*(abs(old_obj+1)) && iter>1
      return (β, obj, iter, h)
    else
      @. γ = β
      old_obj = obj
    end
  end
  return (β, obj, iter, h)
end