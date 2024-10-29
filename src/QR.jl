# smoothed quantile loss
function QuantileLoss(r::Vector, τ::Real, h::Real)
  T = eltype(r)
  l, half = zero(T), T(0.5)
  taumhalf = τ - half
  hover2 = half * h
  oneover2h = 1 / (2h)
  @inbounds @simd for i in eachindex(r)
    abs_ri = abs(r[i])
    l += half * ifelse(abs_ri > h, abs_ri, hover2 + r[i]^2 * oneover2h) + taumhalf * r[i]
  end
  l / length(r)
end

function MoreauProx!(s, r, h, τ)
  twotaum1timesh = (2τ - 1) * h
  @inbounds @simd for i in eachindex(s)
    s[i] = sign(r[i]) * max(abs(r[i]) - h, 0) - twotaum1timesh
  end
  s
end

# nonsparse quantile regression
function FastQR(X::Matrix, y::Vector, τ::Real; h=0.25, tol=1e-6, verbose=true)
  ((n, p), T) = (size(X), eltype(X)) # cases, predictors, float precision
  (h, max_iters) = (T(h), 3000)
  (r, s) = (zeros(T, n), zeros(T, n)) # work vectors
  XTX, XTy = X'X, X'y
  L = cholesky!(XTX)
  β = L \ XTy # least squares initialization
  (obj, old_obj) = (zero(T), Inf)
  (γ, δ) = (copy(β), copy(β))
  nesterov = 0 
  iter = 0
  for iter = 1:max_iters
    nesterov = nesterov + 1
    @. β = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
    @. δ = γ # Nesterov acceleration
    mul!(copy!(r, y), X, β, T(-1), T(1)) # r = y - X * β
    obj = QuantileLoss(r, τ, h)
    if verbose
      println(iter,"  ",obj,"  ", h, " ",k)
    end
    MoreauProx!(s, r, h, τ) # MM shift
    # shift the response
    @inbounds @simd for i in eachindex(r)
      r[i] = y[i] - s[i]
    end
    mul!(XTy, Transpose(X), r)
    ldiv!(β, L, XTy)
    # restart Nesterov momentum
    if old_obj < obj 
      nesterov = 0 
    end
    # convergence check 
    if abs(obj - old_obj) < tol * (abs(old_obj + 1)) && iter > 1
      return (β, obj, iter, h)
    else
      @. γ = β
      old_obj = obj
    end
  end
  return (β, obj, iter, h)
end