function QuantileLoss(r::Vector, q::Real, μ::Real)
  T = eltype(r)
  l, half = zero(T), T(0.5)
  qmhalf = q - half
  μover2 = half * μ
  oneover2μ = 1 / (2μ)
  @inbounds @simd for i in eachindex(r)
    abs_ri = abs(r[i])
    l += half * ifelse(abs_ri > μ, abs_ri, μover2 + r[i]^2 * oneover2μ) + qmhalf * r[i]
  end
  l / length(r)
end

function MoreauProx!(s, r, μ, q)
  two_qm1_times_μ = (2q - 1) * μ
  @inbounds @simd for i in eachindex(s)
    s[i] = sign(r[i]) * max(abs(r[i]) - μ, 0) - two_qm1_times_μ
  end
  s
end

function FastQR(
    X::Matrix, y::Vector, q::Real;
    μ_max          = 10.0,
    μ_min          = 0.25,
    decay_iters    = 20,
    tol            = 1e-6,
    max_iters      = 3000,
    strategy       = "nesterov",
    doubling_start = 21,
    verbose        = true,
)
    strategy in ("plain", "nesterov", "doubling") ||
        throw(ArgumentError("strategy must be \"plain\", \"nesterov\", or \"doubling\", got \"$strategy\""))
    (n, p), T = size(X), eltype(X)
    μ_max, μ_min = T(μ_max), T(μ_min)
    r, s = zeros(T, n), zeros(T, n)
    XTX = X'X
    XTy = X'y
    L   = cholesky!(XTX)
    β        = L \ XTy
    γ        = copy(β)
    δ        = copy(β)
    γ_mm     = similar(β)
    obj      = zero(T)
    old_obj  = typemax(T)
    nesterov = 0
    for iter in 1:max_iters
        μ = iter <= decay_iters ?
            μ_max * (μ_min / μ_max)^(T(iter - 1) / T(decay_iters - 1)) :
            μ_min
        if strategy === "nesterov"
            nesterov += 1
            @. β = γ + ((nesterov - 1) / (nesterov + 2)) * (γ - δ)
            @. δ = γ
        else
            @. β = γ
            @. δ = γ
        end
        copy!(r, y)
        mul!(r, X, β, T(-1), T(1))
        obj = QuantileLoss(r, q, μ_min)
        if verbose
            if strategy === "plain"
                phase = "plain"
            elseif strategy === "nesterov"
                phase = @sprintf("nesterov k=%d", nesterov)
            else
                phase = "doubling"
            end
            @printf("%4d  obj=%.6g  μ=%.4g  %s\n", iter, obj, μ, phase)
        end
        if old_obj < obj && strategy === "nesterov" && iter > decay_iters
            nesterov = 0
        end
        if abs(obj - old_obj) < tol * (abs(old_obj) + 1) && iter > decay_iters
            verbose && @printf("Converged at iter %d\n", iter)
            return (β, obj, iter, μ)
        end
        MoreauProx!(s, r, μ, q)
        @. r = y - s
        mul!(XTy, X', r)
        ldiv!(γ_mm, L, XTy)
        step = strategy === "doubling" ? T(1.5) : T(1)
        @. γ = β + step * (γ_mm - β)
        old_obj = obj
        @. β = γ
    end
    verbose && @printf("Max iters (%d) reached\n", max_iters)
    return (β, obj, max_iters, μ_min)
end