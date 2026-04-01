module MMDeweighting
export QuantileLoss, MoreauProx!, FastQR, SparseQR, CV_SQR
export FastL2E, L2EIsotonic
export loglikelihood, FastLogistic, FastMultinomial
export compute_probs_reduced!
export LowrankMultinomial, proj_rank!
using LinearAlgebra, Random, Statistics, Printf
include("QR.jl")
include("SQR.jl")
include("L2E.jl")
include("L2EIsotonic.jl")
include("Multinomial.jl")
include("LowrankMultinomial.jl")
end
