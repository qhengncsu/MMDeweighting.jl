module MMDeweighting
export QuantileLoss,FastQR, SparseQR, CV_SQR, FastL2E, L2EIsotonic, loglikelihood, FastLogistic, FastMultinomial, compute_probs!
export LowrankMultinomial,proj_rank!
using LinearAlgebra, Random, Statistics
include("QR.jl")
include("SQR.jl")
include("L2E.jl")
include("L2EIsotonic.jl")
include("Multinomial.jl")
include("LowrankMultinomial.jl")
end
