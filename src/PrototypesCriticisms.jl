module PrototypesCriticisms

using Clustering
using KernelFunctions
using Statistics

export criticisms,
        prototypes,
        sqmmd, mmd²,
        witness

"""
    sqmmd(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel=RBFKernel())
    mmd²(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel=RBFKernel())

Compute the squared maximum mean discrepancy between `X` and `Y` using the kernel function `k`.

`X` and `Y` are expected to store observations in columns.
"""
sqmmd(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel=RBFKernel()) = mean(kernelmatrix(k, X, obsdim=2)) - 2 * mean(kernelmatrix(k, X, Y, obsdim=2)) + mean(kernelmatrix(k, Y, obsdim=2))
const mmd² = sqmmd

"""
    prototypes(X::AbstractMatrix{<:Real}, n::Int, k::Kernel=RBFKernel())

Return the indices of `n` prototypes in `X` using the kernel function `k`.

`X` is expected to store observations in columns.
"""
function prototypes(X::AbstractMatrix{<:Real}, n::Int, k::Kernel=RBFKernel())
    K = kernelmatrix(k, X, obsdim=2)
    return prototypes(n, K)
end

"""
    prototypes(n::Int, K::AbstractMatrix{<:Real})

Return the indices of `n` prototypes using the kernel matrix `K`.
"""
function prototypes(n::Int, K::AbstractMatrix{<:Real})
    doubledkernelmeans = 2 * mean(K, dims=1)
    initialcandidates = 1:size(K, 2)
    protoids = Int[]
    while length(protoids) < n
        candidates = setdiff(initialcandidates, protoids)
        avgproximities1 = vec(doubledkernelmeans[candidates])
        if length(protoids) > 0
            proximities2 = vec(2 * sum(K[protoids, candidates], dims=1))
            avgproximities2 = proximities2 / (length(protoids) + 1)
            diffs = avgproximities1 - avgproximities2
        else
            diffs = avgproximities1
        end
        protoid = candidates[argmax(diffs)]
        push!(protoids, protoid)
    end
    return protoids
end

"""
    prototypes(c::KmedoidsResult, n::Int=1)

Return the indices of the `n` prototypes for every cluster of the k-medoids clustering `c`.
"""
function prototypes(c::KmedoidsResult, n::Int=1)
    n == 1 && return [[i] for i in c.medoids]
    return _instances(nclusters(c), assignments(c), c.costs, n)
end

"""
    prototypes(c::KmeansResult, n::Int=1)

Return the indices of the `n` prototypes for every cluster of the k-means clustering `c`.
"""
prototypes(c::KmeansResult, n::Int=1) = _instances(nclusters(c), assignments(c), c.costs, n)

"""
    prototypes(c::FuzzyCMeansResult, n::Int=1)

Return the indices of the `n` prototypes for every cluster of the fuzzy c-means clustering `c`.
"""
prototypes(c::FuzzyCMeansResult, n::Int=1) = _instances(c.weights, n)

"""
    witness(z::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel=RBFKernel())

Compute the witness function of `X` and `Y` at `z` using the kernel function `k`.

`X` and `Y` are expected to store observations in columns, and `z` is a single observation.
"""
witness(z::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel=RBFKernel()) = mean(map(x -> k(z, x), eachcol(X))) - mean(map(y -> k(z, y), eachcol(Y)))

"""
    criticisms(X::AbstractMatrix{<:Real}, protoids::AbstractVector{Int}, n::Int, k::Kernel=RBFKernel())

Return the indices of `n` criticisms in `X` using the prototype indices `protoids` and kernel function `k`.

`X` is expected to store observations in columns.
"""
function criticisms(X::AbstractMatrix{<:Real}, protoids::AbstractVector{Int}, n::Int, k::Kernel=RBFKernel())
    K = kernelmatrix(k, X, obsdim=2)
    kernelmeans = mean(K, dims=1)
    initialcandidates = setdiff(1:size(X, 2), protoids)
    critids = Int[]
    while length(critids) < n
        candidates = setdiff(initialcandidates, critids)
        avgproximities1 = kernelmeans[candidates]
        avgproximities2 = mean(K[protoids, candidates], dims=1)
        absws = abs.(vec(avgproximities1) - vec(avgproximities2))
        critid = candidates[argmax(absws)]
        push!(critids, critid)
    end
    return critids
end

"""
    criticisms(c::KmedoidsResult, n::Int=1)

Return the indices of the `n` criticisms for every cluster of the k-medoids clustering `c`.
"""
criticisms(c::KmedoidsResult, n::Int=1) = _instances(nclusters(c), assignments(c), c.costs, n, true)

"""
    criticisms(c::KmeansResult, n::Int=1)

Return the indices of the `n` criticisms for every cluster of the k-means clustering `c`.
"""
criticisms(c::KmeansResult, n::Int=1) = _instances(nclusters(c), assignments(c), c.costs, n, true)

"""
    criticisms(c::FuzzyCMeansResult, n::Int=1)

Return the indices of the `n` criticisms for every cluster of the fuzzy c-means clustering `c`.
"""
criticisms(c::FuzzyCMeansResult, n::Int=1) = _instances(c.weights, n, true)

# Return instances via their assignment costs
function _instances(k::Int, assignments::Vector{Int}, costs::Vector{<:Real}, n::Int, rev::Bool=false)
    clustercosts = [[] for _ in 1:k]
    for i = 1:length(assignments)
        push!(clustercosts[assignments[i]], (costs[i], i))
    end
    return map.(i -> i[2], partialsort!.(clustercosts, [1:n], rev=rev))
end

# Return instances via their assignment weights
function _instances(W::AbstractMatrix{<:Real}, n::Int=1, rev::Bool=false)
    instances = Vector{Vector{Int}}()
    for i = 1:size(W, 2)
        push!(instances, partialsortperm(view(W, :, i), 1:n, rev=rev))
    end
    return instances
end

end # module
