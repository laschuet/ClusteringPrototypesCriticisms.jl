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
    doubledkernelmeans = 2 * mean(K, dims=1)
    initialcandidates = 1:size(X, 2)
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
    prototypes(c::KmedoidsResult)

Return the indices of the cluster prototypes.

One cluster contains exactly one prototype. The clustering `c` is the result of the k-medoids algorithm.
"""
prototypes(c::KmedoidsResult) = c.medoids

"""
    prototypes(c::KmeansResult)

Return the indices of the cluster prototypes.

One cluster contains exactly one prototype. The clustering `c` is the result of the k-means algorithm.
"""
function prototypes(c::KmeansResult)
    clustercosts = [[] for _ = 1:nclusters(c)]
    ys = assignments(c)
    for i = 1:length(ys)
        push!(clustercosts[ys[i]], (c.costs[i], i))
    end
    return map(instancecosts -> instancecosts[2], minimum.(clustercosts))
end

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
    criticisms(c::KmeansResult)

Return the indices of the cluster criticisms.

One cluster contains exactly one criticism. The clustering `c` is the result of the k-means algorithm.
"""
function criticisms(c::KmeansResult)
    clustercosts = [[] for _ = 1:nclusters(c)]
    ys = assignments(c)
    for i = 1:length(ys)
        push!(clustercosts[ys[i]], (c.costs[i], i))
    end
    return map(instancecosts -> instancecosts[2], maximum.(clustercosts))
end

end # module
