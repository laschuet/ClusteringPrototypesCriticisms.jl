module PrototypesCriticisms

using Clustering
using Distances
using KernelFunctions
using LinearAlgebra
using Statistics

export criticisms,
        prototypes,
        sqmmd, mmd²,
        witness

abstract type AbstractMethod end

struct MMDCritic <: AbstractMethod end
struct KMeans <: AbstractMethod end
struct KMedoids <: AbstractMethod end
struct FuzzyCMeans <: AbstractMethod end
struct AffinityPropagation <: AbstractMethod end

"""
    prototypes(X::AbstractMatrix{<:Real}, ys::Vector{Int}, n::Int, k::Kernel)

Return the indices of the `n` prototypes for every cluster in `X` using the kernel function `k`.
The cluster assignments of the observations are specified by `ys`.

`X` is expected to store observations in columns.
"""
function prototypes(X::AbstractMatrix{<:Real}, ys::Vector{Int}, n::Int, k::Kernel)
    protoids = Vector{Vector{Int}}()
    numclusters = length(unique(ys))
    for i = 1:numclusters
        v = view(X, :, ys .== i)
        push!(protoids, prototypes(v, k, n))
    end
    return protoids
end

"""
    prototypes(X::AbstractMatrix{<:Real}, ys::Vector{Int}, n::Int, s::Symbol)

Return the indices of the `n` prototypes for every cluster in `X` using the method specified by the method's symbolic name `s`.
The cluster assignments of the observations are specified by `ys`.

`X` is expected to store observations in columns.
`s` must be one of `:kmedoids`, `:kmeans`, `:fuzzycmeans`, and `:affinitypropagation`.
"""
prototypes(X::AbstractMatrix{<:Real}, ys::Union{Vector{Int}, Matrix{<:Real}}, n::Int, s::Symbol) = _instances(X, ys, n, _method(s))

"""
    prototypes(X::AbstractMatrix{<:Real}, k::Kernel, n::Int)

Return the indices of the `n` prototypes in `X` using the kernel function `k`.

`X` is expected to store observations in columns.
"""
function prototypes(X::AbstractMatrix{<:Real}, k::Kernel, n::Int)
    K = kernelmatrix(k, X, obsdim=2)
    return prototypes(K, n)
end

"""
    prototypes(K::AbstractMatrix{<:Real}, n::Int)

Return the indices of the `n` prototypes using the kernel matrix `K`.
"""
function prototypes(K::AbstractMatrix{<:Real}, n::Int)
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
prototypes(c::FuzzyCMeansResult, n::Int=1) = _instances(c.weights, n, true)

"""
    prototypes(c::AffinityPropResult, X::AbstractMatrix{<:Real}, n::Int=1; dist::Metric=Euclidean())

Return the indices of the `n` prototypes for every cluster of the affinity propagation clustering `c`.
To find `n>1` prototypes, the observations `X` are used for comparing the distances between observations and cluster centers.

# Keyword arguments
- `dist`: the distance metric used for computing the distances.
"""
function prototypes(c::AffinityPropResult, X::AbstractMatrix{<:Real}, n::Int=1; dist::Metric=Euclidean())
    n == 1 && return [[i] for i in c.exemplars]
    D = pairwise(dist, X, dims=2)
    instances = Vector{Vector{Int}}()
    for i = 1:nclusters(c)
        v = view(D, assignments(c) .== i, c.exemplars[i])
        permutation = partialsortperm(v, 1:n)
        parentinstances = parentindices(v)[1]
        push!(instances, parentinstances[permutation])
    end
    return instances
end

"""
    criticisms(X::AbstractMatrix{<:Real}, ys::AbstractVector{Int}, protoids::AbstractVector{Int}, n::Int, k::Kernel)

Return the indices of the `n` criticisms for every cluster in `X` using the prototype indices `protoids` and the kernel function `k`.
The cluster assignments of the observations are specified by `ys`.

`X` is expected to store observations in columns.
"""
function criticisms(X::AbstractMatrix{<:Real}, ys::Vector{Int}, protoids::AbstractVector{Vector{Int}}, n::Int, k::Kernel)
    critids = Vector{Vector{Int}}()
    numclusters = length(unique(ys))
    for i = 1:numclusters
        v = view(X, :, ys .== i)
        push!(critids, criticisms(v, k, protoids[i], n))
    end
    return critids
end

"""
    criticisms(X::AbstractMatrix{<:Real}, ys::Vector{Int}, n::Int, s::Symbol)

Return the indices of the `n` criticisms for every cluster in `X` using the method specified by the method's symbolic name `s`.
The cluster assignments of the observations are specified by `ys`.

`X` is expected to store observations in columns.
`s` must be one of `:kmedoids`, `:kmeans`, `:fuzzycmeans`, and `:affinitypropagation`.
"""
criticisms(X::AbstractMatrix{<:Real}, ys::Union{Vector{Int}, Matrix{<:Real}}, n::Int, s::Symbol) = _instances(X, ys, n, _method(s), true)

"""
    criticisms(X::AbstractMatrix{<:Real}, k::Kernel, protoids::AbstractVector{Int}, n::Int)

Return the indices of the `n` criticisms in `X` using the prototype indices `protoids` and the kernel function `k`.

`X` is expected to store observations in columns.
"""
function criticisms(X::AbstractMatrix{<:Real}, k::Kernel, protoids::AbstractVector{Int}, n::Int)
    K = kernelmatrix(k, X, obsdim=2)
    return criticisms(K, protoids, n)
end

"""
    criticisms(K::AbstractMatrix{<:Real}, protoids::AbstractVector{Int}, n::Int)

Return the indices of the `n` criticisms using the prototype indices `indices` and the kernel matrix `K`.
"""
function criticisms(K::AbstractMatrix{<:Real}, protoids::AbstractVector{Int}, n::Int)
    kernelmeans = mean(K, dims=1)
    initialcandidates = setdiff(1:size(K, 2), protoids)
    critids = Int[]
    while length(critids) < n
        candidates = setdiff(initialcandidates, critids)
        avgproximities1 = kernelmeans[candidates]
        avgproximities2 = mean(K[protoids, candidates], dims=1)
        absws = abs.(vec(avgproximities1) - vec(avgproximities2)) .+ logdet(K[critids, critids])
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
criticisms(c::FuzzyCMeansResult, n::Int=1) = _instances(c.weights, n)

"""
    criticisms(c::AffinityPropResult, X::AbstractMatrix{<:Real}, n::Int=1; dist::Metric=Euclidean())

Return the indices of the `n` criticisms for every cluster of the affinity propagation clustering `c`.
To find `n>1` criticisms, the observations `X` are used for comparing the distances between observations and cluster centers.

# Keyword arguments
- `dist`: the distance metric used for computing the distances.
"""
function criticisms(c::AffinityPropResult, X::AbstractMatrix{<:Real}, n::Int=1; dist::Metric=Euclidean())
    D = pairwise(dist, X, dims=2)
    instances = Vector{Vector{Int}}()
    for i = 1:nclusters(c)
        v = view(D, assignments(c) .== i, c.exemplars[i])
        permutation = partialsortperm(v, 1:n, rev=true)
        parentinstances = parentindices(v)[1]
        push!(instances, parentinstances[permutation])
    end
    return instances
end

# Return a new instance of the method specified by the method's symbolic name
function _method(s::Symbol)
    s === :mmdcritic && return MMDCritic()
    s === :kmeans && return KMeans()
    s === :kmedoids && return KMedoids()
    s === :fuzzycmeans && return FuzzyCMeans()
    s === :affinitypropagation && return AffinityPropagation()
    throw(ArgumentError("Invalid method: $s"))
end

# Return instances based on a combination of data set and cluster assignments
function _instances(X::AbstractMatrix{<:Real}, ys::Vector{Int}, n::Int, ::Union{KMedoids, KMeans, AffinityPropagation}, rev::Bool=false)
    instances = Vector{Vector{Int}}()
    for i = 1:length(unique(ys))
        v = view(X, :, ys .== i)
        centroid = mean(v, dims=2)
        distances = vec(pairwise(Euclidean(), v, centroid, dims=2))
        permutation = partialsortperm(distances, 1:n, rev=rev)
        parentinstances = parentindices(v)[2]
        push!(instances, parentinstances[permutation])
    end
    return instances
end
function _instances(X::AbstractMatrix{<:Real}, W::Matrix{<:Real}, n::Int, ::FuzzyCMeans, rev::Bool=false)
    hardassignments = vec(map(i -> i[2], argmax(W, dims=2)))
    _instances(X, hardassignments, n, KMeans(), rev)
end

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
    hardassignments = vec(map(i -> i[2], argmax(W, dims=2)))
    for i = 1:size(W, 2)
        v = view(W, hardassignments .== i, i)
        permutation = partialsortperm(v, 1:n, rev=rev)
        parentinstances = parentindices(v)[1]
        push!(instances, parentinstances[permutation])
    end
    return instances
end

"""
    sqmmd(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel)
    mmd²(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel)

Compute the squared maximum mean discrepancy between `X` and `Y` using the kernel function `k`.

`X` and `Y` are expected to store observations in columns.
"""
function sqmmd(X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel)
    XX = kernelmatrix(k, X, obsdim=2)
    XY = kernelmatrix(k, X, Y, obsdim=2)
    YY = kernelmatrix(k, Y, obsdim=2)
    return sqmmd(XX, XY, YY)
end

"""
    sqmmd(XX::AbstractMatrix{<:Real}, XY::AbstractMatrix{<:Real}, YY::AbstractMatrix{<:Real})
    mmd²(XX::AbstractMatrix{<:Real}, XY::AbstractMatrix{<:Real}, YY::AbstractMatrix{<:Real})

Compute the squared maximum mean discrepancy using the kernel matrices `XX`, `XY`, and `YY`.

`XX` is the kernel matrix of the matrices `X` and `X` etc.
"""
sqmmd(XX::AbstractMatrix{<:Real}, XY::AbstractMatrix{<:Real}, YY::AbstractMatrix{<:Real}) = mean(XX) - 2 * mean(XY) + mean(YY)

# Alias
const mmd² = sqmmd

"""
    witness(z::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel)

Compute the witness function of `X` and `Y` at `z` using the kernel function `k`.

`X` and `Y` are expected to store observations in columns, and `z` is a single observation.
"""
witness(z::AbstractVector{<:Real}, X::AbstractMatrix{<:Real}, Y::AbstractMatrix{<:Real}, k::Kernel) = mean(map(x -> k(z, x), eachcol(X))) - mean(map(y -> k(z, y), eachcol(Y)))

end # module
