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
    protoids = []
    while length(protoids) < n
        fs = []
        for i in setdiff(1:size(X, 2), protoids)
            P2 = view(X, :, [protoids; i])
            P1 = view(X, :, protoids)
            if length(protoids) > 0
                f = sqmmd(X, P2, k) - sqmmd(X, P1, k)
            else
                f = sqmmd(X, P2, k)
            end
            push!(fs, (f, i))
        end
        push!(protoids, fs[argmin(fs)][2])
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
    critids = []
    while length(critids) < n
        absws = []
        for i in setdiff(1:size(X, 2), union(critids, protoids))
            w = abs(witness(view(X, :, i), X, view(X, :, protoids), k))
            push!(absws, (abs(w), i))
        end
        push!(critids, absws[argmax(absws)][2])
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
