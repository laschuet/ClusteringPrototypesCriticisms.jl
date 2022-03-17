using PrototypesCriticisms
using BenchmarkTools
using Clustering
using Distances
using KernelFunctions
using Random

function naiveprototypes(X::AbstractMatrix{<:Real}, n::Int, k::Kernel=RBFKernel())
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

function naivecriticisms(X::AbstractMatrix{<:Real}, protoids::AbstractVector{Int}, n::Int, k::Kernel=RBFKernel())
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

function main()
    Random.seed!(42)

    n = 200
    p = 5
    c = 5
    k = 3
    D = rand(20, n)

    # MMD-critic
    println("Naive prototypes implementation (MMD-critic)")
    @btime naiveprototypes($D, $p)
    println("Prototypes implementation (MMD-critic)")
    @btime prototypes($D, $p)
    protoids = rand(1:n, p)
    println("Naive criticisms implementation (MMD-critic)")
    @btime naivecriticisms($D, $protoids, $c)
    println("Criticisms implementation (MMD-critic)")
    @btime criticisms($D, $protoids, $c)

    # k-medoids
    c = kmedoids(pairwise(Euclidean(), D), k)
    println("Prototypes implementation (k-medoids)")
    @btime prototypes($c)

    # k-means
    c = kmeans(D, k)
    println("Prototypes implementation (k-means)")
    @btime prototypes($c)
    println("Criticisms implementation (k-means)")
    @btime criticisms($c)
end
