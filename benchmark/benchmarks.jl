using PrototypesCriticisms
using BenchmarkTools
using Clustering
using Distances
using KernelFunctions
using LinearAlgebra
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
    p = 3
    c = 3
    k = 3
    D = rand(20, n)
 
    # k-medoids
    clustering = kmedoids(pairwise(Euclidean(), D), k)
    println("Prototypes implementation (k-medoids)")
    @btime prototypes($clustering, $p)
    println("Criticisms implementation (k-medoids)")
    @btime criticisms($clustering, $c)

    # k-means
    clustering = kmeans(D, k)
    println("Prototypes implementation (k-means)")
    @btime prototypes($clustering, $p)
    println("Criticisms implementation (k-means)")
    @btime criticisms($clustering, $c)

    # fuzzy c-means
    clustering = fuzzy_cmeans(D, k, 2)
    println("Prototypes implementation (fuzzy c-means)")
    @btime prototypes($clustering, $p)
    println("Criticisms implementation (fuzzy c-means)")
    @btime criticisms($clustering, $c)

    # affinity propagation
    S = -pairwise(Euclidean(), D, dims=2)
    S = S - diagm(0 => diag(S)) + median(S) * I
    clustering = affinityprop(S)
    println("Prototypes implementation (affinity propagation)")
    @btime prototypes($clustering, $D, $p)
    println("Criticisms implementation (affinity propagation)")
    @btime criticisms($clustering, $D, $c)

    # MMD-critic
    kernel = with_lengthscale(RBFKernel(), sqrt(20))
    println("Naive prototypes implementation (MMD-critic)")
    @btime naiveprototypes($D, $p, $kernel)
    println("Prototypes implementation (MMD-critic)")
    @btime prototypes($D, $kernel, $p)
    protoids = rand(1:n, p)
    println("Naive criticisms implementation (MMD-critic)")
    @btime naivecriticisms($D, $protoids, $c, $kernel)
    println("Criticisms implementation (MMD-critic)")
    @btime criticisms($D, $kernel, $protoids, $c)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
