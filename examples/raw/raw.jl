using CairoMakie
using Clustering
using Distances
using Distributions
using KernelFunctions
using PrototypesCriticisms
using Random

function main()
    Random.seed!(42)

    n = 40
    D1 = [rand(Normal(1, 0.1), n) rand(Normal(1, 0.3), n)]
    D2 = [rand(Normal(3, 0.3), n) rand(Normal(3, 0.3), n)]
    D3 = [rand(Normal(4, 0.5), n) rand(Normal(2, 0.3), n)]
    D4 = [rand(Normal(1.5, 0.1), n) rand(Normal(3, 0.1), n)]
    D = [D1; D2; D3; D4]'

    k = 4
    p = 1
    c = 1

    fig = Figure()
    axes = [Axis(fig[i, j], limits=(0, 6, 0, 4), xlabel="x", ylabel="y") for i = 1:2, j = 1:2]

    # k-medoids
    clustering = kmedoids(pairwise(Euclidean(), D), k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    println("k-medoids:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
    protos = D[:, first.(protoids)]
    crits = D[:, first.(critids)]
    scatter!(axes[1], D[1, :], D[2, :])
    scatter!(axes[1], protos[1, :], protos[2, :])
    scatter!(axes[1], crits[1, :], crits[2, :])

    # k-means
    clustering = kmeans(D, k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    println("k-means:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
    protos = D[:, first.(protoids)]
    crits = D[:, first.(critids)]
    scatter!(axes[2], D[1, :], D[2, :])
    scatter!(axes[2], protos[1, :], protos[2, :])
    scatter!(axes[2], crits[1, :], crits[2, :])

    # fuzzy c-means
    clustering = fuzzy_cmeans(D, k, 2)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    println("fuzzy c-means:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
    protos = D[:, first.(protoids)]
    crits = D[:, first.(critids)]
    scatter!(axes[3], D[1, :], D[2, :])
    scatter!(axes[3], protos[1, :], protos[2, :])
    scatter!(axes[3], crits[1, :], crits[2, :])

    # MMD-critic
    kernel = with_lengthscale(RBFKernel(), sqrt(size(D, 1)))
    protoids = prototypes(D, kernel, k)
    critids = criticisms(D, kernel, protoids, k)
    println("MMD-critic:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
    protos = D[:, first.(protoids)]
    crits = D[:, first.(critids)]
    scatter!(axes[4], D[1, :], D[2, :])
    scatter!(axes[4], protos[1, :], protos[2, :])
    scatter!(axes[4], crits[1, :], crits[2, :])

    display(fig)
end
