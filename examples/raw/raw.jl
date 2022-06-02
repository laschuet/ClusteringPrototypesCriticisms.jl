using CairoMakie
using Clustering
using Distances
using Distributions
using KernelFunctions
using LinearAlgebra
using PrototypesCriticisms
using Random

"""
    plot(D, protoids, critids, axis)

Produce plot for the data, prototypes, and criticisms.
"""
function plot(D, protoids, critids, axis)
    protos = D[:, vcat(protoids...)]
    crits = D[:, vcat(critids...)]
    scatter!(axis, D[1, :], D[2, :], color=(:grey, 0.75), markersize=4)
    scatter!(axis, protos[1, :], protos[2, :], marker=:rect, markersize=8)
    scatter!(axis, crits[1, :], crits[2, :], marker=:xcross, markersize=12)
end

"""
    output(protoids, critids, title)

Print prototypes and criticisms.
"""
function output(protoids, critids, title)
    println("$title:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
end

"""
    main()

Run the example.
"""
function main()
    Random.seed!(42)
    update_theme!(font="Libertinus Serif")
    mkpath("out")

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
    axes = [
        Axis(fig[1, 1], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title="k-medoids"),
        Axis(fig[1, 2], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title="k-means"),
        Axis(fig[1, 3], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title="fuzzy c-means"),
        Axis(fig[2, 1], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title="affinity propagation"),
        Axis(fig[2, 2], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title="MMD-critic"),
    ]

    # k-medoids
    clustering = kmedoids(pairwise(Euclidean(), D), k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "k-medoids")
    plot(D, protoids, critids, axes[1])

    # k-means
    clustering = kmeans(D, k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "k-means")
    plot(D, protoids, critids, axes[2])

    # fuzzy c-means
    clustering = fuzzy_cmeans(D, k, 2)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "Fuzzy c-means")
    plot(D, protoids, critids, axes[3])

    # affinity propagation
    S = -pairwise(Euclidean(), D, dims=2)
    S = S - diagm(0 => diag(S)) + median(S) * I
    clustering = affinityprop(S)
    protoids = prototypes(clustering, D, p)
    critids = criticisms(clustering, D, c)
    output(protoids, critids, "Affinity propagation")
    plot(D, protoids, critids, axes[4])

    # MMD-critic
    kernel = with_lengthscale(RBFKernel(), sqrt(size(D, 1)))
    protoids = prototypes(D, kernel, k)
    critids = criticisms(D, kernel, protoids, k)
    output(protoids, critids, "MMD-critic")
    plot(D, protoids, critids, axes[5])

    save("out/raw.pdf", fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
