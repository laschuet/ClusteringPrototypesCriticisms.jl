using CairoMakie
using Clustering
using Distances
using Distributions
using KernelFunctions
using LinearAlgebra
using PrototypesCriticisms
using Random

"""
    output(D, protoids, critids, axis, title; color=nothing)

Print basic text output, and produce plot for the data, prototypes, and criticisms.
"""
function output(D, protoids, critids, axis, title; color=nothing)
    println("$title:")
    println("  Prototypes: ", protoids)
    println("  Criticisms: ", critids)
    protos = D[:, first.(protoids)]
    crits = D[:, first.(critids)]
    axis.title = title
    if isnothing(color)
        scatter!(axis, D[1, :], D[2, :], markersize=4)
    else
        scatter!(axis, D[1, :], D[2, :], color=color, markersize=4)
    end
    scatter!(axis, protos[1, :], protos[2, :], color=:red, marker=:cross)
    scatter!(axis, crits[1, :], crits[2, :], color=:green, marker=:xcross)
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
    axes = [Axis(fig[i, j], xgridvisible=false, ygridvisible=false, limits=(0, 6, 0, 4), xlabel=L"x", ylabel=L"y") for i = 1:2, j = 1:3]

    # k-medoids
    clustering = kmedoids(pairwise(Euclidean(), D), k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(D, protoids, critids, axes[1], "k-medoids", color=assignments(clustering))

    # k-means
    clustering = kmeans(D, k)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(D, protoids, critids, axes[2], "k-means", color=assignments(clustering))

    # fuzzy c-means
    clustering = fuzzy_cmeans(D, k, 2)
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(D, protoids, critids, axes[3], "fuzzy c-means", color=map(i -> i[2], argmax(clustering.weights, dims=2)))

    # affinity propagation
    S = -pairwise(Euclidean(), D)
    S = S - diagm(0 => diag(S)) + median(S) * I
    clustering = affinityprop(S)
    println("affinity propagation:")
    axes[4].title = "affinity propagation"
    scatter!(axes[4], D[1, :], D[2, :], color=assignments(clustering), markersize=5)

    # MMD-critic
    kernel = with_lengthscale(RBFKernel(), sqrt(size(D, 1)))
    protoids = prototypes(D, kernel, k)
    critids = criticisms(D, kernel, protoids, k)
    output(D, protoids, critids, axes[5], "MMD-critic", color=:grey)

    save("out/raw.pdf", fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
