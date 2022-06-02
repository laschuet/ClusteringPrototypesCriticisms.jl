using CairoMakie
using Clustering
using DataFrames
using Distances
using Distributions
using KernelFunctions
using LinearAlgebra
using MLDatasets: Iris
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

    # Load data set
    @info "Load data..."
    D = Iris(as_df=false).features[1:2, :]

    # Set main program parameters
    k = 3 # Number of clusters to compute
    p = 1 # Number of prototypes to find
    c = 1 # Number of criticisms to find

    # Prepare final plot
    @info "Prepare plot..."
    fig = Figure()
    xlabel = "x"
    ylabel = "y"
    axes = [
        Axis(fig[1, 1], xgridvisible=false, ygridvisible=false, limits=(4, 8, 1, 5), xlabel=xlabel, ylabel=ylabel, title="k-medoids"),
        Axis(fig[1, 2], xgridvisible=false, ygridvisible=false, limits=(4, 8, 1, 5), xlabel=xlabel, ylabel=ylabel, title="k-means"),
        Axis(fig[1, 3], xgridvisible=false, ygridvisible=false, limits=(4, 8, 1, 5), xlabel=xlabel, ylabel=ylabel, title="fuzzy c-means"),
        Axis(fig[2, 1], xgridvisible=false, ygridvisible=false, limits=(4, 8, 1, 5), xlabel=xlabel, ylabel=ylabel, title="affinity propagation"),
        Axis(fig[2, 2], xgridvisible=false, ygridvisible=false, limits=(4, 8, 1, 5), xlabel=xlabel, ylabel=ylabel, title="MMD-critic"),
    ]

    # Run k-medoids method
    @info "Run k-medoids..."
    clustering = kmedoids(pairwise(Euclidean(), D), k)
    @info "Find prototypes and criticisms (k-medoids method)..."
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "k-medoids")
    plot(D, protoids, critids, axes[1])

    # Run k-means method
    @info "Run k-means..."
    clustering = kmeans(D, k)
    @info "Find prototypes and criticisms (k-means method)..."
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "k-means")
    plot(D, protoids, critids, axes[2])

    # Run fuzzy c-means method
    @info "Run fuzzy c-means..."
    clustering = fuzzy_cmeans(D, k, 2)
    @info "Find prototypes and criticisms (fuzzy c-means method)..."
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    output(protoids, critids, "Fuzzy c-means")
    plot(D, protoids, critids, axes[3])

    # Run affinity propagation method
    @info "Run affinity propagation..."
    S = -pairwise(Euclidean(), D, dims=2)
    S = S - diagm(0 => diag(S)) + median(S) * I
    clustering = affinityprop(S, damp=0.95)
    @info "Find prototypes and criticisms (affinity propagation method)..."
    protoids = prototypes(clustering, D, p)
    critids = criticisms(clustering, D, c)
    output(protoids, critids, "Affinity propagation")
    plot(D, protoids, critids, axes[4])

    # Run MMD-critic method
    @info "Run MMD-critic..."
    kernel = with_lengthscale(RBFKernel(), sqrt(size(D, 1)))
    @info "Find prototypes and criticisms (MMD-critic method)..."
    protoids = prototypes(D, kernel, k)
    critids = criticisms(D, kernel, protoids, k)
    output(protoids, critids, "MMD-critic")
    plot(D, protoids, critids, axes[5])

    save("out/iris.pdf", fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
