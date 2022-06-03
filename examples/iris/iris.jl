using CairoMakie
using ClusteringPrototypesCriticisms
using DataFrames
using KernelFunctions
using MLDatasets: Iris
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
    ys = repeat(1:3, inner=50)
    Y = zeros(size(D, 2), 3)
    for i = 1:3
        Y[(1 + 50 * (i - 1)):(50 * i), i] .= 1
    end

    # Set main program parameters
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
    @info "Find prototypes and criticisms (k-medoids method)..."
    protoids = prototypes(D, ys, :kmedoids, p)
    critids = criticisms(D, ys, :kmedoids, c)
    output(protoids, critids, "k-medoids")
    plot(D, protoids, critids, axes[1])

    # Run k-means method
    @info "Run k-means..."
    @info "Find prototypes and criticisms (k-means method)..."
    protoids = prototypes(D, ys, :kmeans, p)
    critids = criticisms(D, ys, :kmeans, c)
    output(protoids, critids, "k-means")
    plot(D, protoids, critids, axes[2])

    # Run fuzzy c-means method
    @info "Run fuzzy c-means..."
    @info "Find prototypes and criticisms (fuzzy c-means method)..."
    protoids = prototypes(D, Y, :fuzzycmeans, p)
    critids = criticisms(D, Y, :fuzzycmeans, c)
    output(protoids, critids, "Fuzzy c-means")
    plot(D, protoids, critids, axes[3])

    # Run affinity propagation method
    @info "Run affinity propagation..."
    @info "Find prototypes and criticisms (affinity propagation method)..."
    protoids = prototypes(D, ys, :affinitypropagation, p)
    critids = criticisms(D, ys, :affinitypropagation, c)
    output(protoids, critids, "Affinity propagation")
    plot(D, protoids, critids, axes[4])

    # Run MMD-critic method
    @info "Run MMD-critic..."
    kernel = with_lengthscale(RBFKernel(), sqrt(size(D, 1)))
    @info "Find prototypes and criticisms (MMD-critic method)..."
    protoids, mappedprotoids = prototypes(D, ys, kernel, p)
    critids = criticisms(D, ys, kernel, mappedprotoids, c)
    output(protoids, critids, "MMD-critic")
    plot(D, protoids, critids, axes[5])

    save("out/iris.pdf", fig)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
