using Clustering
using Distances
using FileIO
using KernelFunctions
using MLDatasets
using PrototypesCriticisms
using Random

"""
    saveasimage(d, ids; outdir="out")
    saveasimage(d; kwargs...)

Save the data instances `ids` of the dataset `d` as image files.

If `ids` is unspecified, all data instances of the dataset are saved as image files.
Assumes that each data instance actually represents an image.

# Keyword arguments
- `outdir`: the location where the image files are saved.
- `ext`: the file extension of the image files.

"""
function saveasimage(d, ids; outdir="out", ext="png")
    mkpath(outdir)
    for i in ids
        image = convert2image(d, i)
        classname = d.targets[i]
        if haskey(d.metadata, "class_names")
            classnames = d.metadata["class_names"]
            classname = classnames[d.targets[i] + 1]
        end
        save("$outdir/$("0" ^ (ndigits(length(d)) - ndigits(i)))$(i)_$classname.$ext", image)
    end
end
saveasimage(d; kwargs...) = saveasimage(d, 1:length(d); kwargs...)

"""
    printclusters(clusters; headline="", indent=2)

Print all instances of every cluster in `clusters`.

# Keyword arguments
- `headline`: the headline that is printed before the output of the clusters.
- `indent`: the number of spaces to indent the output of every cluster.
"""
function printclusters(clusters; headline="", indent=2)
    println(headline)
    for (i, instances) in enumerate(clusters)
        println("$(" " ^ indent)Cluster $(lpad(i, ndigits(clusters))): $instances")
    end
end

"""
    main()

Run the example.
"""
function main()
    Random.seed!(42)

    # Load dataset and the corresponding pre-trained image embeddings
    @info "Load data..."
    dataset = CIFAR10()
    embedding = load("cifar-10_train_embedding_vgg-19.jld2", "embedding")'

    # Save whole data set as images
    @info "Save images..."
    saveasimage(dataset, outdir="out/cifar-10")

    # Set main program parameters
    k = 10 # Number of clusters to compute
    p = 5 # Number of prototypes to find
    c = 5 # Number of criticisms to find

    # Cluster data using k-medoids
    @info "Cluster data..."
    clustering = kmedoids(pairwise(Euclidean(), embedding), k)

    # Evaluate the clustering
    @info "Evaluate clustering..."
    targets = dataset.targets .+ 1 # Cluster indices must start with 1 instead of 0
    ris = randindex(clustering, targets)
    println("Rand-related indices:")
    println("  Adjusted rand index: ", ris[1])
    println("  P(agree): ", ris[2])
    println("  P(disagree): ", ris[3])
    println("  P(agree) - P(disagree): ", ris[4])
    println("Variation of information: ", Clustering.varinfo(clustering, targets))
    println("V-measure: ", vmeasure(clustering, targets))
    println("Mutual information: ", mutualinfo(clustering, targets))

    # Naive prototypes and criticisms for the clustering
    @info "Compute naive prototypes and criticisms..."
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    printclusters(protoids, headline="Prototype ids:")
    printclusters(critids, headline="Criticism ids:")
    for i = 1:k
        saveasimage(dataset, protoids[i], outdir="out/cifar-10/_protos/naive/$i")
        saveasimage(dataset, critids[i], outdir="out/cifar-10/_crits/naive/$i")
    end

    # Prototypes and criticisms for the clustering using MMD-critic
    @info "Compute MMD-critic prototypes and criticisms..."
    ys = assignments(clustering)
    clusters = Vector{Vector{Int}}(undef, k)
    for i = 1:k
        clusters[i] = findall(ys .== i)
    end
    protoids = Vector{Vector{Int}}(undef, k)
    critids = Vector{Vector{Int}}(undef, k)
    for i = 1:k
        subembedding = view(embedding, :, clusters[i])
        kernel = with_lengthscale(RBFKernel(), sqrt(size(embedding, 1)))
        clusterpids = prototypes(subembedding, kernel, p)
        protoids[i] = clusters[i][clusterpids]
        clustercids = criticisms(subembedding, kernel, clusterpids, c)
        critids[i] = clusters[i][clustercids]
    end
    printclusters(protoids, headline="Prototype ids:")
    printclusters(critids, headline="Criticism ids:")
    for i = 1:k
        saveasimage(dataset, protoids[i], outdir="out/cifar-10/_protos/mmd-critic/$i")
        saveasimage(dataset, critids[i], outdir="out/cifar-10/_crits/mmd-critic/$i")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
