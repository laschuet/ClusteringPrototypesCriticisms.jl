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
- `useposprefix`: prepend the position of the data instances in `ids` to the name of the image file.

"""
function saveasimage(d, ids; outdir="out", ext="png", useposprefix=true)
    mkpath(outdir)
    for (i, id) in enumerate(ids)
        image = convert2image(d, i)
        classname = d.targets[i]
        if haskey(d.metadata, "class_names")
            classnames = d.metadata["class_names"]
            classname = classnames[d.targets[i] + 1]
        end
        prefix = useposprefix ? "$(i)_" : ""
        save("$outdir/$prefix$("0" ^ (ndigits(length(d)) - ndigits(id)))$(id)_$classname.$ext", image)
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
        println("$(" " ^ indent)Cluster $(lpad(i, ndigits(length(clusters)))): $instances")
    end
end

"""
    evaluate(c, ys)

Evaluate the clustering `c` using the labels `ys`.
"""
function evaluate(c, ys)
    ris = randindex(c, ys)
    println("Rand-related indices:")
    println("  Adjusted rand index: ", ris[1])
    println("  P(agree): ", ris[2])
    println("  P(disagree): ", ris[3])
    println("  P(agree) - P(disagree): ", ris[4])
    println("Variation of information: ", Clustering.varinfo(c, ys))
    println("V-measure: ", vmeasure(c, ys))
    println("Mutual information: ", mutualinfo(c, ys))
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
    saveasimage(dataset, outdir="out/raw", useposprefix=false)

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
    evaluate(clustering, targets)

    # Naive prototypes and criticisms for the clustering
    @info "Compute naive prototypes and criticisms..."
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    printclusters(protoids, headline="Prototype ids:")
    printclusters(critids, headline="Criticism ids:")
    for i = 1:k
        saveasimage(dataset, protoids[i], outdir="out/prototypes/naive/$i")
        saveasimage(dataset, critids[i], outdir="out/criticisms/naive/$i")
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
        saveasimage(dataset, protoids[i], outdir="out/prototypes/mmd-critic/$i")
        saveasimage(dataset, critids[i], outdir="out/criticisms/mmd-critic/$i")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
