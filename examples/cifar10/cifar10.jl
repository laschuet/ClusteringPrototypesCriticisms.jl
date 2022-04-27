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
        classnames = d.metadata["class_names"]
        classname = classnames[d.targets[i] + 1]
        save("$(outdir)/$("0" ^ (length(string(length(d))) - length(string(i))))$(i)_$(classname).$(ext)", image)
    end
end
saveasimage(d; kwargs...) = saveasimage(d, 1:length(d); kwargs...)

"""
    main()

Run the example.
"""
function main()
    Random.seed!(42)

    # Load dataset and the corresponding pre-trained image embeddings
    dataset = CIFAR10()
    embedding = load("cifar-10_train_embedding_vgg-19.jld2", "embedding")'

    # Save whole data set as images
    saveasimage(dataset, outdir="out/cifar-10")

    # Set main program parameters
    k = 10 # Number of clusters to compute
    p = 5 # Number of prototypes to find
    c = 5 # Number of criticisms to find

    # Cluster data using k-medoids
    clustering = kmedoids(pairwise(Euclidean(), embedding), k)

    # Naive prototypes and criticisms for the clustering
    protoids = prototypes(clustering, p)
    critids = criticisms(clustering, c)
    for i = 1:k
        saveasimage(dataset, protoids[i], outdir="out/cifar-10/_protos/naive/$i")
        saveasimage(dataset, critids[i], outdir="out/cifar-10/_crits/naive/$i")
    end

    # Prototypes and criticisms for the clustering using MMD-critic
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
    for i = 1:k
        saveasimage(dataset, protoids[i], outdir="out/cifar-10/_protos/mmd-critic/$i")
        saveasimage(dataset, critids[i], outdir="out/cifar-10/_crits/mmd-critic/$i")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
