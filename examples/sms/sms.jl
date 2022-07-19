using ClusteringPrototypesCriticisms
using Embeddings
using FileIO
using KernelFunctions
using Languages
using MLDatasets: SMSSpamCollection
using Random
using TextAnalysis

"""
    preprocess(txt::AbstractString, lang::Language; stem=true)
    preprocess(txt::AbstractString; kwargs...)

Apply a sequence of text pre-processing steps to the input text `txt` that is supposed to be
written in the language `lang`. The language of the text will be determined automatically,
when `lang` is not specified. Non-alphanumeric characters are removed, and letters are
converted to lowercase.

# Keyword arguments
- `stem`: whether to additionally stem the tokens or not.
"""
function preprocess(txt::AbstractString, lang::Language; dostem=true)
    # We replace by " " to handle cases such as "end of sentence.begin of next sentence"
    txt = replace(txt, r"[.:,;?!()\[\]\\=*/+-<>@]" => " ")
    txt = lowercase(txt)
    doc = StringDocument(txt)
    prepare!(doc, strip_stopwords)
    txt = text(doc)
    tokens = tokenize(lang, txt)
    if dostem
        stemmer = Stemmer(isocode(lang))
        tokens = stem(stemmer, tokens)
    end
    return join(tokens, " ")
end
preprocess(txt::AbstractString; kwargs...) = preprocess(txt, LanguageDetector()(txt)[1]; kwargs...)

"""
    avgembeddings(txts::Vector{<:AbstractString}, embeddingtable::Embeddings.EmbeddingTable, wordindextable::AbstractDict)

Return the embeddings of the input texts `txts`. This is the average of the word embeddings for every text.
"""
function avgembeddings(txts::Vector{<:AbstractString}, embeddingtable::Embeddings.EmbeddingTable, wordindextable::AbstractDict)
    embeddingsize = size(embeddingtable.embeddings, 1)
    embeddings = Matrix{Float64}(undef, embeddingsize, length(txts))
    for (index, txt) in enumerate(txts)
        words = split(txt, " ")
        txtembedding = zeros(embeddingsize)
        numunknownwords = 0
        for word in words
            if !haskey(wordindextable, word)
                numunknownwords += 1
                continue
            end
            wordindex = wordindextable[word]
            embedding = embeddingtable.embeddings[:, wordindex]
            txtembedding += embedding
        end
        numknownwords = length(words) - numunknownwords
        if numknownwords > 0
            txtembedding /= numknownwords
        end
        embeddings[:, index] = txtembedding
    end
    return embeddings
end

"""
    mdtable(filename, title, protos, crits; mode="w")

Generate and save Markdown table with the prototypes `protos` and criticisms `crits`.
"""
function mdtable(filename, title, protos, crits; mode="w")
    open(filename, mode) do io
        write(io, "# $title\n\n")
        write(io, "| Prototypes |\n")
        write(io, "| :--------- |\n")
        for p in protos
            write(io, "| $(p[1:min(128, length(p))])... |\n")
        end
        write(io, "\n")
        write(io, "| Criticisms |\n")
        write(io, "| :--------- |\n")
        for c in crits
            write(io, "| $(c[1:min(128, length(c))])... |\n")
        end
    end
end

"""
    main(; genmd=false)

Run the example.

# Keyword arguments
- `genmd`: whether to generate and save the example output as a Markdown table or not.
"""
function main(; genmd=false)
    Random.seed!(42)
    mkpath("out")

    # Load data set
    @info "Load data..."
    dataset = SMSSpamCollection()
    D = dataset.features
    ys = dataset.targets
    spam = (ys .== "spam")
    nospam = (ys .== "ham")

    # Pre-process data set
    @info "Pre-process data..."
    X = preprocess.(D, [Languages.English()])
    embeddingsfile = "out/embeddings.jld2"
    if isfile(embeddingsfile)
        @info "Load embeddings..."
        E = load(embeddingsfile, "embeddings")
    else
        @info "Compute embeddings...This might take some time..."
        embeddingtable = load_embeddings(FastText_Text{:en})
        wordindextable = Dict(word => index for (index, word) in enumerate(embeddingtable.vocab))
        E = avgembeddings(X, embeddingtable, wordindextable)
        save(embeddingsfile, Dict("embeddings" => E))
    end

    # Set main program parameters
    p = 5 # Number of prototypes to find
    c = 5 # Number of criticisms to find

    # Spam
    @info "Find prototypes and criticisms for spam SMS (MMD-critic method)..."
    E1 = E[:, spam]
    kernel = with_lengthscale(RBFKernel(), sqrt(size(E1, 1)))
    protoids = prototypes(E1, kernel, p)
    critids = criticisms(E1, kernel, protoids, c)
    protos = (D[spam][protoids])
    crits = (D[spam][critids])
    genmd && mdtable("sms.md", "Spam", protos, crits)
   
    # No spam
    @info "Find prototypes and criticisms for no spam SMS (MMD-critic method)..."
    E2 = E[:, nospam]
    kernel = with_lengthscale(RBFKernel(), sqrt(size(E2, 1)))
    protoids = prototypes(E2, kernel, p)
    critids = criticisms(E2, kernel, protoids, c)
    protos = (D[nospam][protoids])
    crits = (D[nospam][critids])
    genmd && mdtable("sms.md", "No spam", protos, crits, mode="a")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
    # main(genmd=true)
end
