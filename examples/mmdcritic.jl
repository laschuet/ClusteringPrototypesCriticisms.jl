using PrototypesCriticisms
using CairoMakie
using Distributions
using Random

function main()
    Random.seed!(42)

    n = 40
    D1 = [rand(Normal(1, 0.1), n) rand(Normal(1, 0.3), n)]
    D2 = [rand(Normal(3, 0.3), n) rand(Normal(3, 0.3), n)]
    D3 = [rand(Normal(4, 0.5), n) rand(Normal(2, 0.3), n)]
    D4 = [rand(Normal(1.5, 0.1), n) rand(Normal(3, 0.1), n)]
    D = [D1; D2; D3; D4]

    p = 4
    protoids = prototypes(D', p)
    protos = D[protoids, :]
    mmd2 = mmd²(D', protos')

    c = 8
    critids = criticisms(D', protoids, c)
    crits = D[critids, :]

    fig = Figure()
    title = "Data Prototypes (p=$p, mmd²=$(round(mmd2, digits=4))) and Criticisms (c=$c)"
    Axis(fig[1, 1], limits=(0, 6, 0, 4), xlabel="x", ylabel="y", title=title)
    scatter!(D[:, 1], D[:, 2])
    scatter!(protos[:, 1], protos[:, 2])
    scatter!(crits[:, 1], crits[:, 2])
    display(fig)
    save("mmdcritic.pdf", fig)
end
