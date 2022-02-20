module PrototypesCriticisms

using Clustering
using KernelFunctions
using Statistics

export criticisms,
        prototypes,
        sqmmd, mmd²,
        witness

"""
"""
sqmmd(X, Y, k=RBFKernel()) = mean(kernelmatrix(k, X, obsdim=2)) - 2 * mean(kernelmatrix(k, X, Y, obsdim=2)) + mean(kernelmatrix(k, Y, obsdim=2))
const mmd² = sqmmd

"""
"""
function prototypes(X, n, k=RBFKernel())
    protoids = []
    while length(protoids) < n
        fs = []
        for i in setdiff(1:size(X, 2), protoids)
            P2 = view(X, :, [protoids; i])
            P1 = view(X, :, protoids)
            if length(protoids) > 0
                f = sqmmd(X, P2, k) - sqmmd(X, P1, k)
            else
                f = sqmmd(X, P2, k)
            end
            push!(fs, (f, i))
        end
        push!(protoids, fs[argmin(fs)][2])
    end
    return protoids
end

"""
"""
prototypes(c::KmedoidsResult) = c.medoids

"""
"""
witness(z, X, Y, k=RBFKernel()) = mean(map(x -> k(z, x), eachcol(X))) - mean(map(y -> k(z, y), eachcol(Y)))

"""
"""
function criticisms(X, protoids, n, k=RBFKernel())
    critids = []
    while length(critids) < n
        absws = []
        for i in setdiff(1:size(X, 2), union(critids, protoids))
            w = abs(witness(view(X, :, i), X, view(X, :, protoids), k))
            push!(absws, (abs(w), i))
        end
        push!(critids, absws[argmax(absws)][2])
    end
    return critids
end

end # module
