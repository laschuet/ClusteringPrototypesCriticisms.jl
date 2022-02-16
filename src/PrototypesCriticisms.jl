module PrototypesCriticisms

using KernelFunctions
using Statistics

export prototypes,
        sqmmd, mmd²,
        witness

"""
"""
sqmmd(X, Y, k=RBFKernel()) = mean(kernelmatrix(k, X, obsdim=2)) - 2 * mean(kernelmatrix(k, X, Y, obsdim=2)) + mean(kernelmatrix(k, Y, obsdim=2))
const mmd² = sqmmd

"""
"""
function prototypes(X, n)
    protoids = []
    while length(protoids) < n
        fs = []
        for i in setdiff(1:size(X, 2), protoids)
            P2 = view(X, :, [protoids; i])
            P1 = view(X, :, protoids)
            if length(protoids) > 0
                f = sqmmd(X, P2) - sqmmd(X, P1)
            else
                f = sqmmd(X, P2)
            end
            push!(fs, (f, i))
        end
        push!(protoids, fs[argmin(fs)][2])
    end
    return protoids
end

"""
"""
witness(z, X, Y, k=RBFKernel()) = mean(map(x -> k(z, x), eachcol(X))) - mean(map(y -> k(z, y), eachcol(Y)))

end # module
