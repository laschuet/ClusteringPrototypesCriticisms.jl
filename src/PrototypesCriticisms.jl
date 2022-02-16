module PrototypesCriticisms

using KernelFunctions
using Statistics

export sqmmd, mmd²,
        witness

"""
"""
sqmmd(X, Y, k=RBFKernel()) = mean(kernelmatrix(k, X, obsdim=2)) - 2 * mean(kernelmatrix(k, X, Y, obsdim=2)) + mean(kernelmatrix(k, Y, obsdim=2))
const mmd² = sqmmd

"""
"""
witness(z, X, Y, k=RBFKernel()) = mean(map(x -> k(z, x), eachcol(X))) - mean(map(y -> k(z, y), eachcol(Y)))

end # module
