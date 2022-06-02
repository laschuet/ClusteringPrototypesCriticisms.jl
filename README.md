# PrototypesCriticisms.jl

[action-img]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml?query=branch%3Amain
[action-ci-url]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml
[clustering.jl-url]: https://github.com/JuliaStats/Clustering.jl
[codecov-img]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl
[julia-install-url]: https://julialang.org/downloads
[julia-repl-url]: https://docs.julialang.org/en/v1/stdlib/REPL
[mmdcritic-url]: https://dl.acm.org/doi/10.5555/3157096.3157352

[![][action-img]][action-url]
[![][codecov-img]][codecov-url]

A Julia package for computing prototypes and criticisms.

Currently, there is a focus on finding prototypes and criticisms in clusterings.
Nonetheless, this package can also be used to find prototypes and criticisms in
data sets. In this case, the data set needs to be clustered first. However,
there are also other methods implemented that work without clustering at all.

In particular, we currently support the selection of prototypes and criticisms
with the following methods:

|                             | Prototypes | Criticisms |
|-----------------------------|-----------:|-----------:|
| [MMD-critic][mmdcritic-url] | ✓          | ✓          |
| k-medoids                   | ✓          | ✓          |
| k-means                     | ✓          | ✓          |
| fuzzy c-means               | ✓          | ✓          |
| affinity propagation        | ✓          | ✓          |
| DBSCAN                      | ✓          | ✓          |

In order to compute prototypes and criticisms for clusterings, we provide two approaches:
1. Directly use a clustering result computed with the
    [Clustering.jl][clustering.jl-url] package.
2. Provide the raw data set, the clustering assignments of the data instances,
    and the kind of method to be used for computing the prototypes and criticisms.

In order to compute prototypes and criticisms for raw data sets, provide the raw
data set and the kind of method to be used for computing the prototypes and criticisms.

## Installation

We assume that you have a working [installation of Julia][julia-install-url].
This package is tested on macOS and Ubuntu (x64 platform, Julia 1.7 and nightly)
by [CI][action-ci-url] processes.

Open a [Julia REPL][julia-repl-url] and install the package:
```julia
julia>]
pkg>add https://github.com/laschuet/PrototypesCriticisms.jl
```

## Getting started

Open a [Julia REPL][julia-repl-url] and start using the package:
```julia
julia>using PrototypesCriticisms
```

Then you can easily read the API documentation:
```julia
julia>?
help?>prototypes
help?>criticisms
help?>sqmmd
help?>witness
```
Alternatively, we invite you to take a quick look at the [source code](src).

## Examples

We also provide some usage [examples](examples).
