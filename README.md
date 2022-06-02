# PrototypesCriticisms.jl

[![][action-img]][action-url]
[![][codecov-img]][codecov-url]

[action-img]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml?query=branch%3Amain
[codecov-img]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl

A Julia package for computing prototypes and criticisms.

Currently, there is a focus on finding prototypes and criticisms in clusterings.
Nonetheless, this package can be used to find prototypes and criticisms in raw
data sets. In this case, the clustering of the raw data set is an intermediate
step that is hidden from the user. However, there are also other methods
implemented that do not cluster the raw data set at all.

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

[mmdcritic-url]: https://dl.acm.org/doi/10.5555/3157096.3157352
[clustering.jl-url]: https://github.com/JuliaStats/Clustering.jl

## Installation

We assume that you have a working [installation of Julia][juliainstall-url].
This package is tested on macOS and Ubuntu (x64 platform, Julia 1.7 and nightly)
by [CI][actionsci-url] processes.

The package has not been registered yet. Install the package:
```julia
julia>]
pkg> add https://github.com/laschuet/PrototypesCriticisms.jl
```

Start using the package:
```julia
using PrototypesCriticisms
```

[actionsci-url]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml
[juliainstall-url]: https://julialang.org/downloads

## Quick start

[juliarepl-url]: https://docs.julialang.org/en/v1/stdlib/REPL/

You can easily read the API documentation by opening a [Julia
REPL][juliarepl-url] session and the following statements:
```julia
julia>using PrototypesCriticisms
julia>?
help?>prototypes
help?>criticisms
help?>sqmmd
help?>witness
```

Refer to the [examples](examples).
