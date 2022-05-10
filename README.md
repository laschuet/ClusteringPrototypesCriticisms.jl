# PrototypesCriticisms.jl

[![][action-img]][action-url]
[![][codecov-img]][codecov-url]

[action-img]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml/badge.svg?branch=main
[action-url]: https://github.com/laschuet/PrototypesCriticisms.jl/actions/workflows/CI.yml?query=branch%3Amain
[codecov-img]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl/branch/main/graph/badge.svg
[codecov-url]: https://codecov.io/gh/laschuet/PrototypesCriticisms.jl

A Julia package for computing prototypes and criticisms. In particular, we
support the selection of prototypes and criticisms with the following methods:

|                             | Prototypes | Criticisms |
|-----------------------------|-----------:|-----------:|
| [MMD-critic][mmdcritic-url] | ✓          | ✓          |
| k-medoids                   | ✓          | ✓          |
| k-means                     | ✓          | ✓          |
| fuzzy c-means               | ✓          | ✓          |
| affinity propagation        | ✓          | ✓          |

[mmdcritic-url]: https://dl.acm.org/doi/10.5555/3157096.3157352

## Installation

The package has not been registered yet.

```julia
] add https://github.com/laschuet/PrototypesCriticisms.jl
```
