# Examples

Every example comes with its own [Julia
project](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project)
that contains a manifest.

## Usage

`cd` into `examples/<example>`, start the Julia REPL by typing and confirming `julia`, and instantiate the environment:
```julia
julia> using Pkg; Pkg.activate("."); Pkg.instantiate()
```
Then run the example with `include("<example>.jl")`.

Or `cd` into `examples/<example>`, and directly run the example from the command line:
```
$ julia --project=. <example>.jl
```

## List

- [cifar-10](cifar10): Explaining the clustering of the CIFAR-10 data set with prototypes and criticisms.
- [raw](raw): Explaining the raw Iris flower data set.