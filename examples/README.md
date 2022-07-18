# Examples

[julia-project-url]: https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project

Every example comes with its own [Julia project][julia-project-url] that
contains a manifest. All package dependencies will be resolved during
instantiation.

## Usage

`cd` into `examples/<example>`, and directly run the example from the command line:
```
$ julia --project=. <example>.jl
```

## List

- [cifar-10](cifar10): Explaining the clustering of the CIFAR-10 data set with prototypes and criticisms.
- [iris](iris): Explaining the clustering of the Iris flower data set with prototypes and criticisms.
- [sms](sms): Explaining the SMS spam collection data set with prototypes and criticisms.
