# Benchmark

Every method is benchmarked. This benchmark comes with its own Julia project
that contains a manifest. All package dependencies will be resolved during
instantiation.

## Usage

`cd` into `benchmark`, and directly run the benchmarks from the command line:
```
$ julia --project=. benchmarks.jl
```
