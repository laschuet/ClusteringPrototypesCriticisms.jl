using PrototypesCriticisms
using Clustering
using Distances
using KernelFunctions
using LinearAlgebra
using Statistics
using Test

function test(instanceids, k, n)
    @test typeof(instanceids) == Vector{Vector{Int}}
    @test length(instanceids) == k
    for i = 1:k
        @test length(instanceids[i]) == n
    end
end

@testset verbose=true "PrototypesCriticisms.jl" begin
    @testset "sqmmd (mmd²)" begin
        X = rand(5, 10)
        k = RBFKernel()

        @test sqmmd(X, X, k) ≈ 0

        Y = X
        XX = kernelmatrix(k, X, obsdim=2)
        XY = kernelmatrix(k, X, Y, obsdim=2)
        YY = kernelmatrix(k, Y, obsdim=2)
        @test sqmmd(XX, XY, YY) ≈ 0

        @test mmd²(X, X, k) == sqmmd(X, X, k)
        @test mmd²(XX, XY, YY) == sqmmd(XX, XY, YY)
    end

    @testset "witness" begin
        X = zeros(5, 10)
        Y = ones(5, 10)
        x = X[:, 1]
        y = Y[:, 1]
        k = RBFKernel()
        @test witness(x, X, X, k) ≈ 0
        @test witness(x, X, Y, k) > 0
        @test witness(y, X, Y, k) < 0
    end

    @testset "prototypes" begin
        n = 50
        X = rand(5, n)
        k = RBFKernel()
        numclusters = 2

        @test length(prototypes(X, k, 0)) == 0
        @test Set(prototypes(X, k, n)) == Set(1:n)

        K = kernelmatrix(k, X, obsdim=2)
        @test length(prototypes(K, 0)) == 0
        protoids = prototypes(K, n)
        @test Set(protoids) == Set(1:n)
        protoids2 = prototypes(X, ones(Int, n), n, k)
        @test Set(protoids2...) == Set(1:n)
        @test collect(protoids2...) == protoids

        c = kmedoids(pairwise(Euclidean(), X, dims=2), numclusters)
        test(prototypes(c), numclusters, 1)
        test(prototypes(c, 2), numclusters, 2)
        test(prototypes(X, assignments(c), 2, :kmedoids), numclusters, 2)

        c = kmeans(X, numclusters)
        test(prototypes(c), numclusters, 1)
        test(prototypes(c, 2), numclusters, 2)
        test(prototypes(X, assignments(c), 2, :kmeans), numclusters, 2)

        c = fuzzy_cmeans(X, numclusters, 2)
        test(prototypes(c), numclusters, 1)
        test(prototypes(c, 2), numclusters, 2)
        test(prototypes(X, c.weights, 2, :fuzzycmeans), numclusters, 2)

        S = -pairwise(Euclidean(), X, dims=2)
        S = S - diagm(0 => diag(S)) + median(S) * I
        c = affinityprop(S)
        numclusters = nclusters(c)
        test(prototypes(c, X), numclusters, 1)
        test(prototypes(c, X, 2), numclusters, 2)
        test(prototypes(X, assignments(c), 2, :affinitypropagation), numclusters, 2)
    end

    @testset "criticisms" begin
        n = 50
        X = rand(5, n)
        k = RBFKernel()
        numclusters = 2

        @test length(criticisms(X, k, [1, 2], 0)) == 0
        @test Set(criticisms(X, k, [1, 2], n - 2)) == Set(3:n)

        K = kernelmatrix(k, X, obsdim=2)
        @test length(criticisms(K, [1, 2], 0)) == 0
        @test Set(criticisms(K, [1, 2], n - 2)) == Set(3:n)

        c = kmedoids(pairwise(Euclidean(), X, dims=2), numclusters)
        test(criticisms(c), numclusters, 1)
        test(criticisms(c, 2), numclusters, 2)
        test(criticisms(X, assignments(c), 2, :kmedoids), numclusters, 2)

        c = kmeans(X, numclusters)
        test(criticisms(c), numclusters, 1)
        test(criticisms(c, 2), numclusters, 2)
        test(criticisms(X, assignments(c), 2, :kmeans), numclusters, 2)

        c = fuzzy_cmeans(X, numclusters, 2)
        test(criticisms(c), numclusters, 1)
        test(criticisms(c, 2), numclusters, 2)
        test(criticisms(X, c.weights, 2, :fuzzycmeans), numclusters, 2)

        S = -pairwise(Euclidean(), X, dims=2)
        S = S - diagm(0 => diag(S)) + median(S) * I
        c = affinityprop(S)
        numclusters = nclusters(c)
        test(criticisms(c, X), numclusters, 1)
        test(criticisms(c, X, 2), numclusters, 2)
        test(criticisms(X, assignments(c), 2, :affinitypropagation), numclusters, 2)
    end
end
