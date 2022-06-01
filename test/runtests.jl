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
    @testset verbose=true "prototypes" begin
        n = 50
        X = rand(5, n)
        k = RBFKernel()
        numclusters = 2

        @testset "MMD-critic" begin
            @test length(prototypes(X, k)) == 1
            @test length(prototypes(X, k, 0)) == 0
            @test Set(prototypes(X, k, n)) == Set(1:n)
            K = kernelmatrix(k, X, obsdim=2)
            @test length(prototypes(K)) == 1
            @test length(prototypes(K, 0)) == 0
            protoids = prototypes(K, n)
            @test Set(protoids) == Set(1:n)
            protoids2 = prototypes(X, ones(Int, n), k, n)
            @test Set(protoids2...) == Set(1:n)
            @test collect(protoids2...) == protoids
        end

        @testset "k-medoids" begin
            c = kmedoids(pairwise(Euclidean(), X, dims=2), numclusters)
            test(prototypes(c), numclusters, 1)
            test(prototypes(c, 2), numclusters, 2)
            test(prototypes(X, assignments(c), :kmedoids, 2), numclusters, 2)
        end

        @testset "k-means" begin
            c = kmeans(X, numclusters)
            test(prototypes(c), numclusters, 1)
            test(prototypes(c, 2), numclusters, 2)
            test(prototypes(X, assignments(c), :kmeans, 2), numclusters, 2)
        end

        @testset "fuzzy c-means" begin
            c = fuzzy_cmeans(X, numclusters, 2)
            test(prototypes(c), numclusters, 1)
            test(prototypes(c, 2), numclusters, 2)
            test(prototypes(X, c.weights, :fuzzycmeans, 2), numclusters, 2)
        end

        @testset "DBSCAN" begin
            c = dbscan(X, 0.01)
            test(prototypes(c), n, 1)
        end

        @testset "affinity propagation" begin
            S = -pairwise(Euclidean(), X, dims=2)
            S = S - diagm(0 => diag(S)) + median(S) * I
            c = affinityprop(S)
            numclusters = nclusters(c)
            test(prototypes(c, X), numclusters, 1)
            test(prototypes(c, X, 2), numclusters, 2)
            test(prototypes(X, assignments(c), :affinitypropagation, 2), numclusters, 2)
        end

        @test_throws ArgumentError prototypes(X, ones(Int, n), :somenotexistingmethod)
    end

    @testset verbose=true "criticisms" begin
        n = 50
        X = rand(5, n)
        k = RBFKernel()
        numclusters = 2

        @testset "MMD-critic" begin
            @test length(criticisms(X, k, [1, 2])) == 1
            @test length(criticisms(X, k, [1, 2], 0)) == 0
            @test Set(criticisms(X, k, [1, 2], n - 2)) == Set(3:n)
            K = kernelmatrix(k, X, obsdim=2)
            @test length(criticisms(K, [1, 2])) == 1
            @test length(criticisms(K, [1, 2], 0)) == 0
            critids = criticisms(K, [1, 2], n - 2)
            @test Set(critids) == Set(3:n)
            critids2 = criticisms(X, ones(Int, n), k, [[1, 2]], n - 2)
            @test Set(critids2...) == Set(3:n)
            @test collect(critids2...) == critids
        end

        @testset "k-medoids" begin
            c = kmedoids(pairwise(Euclidean(), X, dims=2), numclusters)
            test(criticisms(c), numclusters, 1)
            test(criticisms(c, 2), numclusters, 2)
            test(criticisms(X, assignments(c), :kmedoids, 2), numclusters, 2)
        end

        @testset "k-means" begin
            c = kmeans(X, numclusters)
            test(criticisms(c), numclusters, 1)
            test(criticisms(c, 2), numclusters, 2)
            test(criticisms(X, assignments(c), :kmeans, 2), numclusters, 2)
        end

        @testset "fuzzy c-means" begin
            c = fuzzy_cmeans(X, numclusters, 2)
            test(criticisms(c), numclusters, 1)
            test(criticisms(c, 2), numclusters, 2)
            test(criticisms(X, c.weights, :fuzzycmeans, 2), numclusters, 2)
        end

        @testset "affinity propagation" begin
            S = -pairwise(Euclidean(), X, dims=2)
            S = S - diagm(0 => diag(S)) + median(S) * I
            c = affinityprop(S)
            numclusters = nclusters(c)
            test(criticisms(c, X), numclusters, 1)
            test(criticisms(c, X, 2), numclusters, 2)
            test(criticisms(X, assignments(c), :affinitypropagation, 2), numclusters, 2)
        end

        @testset "DBSCAN" begin
            c = dbscan(X, 0.01)
            test(criticisms(c), n, 1)
        end

        @test_throws ArgumentError criticisms(X, ones(Int, n), :somenotexistingmethod)
    end

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
end
