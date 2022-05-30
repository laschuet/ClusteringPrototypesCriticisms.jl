using PrototypesCriticisms
using Clustering
using Distances
using KernelFunctions
using LinearAlgebra
using Statistics
using Test

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
        function test(X, c, k, s)
            protoids = prototypes(c)
            @test typeof(protoids) == Vector{Vector{Int}}
            @test length(protoids) == k
            for i = 1:k
                @test length(protoids[i]) == 1
            end
            protoids = prototypes(c, 2)
            @test typeof(protoids) == Vector{Vector{Int}}
            @test length(protoids) == k
            for i = 1:k
                @test length(protoids[i]) == 2
            end
            protoids2 = prototypes(X, assignments(c), 2, s)
            @test typeof(protoids2) == Vector{Vector{Int}}
            @test length(protoids2) == k
            for i = 1:k
                @test length(protoids2[i]) == 2
            end
            @test protoids2 == protoids
        end

        n = 50
        X = rand(5, n)
        k = RBFKernel()

        @test length(prototypes(X, k, 0)) == 0
        @test Set(prototypes(X, k, n)) == Set(1:n)

        K = kernelmatrix(k, X, obsdim=2)
        @test length(prototypes(K, 0)) == 0
        protoids = prototypes(K, n)
        @test Set(protoids) == Set(1:n)
        protoids2 = prototypes(X, ones(Int, n), n, k)
        @test Set(protoids2...) == Set(1:n)
        @test collect(protoids2...) == protoids

        k = 2
        c = kmedoids(pairwise(Euclidean(), X, dims=2), k)
        test(X, c, k, :kmedoids)

        k = 2
        c = kmeans(X, k)
        test(X, c, k, :kmeans)

        k = 2
        c = fuzzy_cmeans(X, k, 2)
        test(X, c, k, :fuzzycmeans)

        S = -pairwise(Euclidean(), X, dims=2)
        S = S - diagm(0 => diag(S)) + median(S) * I
        c = affinityprop(S)
        protoids = prototypes(c, X)
        @test typeof(protoids) == Vector{Vector{Int}}
        @test length(protoids) == nclusters(c)
        for i = 1:nclusters(c)
            @test length(protoids[i]) == 1
        end
        protoids = prototypes(c, X, 2)
        @test typeof(protoids) == Vector{Vector{Int}}
        @test length(protoids) == nclusters(c)
        for i = 1:nclusters(c)
            @test length(protoids[i]) == 2
        end
    end

    @testset "criticisms" begin
        n = 50
        X = rand(5, n)
        k = RBFKernel()

        @test length(criticisms(X, k, [1, 2], 0)) == 0
        @test Set(criticisms(X, k, [1, 2], n - 2)) == Set(3:n)

        K = kernelmatrix(k, X, obsdim=2)
        @test length(criticisms(K, [1, 2], 0)) == 0
        @test Set(criticisms(K, [1, 2], n - 2)) == Set(3:n)

        k = 2
        c = kmedoids(pairwise(Euclidean(), X, dims=2), k)
        critids = criticisms(c)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 1
        end
        critids = criticisms(c, 2)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 2
        end

        k = 2
        c = kmeans(X, k)
        critids = criticisms(c)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 1
        end
        critids = criticisms(c, 2)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 2
        end
        critids2 = criticisms(X, assignments(c), 2, :kmeans)
        @test typeof(critids2) == Vector{Vector{Int}}
        @test length(critids2) == k
        for i = 1:k
            @test length(critids2[i]) == 2
        end
        @test critids2 == critids

        k = 2
        c = fuzzy_cmeans(X, k, 2)
        critids = criticisms(c)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 1
        end
        critids = criticisms(c, 2)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == k
        for i = 1:k
            @test length(critids[i]) == 2
        end

        S = -pairwise(Euclidean(), X, dims=2)
        S = S - diagm(0 => diag(S)) + median(S) * I
        c = affinityprop(S)
        critids = criticisms(c, X)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == nclusters(c)
        for i = 1:nclusters(c)
            @test length(critids[i]) == 1
        end
        critids = prototypes(c, X, 2)
        @test typeof(critids) == Vector{Vector{Int}}
        @test length(critids) == nclusters(c)
        for i = 1:nclusters(c)
            @test length(critids[i]) == 2
        end
    end
end
