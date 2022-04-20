using PrototypesCriticisms
using Clustering
using Distances
using KernelFunctions
using Test

@testset verbose=true "PrototypesCriticisms.jl" begin
    @testset "sqmmd (mmd²)" begin
        X = rand(5, 10)
        @test sqmmd(X, X) == mmd²(X, X)
        @test sqmmd(X, X) ≈ 0
    end

    @testset "witness" begin
        X = zeros(5, 10)
        Y = ones(5, 10)
        x = X[:, 1]
        y = Y[:, 1]
        @test witness(x, X, X) ≈ 0
        @test witness(x, X, Y) > 0
        @test witness(y, X, Y) < 0
    end

    @testset "prototypes" begin
        n = 50
        X = rand(5, n)

        @test length(prototypes(X, 0)) == 0
        @test Set(prototypes(X, n)) == Set(1:n)

        K = kernelmatrix(RBFKernel(), X, obsdim=2)
        @test length(prototypes(0, K)) == 0
        @test Set(prototypes(n, K)) == Set(1:n)

        k = 2
        c = kmedoids(pairwise(Euclidean(), X, dims=2), k)
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

        k = 2
        c = kmeans(X, k)
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

        k = 2
        c = fuzzy_cmeans(X, k, 2)
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
    end

    @testset "criticisms" begin
        n = 50
        X = rand(5, n)

        @test length(criticisms(X, [1, 2], 0)) == 0
        @test Set(criticisms(X, [1, 2], n - 2)) == Set(3:n)

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
    end
end
