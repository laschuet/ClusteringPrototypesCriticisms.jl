using PrototypesCriticisms
using Clustering
using Distances
using Test

@testset verbose=true "PrototypesCriticisms.jl" begin
    @testset "sqmmd (mmd²)" begin
        X = rand(5, 10)
        @test sqmmd(X, X) == mmd²(X, X)
        @test sqmmd(X, X) ≈ 0
    end

    @testset "witness" begin
        X = rand(5, 10)
        Y = ones(5, 10)
        x = X[:, 1]
        y = Y[:, 1]
        @test witness(x, X, X) ≈ 0
        @test witness(x, X, Y) > 0
        @test witness(y, X, Y) < 0
    end

    @testset "prototypes" begin
        X = rand(5, 10)

        @test length(prototypes(X, 0)) == 0
        @test Set(prototypes(X, 10)) == Set(1:10)

        k = 2
        c = kmedoids(pairwise(Euclidean(), X, dims=2), k)
        protoids = prototypes(c)
        @test length(protoids) == k
        @test protoids == c.medoids
    end

    @testset "criticisms" begin
        X = rand(5, 10)
        @test length(criticisms(X, [1, 2], 0)) == 0
        @test Set(criticisms(X, [1, 2], 8)) == Set(3:10)
    end
end
