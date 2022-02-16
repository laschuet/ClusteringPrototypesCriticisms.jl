using PrototypesCriticisms
using Test

@testset verbose=true "PrototypesCriticisms.jl" begin
    @testset "sqmmd (mmd²)" begin
        X = rand(10, 10)
        @test sqmmd(X, X) == mmd²(X, X)
        @test sqmmd(X, X) ≈ 0
    end

    @testset "witness" begin
        X = rand(10, 10)
        Y = ones(10, 10)
        x = X[:, 1]
        y = Y[:, 1]
        @test witness(x, X, X) ≈ 0
        @test witness(x, X, Y) > 0
        @test witness(y, X, Y) < 0
    end

    @testset "prototypes" begin
        X = [1 2 4 5; 1 1 1 1]
        @test length(prototypes(X, 0)) == 0
    end
end
