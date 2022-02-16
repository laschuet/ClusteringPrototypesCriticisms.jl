using PrototypesCriticisms
using Test

@testset verbose=true "PrototypesCriticisms.jl" begin
    @testset "sqmmd (mmd²)" begin
        X = rand(10, 10)
        @test sqmmd(X, X) == mmd²(X, X)
        @test sqmmd(X, X) ≈ 0
    end
end
