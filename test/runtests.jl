using Test
using MLJText

@testset "tfidf_transformer" begin
    include("tfidf_transformer.jl")
end

@testset "scitypes" begin
    include("scitypes.jl")
end
