using Test
using MLJText

@testset "tfidf_transformer" begin
    include("tfidf_transformer.jl")
end

@testset "bm25_transformer" begin
    include("bm25_transformer.jl")
end

@testset "bagofwords_transformer" begin
    include("bagofwords_transformer.jl")
end

@testset "scitypes" begin
    include("scitypes.jl")
end
