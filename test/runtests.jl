using Test
using MLJText

@testset "abstract text transformer" begin
    include("abstract_text_transformer.jl")
end

@testset "scitypes" begin
    include("scitypes.jl")
end
