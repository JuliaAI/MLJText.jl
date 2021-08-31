using MLJText # substitute for correct interface pkg name
using Test
using MLJBase

@testset "tfidf transformer" begin
    # add some test docs
    docs = ["Hi my name is Sam.", "How are you today?"]

    tfidf_transformer = MLJText.TfidfTransformer()
    test = machine(tfidf_transformer, docs)
    fit!(test)

    test1 = transform(test, ["Another sentence ok"])
    @test sum(test1, dims=2)[1] == 0.0
    @test size(test1) == (1, 11)

    test2 = transform(test, ["Listen Sam, today is not the day."])
    @test sum(test2, dims=2)[1] > 0.0
    @test size(test2) == (1, 11)

    test3 = transform(test, ["Another sentence ok", "Listen Sam, today is not the day."])
    @test sum(test3, dims=2)[1] == 0.0
    @test sum(test3, dims=2)[2] > 0.0
    @test size(test3) == (2, 11)
end
