using MLJText # substitute for correct interface pkg name
using Test
using MLJBase
using TextAnalysis

@testset "tfidf transformer" begin
    # add some test docs
    docs = ["Hi my name is Sam.", "How are you today?"]

    # convert to ngrams
    ngram_vec = ngrams.(documents(Corpus(NGramDocument.(docs))))

    # train transformer
    tfidf_transformer = MLJText.TfidfTransformer()
    test = machine(tfidf_transformer, ngram_vec)
    MLJBase.fit!(test)

    # test
    test_doc = ngrams(NGramDocument("Another sentence ok"))
    transform(test, [test_doc])
    @test sum(test1, dims=2)[1] == 0.0
    @test size(test1) == (1, 11)

    test_doc2 = ngrams(NGramDocument("Listen Sam, today is not the day."))
    transform(test, [test_doc2])
    @test sum(test2, dims=2)[1] > 0.0
    @test size(test2) == (1, 11)

    test_doc3 = ngrams.(
        Corpus(NGramDocument("Another sentence ok"), NGramDocument("Listen Sam, today is not the day."))
    )
    transform(test, test_doc3)
    @test sum(test3, dims=2)[1] == 0.0
    @test sum(test3, dims=2)[2] > 0.0
    @test size(test3) == (2, 11)
end
