using MLJText
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
    test_machine = @test_logs machine(tfidf_transformer, ngram_vec)
    MLJBase.fit!(test_machine)

    # test
    test_doc = ngrams(NGramDocument("Another sentence ok"))
    test1 = transform(test_machine, [test_doc])
    @test sum(test1, dims=2)[1] == 0.0
    @test size(test1) == (1, 11)

    test_doc2 = ngrams(NGramDocument("Listen Sam, today is not the day."))
    test2 = transform(test_machine, [test_doc2])
    @test sum(test2, dims=2)[1] > 0.0
    @test size(test2) == (1, 11)

    test_doc3 = ngrams.(
        Corpus([NGramDocument("Another sentence ok"), NGramDocument("Listen Sam, today is not the day.")])
    )
    test3 = transform(test_machine, test_doc3)
    @test sum(test3, dims=2)[1] == 0.0
    @test sum(test3, dims=2)[2] > 0.0
    @test size(test3) == (2, 11)

    test_doc4 = [["Another", "sentence", "ok"], ["Listen", "Sam", ",", "today", "is", "not", "the", "day", "."]]
    test4 = transform(test_machine, test_doc4)
    @test sum(test4, dims=2)[1] == 0.0
    @test sum(test4, dims=2)[2] > 0.0
    @test size(test4) == (2, 11)
    # test with bag of words
    bag_of_words = Dict(
        "cat in" => 1,
        "the hat" => 1,
        "the" => 2,
        "cat" => 1,
        "hat" => 1,
        "in the" => 1,
        "in" => 1,
        "the cat" => 1
    )
    bag = Dict(Tuple(String.(split(k))) => v for (k, v) in bag_of_words)
    tfidf_transformer2 = MLJText.TfidfTransformer()
    test_machine2 = @test_logs machine(tfidf_transformer2, [bag])
    MLJBase.fit!(test_machine2)

    test_doc5 = ["How about a cat in a hat"]
    test5 = transform(test_machine2, test_doc5)
    @test sum(test5, dims=2)[1] > 0.0
    @test size(test5) == (1, 8)

end
