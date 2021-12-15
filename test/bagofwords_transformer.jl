using MLJBase
using TextAnalysis

@testset "bag of words vectorizer transformer" begin
    # add some test docs
    docs = ["Hi my name is Sam.", "How are you today?"]

    # convert to ngrams
    ngram_vec = ngrams.(documents(Corpus(NGramDocument.(docs))))

    # train transformer
    bagofwords_vectorizer = MLJText.BagOfWordsTransformer()
    test_machine = @test_logs machine(bagofwords_vectorizer, ngram_vec)
    MLJBase.fit!(test_machine)

    # test
    test_doc = ngrams(NGramDocument("Another sentence ok"))
    test1 = transform(test_machine, [test_doc])
    @test sum(test1, dims=2)[1] == 0
    @test size(test1) == (1, 11)

    test_doc2 = ngrams(NGramDocument("Listen Sam, today is not the day."))
    test2 = transform(test_machine, [test_doc2])
    @test sum(test2, dims=2)[1] > 0.0
    @test size(test2) == (1, 11)

    test_doc3 = ngrams.(
        Corpus([NGramDocument("Another sentence ok"), NGramDocument("Listen Sam, today is not the day.")])
    )
    test3 = transform(test_machine, test_doc3)
    @test sum(test3, dims=2)[1] == 0
    @test sum(test3, dims=2)[2] == 4
    @test size(test3) == (2, 11)

    test_doc4 = [["Another", "sentence", "ok"], ["Listen", "Sam", ",", "today", "is", "not", "the", "day", "."]]
    test4 = transform(test_machine, test_doc4)
    @test sum(test4, dims=2)[1] == 0
    @test sum(test4, dims=2)[2] == 4
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
    bagofwords_vectorizer2 = MLJText.BagOfWordsTransformer()
    test_machine2 = @test_logs machine(bagofwords_vectorizer2, [bag])
    MLJBase.fit!(test_machine2)

    test_doc5 = ["How about a cat in a hat"]
    test5 = transform(test_machine2, test_doc5)
    @test sum(test5, dims=2)[1] == 3
    @test size(test5) == (1, 8)

    # test min/max features
    docs = [
        "the BIL opens the door to new possibilities and should raise our collective expectations",
        "about what we can achieve in the near term.",
        "the following projects are not yet at a stage where they could be competitive",
        "for construction dollars over the next five years,",
        "but with some attention and preliminary work, our transportation leaders could turn",
        "these stretch projects into shovel-worthy ones."
    ]
    ngram_vec = ngrams.(documents(Corpus(NGramDocument.(docs))))
    bagofwords_vectorizer3 = MLJText.BagOfWordsTransformer(max_doc_freq=0.8)
    test_machine3 = @test_logs machine(bagofwords_vectorizer3, ngram_vec)
    MLJBase.fit!(test_machine3)

    test6 = transform(test_machine3, ngram_vec)
    @test Vector(vec(sum(test6, dims=2))) == [14, 10, 14, 9, 13, 7]

end