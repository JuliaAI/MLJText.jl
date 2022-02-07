using MLJBase
using TextAnalysis

@testset "basic use" begin
    # add some test docs
    docs = ["Hi my name is Sam.", "How are you today?"]

    # convert to ngrams
    ngram_vec = ngrams.(documents(Corpus(NGramDocument.(docs))))

    # train tfidf transformer
    tfidf_transformer = MLJText.TfidfTransformer()
    test_tfidf_machine = @test_logs machine(tfidf_transformer, ngram_vec)
    MLJBase.fit!(test_tfidf_machine)

    # train count transformer
    count_transformer = MLJText.CountTransformer()
    test_count_machine = @test_logs machine(count_transformer, ngram_vec)
    MLJBase.fit!(test_count_machine)

    # train bm25 transformer
    bm25_transformer = MLJText.BM25Transformer()
    test_bm25_machine = @test_logs machine(bm25_transformer, ngram_vec)
    MLJBase.fit!(test_bm25_machine)

    test_machines = [test_tfidf_machine, test_count_machine, test_bm25_machine]

    # test single doc
    test_doc1 = ngrams(NGramDocument("Another sentence ok"))
    for mach = test_machines
        test_doc_transform = transform(mach, [test_doc1])
        @test sum(test_doc_transform, dims=2)[1] == 0.0
        @test size(test_doc_transform) == (1, 11)
    end
    
    # test another single doc
    test_doc2 = ngrams(NGramDocument("Listen Sam, today is not the day."))
    for mach = test_machines
        test_doc_transform = transform(mach, [test_doc2])
        @test sum(test_doc_transform, dims=2)[1] > 0.0
        @test size(test_doc_transform) == (1, 11)
    end
    
    # test two docs
    test_doc3 = ngrams.(
        Corpus([NGramDocument("Another sentence ok"), NGramDocument("Listen Sam, today is not the day.")])
    )
    for mach = test_machines
        test_doc_transform = transform(mach, test_doc3)
        @test sum(test_doc_transform, dims=2)[1] == 0.0
        @test sum(test_doc_transform, dims=2)[2] > 0.0
        @test size(test_doc_transform) == (2, 11)
    end

    # test tokenized docs
    test_doc4 = [["Another", "sentence", "ok"], ["Listen", "Sam", ",", "today", "is", "not", "the", "day", "."]]
    for mach = test_machines
        test_doc_transform = transform(mach, test_doc4)
        @test sum(test_doc_transform, dims=2)[1] == 0.0
        @test sum(test_doc_transform, dims=2)[2] > 0.0
        @test size(test_doc_transform) == (2, 11)
    end

    # test proper fit:
    # here we are testing to make sure the size of the corpus to be
    # transformed does not alter the transformation that the model
    # is doing.
    for mach = test_machines
        single_doc_transform = transform(mach, [test_doc2])
        multiple_doc_transform = transform(mach, [test_doc2, test_doc2])
        @test single_doc_transform[1, :] == multiple_doc_transform[1, :]
    end
end

@testset "bag of words use" begin
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

    # train tfidf transformer
    tfidf_transformer = MLJText.TfidfTransformer()
    test_tfidf_machine2 = @test_logs machine(tfidf_transformer, [bag])
    MLJBase.fit!(test_tfidf_machine2)

    # train count transformer
    count_transformer = MLJText.CountTransformer()
    test_count_machine2 = @test_logs machine(count_transformer, [bag])
    MLJBase.fit!(test_count_machine2)

    # train bm25 transformer
    bm25_transformer = MLJText.BM25Transformer()
    test_bm25_machine2 = @test_logs machine(bm25_transformer, [bag])
    MLJBase.fit!(test_bm25_machine2)

    test_doc5 = ["How about a cat in a hat"]
    for mach = [test_tfidf_machine2, test_count_machine2, test_bm25_machine2]
        test_doc_transform = transform(mach, test_doc5)
        @test sum(test_doc_transform, dims=2)[1] > 0.0
        @test size(test_doc_transform) == (1, 8)
    end
end

@testset "min max features use" begin
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

    # train tfidf transformer
    tfidf_transformer = MLJText.TfidfTransformer(max_doc_freq=0.8, min_doc_freq=0.2)
    test_tfidf_machine3 = @test_logs machine(tfidf_transformer, ngram_vec)
    MLJBase.fit!(test_tfidf_machine3)

    # train count transformer
    count_transformer = MLJText.CountTransformer(max_doc_freq=0.8)
    test_count_machine3 = @test_logs machine(count_transformer, ngram_vec)
    MLJBase.fit!(test_count_machine3)

    # train bm25 transformer
    bm25_transformer = MLJText.BM25Transformer(max_doc_freq=0.8, min_doc_freq=0.2)
    test_bm25_machine3 = @test_logs machine(bm25_transformer, ngram_vec)
    MLJBase.fit!(test_bm25_machine3)

    # test all three machines
    test_doc_transform = transform(test_tfidf_machine3, ngram_vec)
    @test (Vector(vec(sum(test_doc_transform, dims=2))) .> 0.2) == Bool[1, 1, 1, 1, 1, 1]

    test_doc_transform = transform(test_count_machine3, ngram_vec)
    @test Vector(vec(sum(test_doc_transform, dims=2))) == [14, 10, 14, 9, 13, 7]

    test_doc_transform = transform(test_bm25_machine3, ngram_vec)
    @test (Vector(vec(sum(test_doc_transform, dims=2))) .> 0.8) == Bool[1, 1, 1, 1, 1, 1]    
end