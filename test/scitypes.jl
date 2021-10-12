@testset "text analysis" begin
    tagged_word = CorpusLoaders.PosTaggedWord("NN", "wheelbarrow")
    tagged_word2 = CorpusLoaders.PosTaggedWord("NN", "soil")
    @test scitype(tagged_word) == Annotated{Textual}
    bag_of_words = Dict("cat"=>1, "dog"=>3)
    @test scitype(bag_of_words) == Multiset{Textual}
    bag_of_tagged_words = Dict(tagged_word => 5)
    @test scitype(bag_of_tagged_words) == Multiset{Annotated{Textual}}
    @test scitype(Document("My Document", "kadsfkj")) == Unknown
    @test scitype(Document([tagged_word, tagged_word2])) ==
        Annotated{AbstractVector{Annotated{Textual}}}
    @test scitype(Document("My Other Doc", [tagged_word, tagged_word2])) ==
        Annotated{AbstractVector{Annotated{Textual}}}
    nested_tokens = [["dog", "cat"], ["bird", "cat"]]
    @test scitype(Document("Essay Number 1", nested_tokens)) ==
        Annotated{AbstractVector{AbstractVector{Textual}}}

    @test scitype(Dict(("cat", "in") => 3)) == Multiset{Tuple{Textual,Textual}}
    bag_of_words = Dict("cat in" => 1,
                        "the hat" => 1,
                        "the" => 2,
                        "cat" => 1,
                        "hat" => 1,
                        "in the" => 1,
                        "in" => 1,
                        "the cat" => 1)
    bag_of_ngrams =
        Dict(Tuple(String.(split(k))) => v for (k, v) in bag_of_words)
    # Dict{Tuple{String, Vararg{String, N} where N}, Int64} with 8 entries:
    #   ("cat",)       => 1
    #   ("cat", "in")  => 1
    #   ("in",)        => 1
    #   ("the", "hat") => 1
    #   ("the",)       => 2
    #   ("hat",)       => 1
    #   ("in", "the")  => 1
    #   ("the", "cat") => 1
    @test scitype(bag_of_ngrams) == Multiset{NTuple{<:Any,Textual}}

    @test scitype(Dict((tagged_word, tagged_word2) => 3)) ==
        Multiset{Tuple{Annotated{Textual},Annotated{Textual}}}
    bag_of_ngrams = Dict((tagged_word, tagged_word2) => 3,
                        (tagged_word,) => 7)
    @test scitype(bag_of_ngrams) == Multiset{NTuple{<:Any,Annotated{Textual}}}

end

