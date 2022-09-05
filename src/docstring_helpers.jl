const DOC_IDF =
    """
    In textbooks and implementations there is variation in the definition of IDF. Here two
    IDF definitions are available. The default, smoothed option provides the IDF for a
    term `t` as `log((1 + n)/(1 + df(t))) + 1`, where `n` is the total number of documents
    and `df(t)` the number of documents in which `t` appears. Setting `smooth_df = false`
    provides an IDF of `log(n/df(t)) + 1`.

    """

const DOC_TRANSFORMER_INPUTS =
    """
    Here:

    - `X` is any vector whose elements are either tokenized documents or bags of
      words/ngrams. Specifically, each element is one of the following:

      - A vector of abstract strings (tokens), e.g., `["I", "like", "Sam", ".", "Sam",
        "is", "nice", "."]` (scitype `AbstractVector{Textual}`)

      - A dictionary of counts, indexed on abstract strings, e.g., `Dict("I"=>1, "Sam"=>2,
        "Sam is"=>1)` (scitype `Multiset{Textual}}`)

      - A dictionary of counts, indexed on plain ngrams, e.g., `Dict(("I",)=>1,
        ("Sam",)=>2, ("I", "Sam")=>1)` (scitype `Multiset{<:NTuple{N,Textual} where N}`);
        here a *plain ngram* is a tuple of abstract strings.

    """

function doc_examples(T)
    t = begin
        T == :TfidfTransformer ? "tfidf_transformer" :
            T == :BM25Transformer  ? "bm25_transformer" :
            T == :CountTransformer ? "count_transformer" :
            error("Problem generating a document string for $T.")
    end

    """

    # Examples

    `$T` accepts a variety of inputs. The example below transforms tokenized documents:

    ```julia
    using MLJ
    import TextAnalysis

    $T = @load $T pkg=MLJText

    docs = ["Hi my name is Sam.", "How are you today?"]
    $t = $T()

    julia> tokenized_docs = TextAnalysis.tokenize.(docs)
    2-element Vector{Vector{String}}:
     ["Hi", "my", "name", "is", "Sam", "."]
     ["How", "are", "you", "today", "?"]

    mach = machine($t, tokenized_docs)
    fit!(mach)

    fitted_params(mach)

    tfidf_mat = transform(mach, tokenized_docs)
    ```

    Alternatively, one can provide documents pre-parsed as ngrams counts:

    ```julia
    using MLJ
    import TextAnalysis

    docs = ["Hi my name is Sam.", "How are you today?"]
    corpus = TextAnalysis.Corpus(TextAnalysis.NGramDocument.(docs, 1, 2))
    ngram_docs = TextAnalysis.ngrams.(corpus)

    julia> ngram_docs[1]
    Dict{AbstractString, Int64} with 11 entries:
      "is"      => 1
      "my"      => 1
      "name"    => 1
      "."       => 1
      "Hi"      => 1
      "Sam"     => 1
      "my name" => 1
      "Hi my"   => 1
      "name is" => 1
      "Sam ."   => 1
      "is Sam"  => 1

    $t = $T()
    mach = machine($t, ngram_docs)
    MLJ.fit!(mach)
    fitted_params(mach)

    tfidf_mat = transform(mach, ngram_docs)
    ```
    """
end
