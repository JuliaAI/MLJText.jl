"""
    BagOfWordsTransformer()

    Convert a collection of raw documents to matrix representing a bag-of-words structure.

    Essentially, a bag-of-words approach to representing documents in a matrix is comprised of
    a count of every word in the document corpus/collection for every document.  This is a simple
    but often quite powerful way of representing documents as vectors.  The end representation is
    a matrix with rows representing every document in the corpus and columns representing every word
    in the corpus.  The value for each cell is the raw count of a particular word in a particular
    document.

    Similarly to the `TfidfTransformer`, the vocabulary considered can be restricted
    to words occuring in a maximum or minimum portion of documents.
"""
MMI.@mlj_model mutable struct BagOfWordsTransformer <: AbstractTextTransformer
    max_doc_freq::Float64 = 1.0
    min_doc_freq::Float64 = 0.0
end

struct BagOfWordsTransformerResult
    vocab::Vector{String}
end

function _fit(transformer::BagOfWordsTransformer, verbosity::Int, X::Corpus)
    transformer.max_doc_freq < transformer.min_doc_freq && 
        error("Max doc frequency cannot be less than Min doc frequency!")

    # process corpus vocab
    update_lexicon!(X)    

    # calculate min and max doc freq limits
    if transformer.max_doc_freq < 1 || transformer.min_doc_freq > 0
        # we need to build out the DTM
        dtm_matrix = build_dtm(X)
        n = size(dtm_matrix.dtm, 2) # docs are columns
        high = round(Int, transformer.max_doc_freq * n)
        low = round(Int, transformer.min_doc_freq * n)
        _, vocab = limit_features(dtm_matrix, high, low)
    else
        vocab = sort(collect(keys(lexicon(X))))
    end

    # prepare result
    fitresult = BagOfWordsTransformerResult(vocab)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

function _transform(::BagOfWordsTransformer, 
                    result::BagOfWordsTransformerResult,
                    v::Corpus)
    dtm_matrix = build_dtm(v, result.vocab)

    # here we return the `adjoint` of our sparse matrix to conform to 
    # the `n x p` dimensions throughout MLJ
    return adjoint(dtm_matrix.dtm)
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::BagOfWordsTransformer, fitresult::BagOfWordsTransformerResult)
    vocab = fitresult.vocab
    return (vocab = vocab,)
end

## META DATA

MMI.metadata_pkg(BagOfWordsTransformer,
             name="$PKG",
             uuid="7876af07-990d-54b4-ab0e-23690620f79a",
             url="https://github.com/JuliaAI/MLJText.jl",
             is_pure_julia=true,
             license="MIT",
             is_wrapper=false
)

MMI.metadata_model(BagOfWordsTransformer,
               input_scitype = Union{
                   AbstractVector{<:AbstractVector{STB.Textual}},
                   AbstractVector{<:STB.Multiset{<:ScientificNGram}},
                   AbstractVector{<:STB.Multiset{STB.Textual}}
                   },
               output_scitype = AbstractMatrix{STB.Continuous},
               docstring = "Build Bag-of-Words matrix for corpus of documents",
               path = "MLJText.BagOfWordsTransformer"
               )