"""
    BM25Transformer()

Convert a collection of raw documents to a matrix using the Okapi BM25 document-word statistic.

BM25 is an approach similar to that of TF-IDF in terms of representing documents in a vector
space.  The BM25 scoring function uses both term frequency (TF) and inverse document frequency 
(IDF) so that, for each term in a document, its relative concentration in the document is
scored (like TF-IDF). However, BM25 improves upon TF-IDF by incorporating probability - particularly,
the probability that a user will consider a search result relevant based on the terms in the search query
and those in each document.

The parameters `max_doc_freq`, `min_doc_freq`, and `smooth_idf` all work identically to those in the
`TfidfTransformer`.  BM25 introduces two additional parameters:

`κ` is the term frequency saturation characteristic.  Higher values represent slower saturation.  What 
we mean by saturation is the degree to which a term occuring extra times adds to the overall score.  This defaults
to 2.

`β` is a parameter, bound between 0 and 1, that amplifies the particular document length compared to the average length.
The bigger β is, the more document length is amplified in terms of the overall score.  The default value is 0.75.

For more explanations, please see:
- http://ethen8181.github.io/machine-learning/search/bm25_intro.html
- https://en.wikipedia.org/wiki/Okapi_BM25
- https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

The parameters `max_doc_freq` and `min_doc_freq` restrict the vocabulary
that the transformer will consider.  `max_doc_freq` indicates that terms in only
up to the specified percentage of documents will be considered.  For example, if
`max_doc_freq` is set to 0.9, terms that are in more than 90% of documents
will be removed.  Similarly, the `min_doc_freq` parameter restricts terms in the
other direction.  A value of 0.01 means that only terms that are at least in 1% of
documents will be included.
"""
mutable struct BM25Transformer <: AbstractTextTransformer
    max_doc_freq::Float64
    min_doc_freq::Float64
    κ::Int
    β::Float64
    smooth_idf::Bool
end

function BM25Transformer(; 
    max_doc_freq::Float64 = 1.0,
    min_doc_freq::Float64 = 0.0,
    κ::Int=2,
    β::Float64=0.75,
    smooth_idf::Bool = true
    )    
    transformer = BM25Transformer(max_doc_freq, min_doc_freq, κ, β, smooth_idf)
    message = MMI.clean!(transformer)
    isempty(message) || @warn message
    return transformer
end

struct BMI25TransformerResult
    vocab::Vector{String}
    idf_vector::Vector{Float64}
end

get_result(::BM25Transformer, idf::Vector{Float64}, vocab::Vector{String}) = BMI25TransformerResult(vocab, idf)

# BM25: Okapi Best Match 25
# Details at: https://en.wikipedia.org/wiki/Okapi_BM25
# derived from https://github.com/zgornel/StringAnalysis.jl/blob/master/src/stats.jl
function build_bm25!(doc_term_mat::SparseMatrixCSC{T},
                    bm25::SparseMatrixCSC{F},
                    idf_vector::Vector{F};
                    κ::Int=2,
                    β::Float64=0.75) where {T <: Real, F <: AbstractFloat}
    @assert size(doc_term_mat) == size(bm25)
    # Initializations
    k = F(κ)
    b = F(β)
    rows = rowvals(doc_term_mat)
    dtmvals = nonzeros(doc_term_mat)
    bm25vals = nonzeros(bm25)
    @assert size(dtmvals) == size(bm25vals)

    p, n = size(doc_term_mat)

    # TF tells us what proportion of a document is defined by a term
    words_in_documents = F.(sum(doc_term_mat; dims=1))
    ln = words_in_documents ./ mean(words_in_documents)
    oneval = one(F)

    for i = 1:n
        for j in nzrange(doc_term_mat, i)
            row = rows[j]
            tf = sqrt.(dtmvals[j] / max(words_in_documents[i], oneval))
            bm25vals[j] = idf_vector[row] * ((k + 1) * tf) /
                        (k * (oneval - b + b * ln[i]) + tf)
        end
    end

    return bm25
end

function _transform(transformer::BM25Transformer, 
                    result::BMI25TransformerResult,
                    v::Corpus)
    dtm_matrix = build_dtm(v, result.vocab)
    bm25 = similar(dtm_matrix.dtm, eltype(result.idf_vector))
    build_bm25!(dtm_matrix.dtm, bm25, result.idf_vector; κ=transformer.κ, β=transformer.β)

    # here we return the `adjoint` of our sparse matrix to conform to 
    # the `n x p` dimensions throughout MLJ
    return adjoint(bm25)
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::BM25Transformer, fitresult)
    vocab = fitresult.vocab
    idf_vector = fitresult.idf_vector
    return (vocab = vocab, idf_vector = idf_vector)
end


## META DATA

MMI.metadata_pkg(BM25Transformer,
             name="$PKG",
             uuid="7876af07-990d-54b4-ab0e-23690620f79a",
             url="https://github.com/JuliaAI/MLJText.jl",
             is_pure_julia=true,
             license="MIT",
             is_wrapper=false
)

MMI.metadata_model(BM25Transformer,
               input_scitype = Union{
                   AbstractVector{<:AbstractVector{STB.Textual}},
                   AbstractVector{<:STB.Multiset{<:ScientificNGram}},
                   AbstractVector{<:STB.Multiset{STB.Textual}}
                   },
               output_scitype = AbstractMatrix{STB.Continuous},
               docstring = "Build BM-25 matrix from raw documents",
               path = "MLJText.BM25Transformer"
               )
