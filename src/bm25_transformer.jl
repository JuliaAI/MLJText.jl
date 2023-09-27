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
    mean_words_in_docs::Float64
end

function get_result(::BM25Transformer, idf::Vector{F}, vocab::Vector{String}, doc_term_mat::SparseMatrixCSC) where {F <: AbstractFloat}
    words_in_documents = F.(sum(doc_term_mat; dims=1))
    mean_words_in_docs = mean(words_in_documents)
    BMI25TransformerResult(vocab, idf, mean_words_in_docs)
end

# BM25: Okapi Best Match 25
# Details at: https://en.wikipedia.org/wiki/Okapi_BM25
# derived from https://github.com/zgornel/StringAnalysis.jl/blob/master/src/stats.jl
function build_bm25!(doc_term_mat::SparseMatrixCSC{T},
                    bm25::SparseMatrixCSC{F},
                    idf_vector::Vector{F},
                    mean_words_in_docs::Float64;
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
    ln = words_in_documents ./ mean_words_in_docs
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
    doc_terms = build_dtm(v, result.vocab)
    bm25 = similar(doc_terms.dtm, eltype(result.idf_vector))
    build_bm25!(doc_terms.dtm, bm25, result.idf_vector, result.mean_words_in_docs; κ=transformer.κ, β=transformer.β)

    # here we return the `adjoint` of our sparse matrix to conform to
    # the `n x p` dimensions throughout MLJ
    return adjoint(bm25)
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::BM25Transformer, fitresult)
    vocab = fitresult.vocab
    idf_vector = fitresult.idf_vector
    mean_words_in_docs = fitresult.mean_words_in_docs
    return (vocab = vocab, idf_vector = idf_vector, mean_words_in_docs = mean_words_in_docs)
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
               path = "MLJText.BM25Transformer"
               )

# # DOC STRING

"""
$(MMI.doc_header(BM25Transformer))

The transformer converts a collection of documents, tokenized or pre-parsed as bags of
words/ngrams, to a matrix of [Okapi BM25 document-word
statistics](https://en.wikipedia.org/wiki/Okapi_BM25). The BM25 scoring function uses both
term frequency (TF) and inverse document frequency (IDF, defined below), as in
[`TfidfTransformer`](@ref), but additionally adjusts for the probability that a user will
consider a search result relevant based, on the terms in the search query and those in
each document.

$DOC_IDF

References:

- http://ethen8181.github.io/machine-learning/search/bm25_intro.html
- https://en.wikipedia.org/wiki/Okapi_BM25
- https://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

$DOC_IDF

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `max_doc_freq=1.0`: Restricts the vocabulary that the transformer will consider.
  Terms that occur in `> max_doc_freq` documents will not be considered by the
  transformer. For example, if `max_doc_freq` is set to 0.9, terms that are in more than
  90% of the documents will be removed.

- `min_doc_freq=0.0`: Restricts the vocabulary that the transformer will consider.
  Terms that occur in `< max_doc_freq` documents will not be considered by the
  transformer. A value of 0.01 means that only terms that are at least in 1% of the
  documents will be included.

- `κ=2`: The term frequency saturation characteristic. Higher values represent slower
  saturation. What we mean by saturation is the degree to which a term occurring extra
  times adds to the overall score.

- `β=0.075`: Amplifies the particular document length compared to the average length. The
  bigger β is, the more document length is amplified in terms of the overall score. The
  default value is 0.75, and the bounds are restricted between 0 and 1.

- `smooth_idf=true`: Control which definition of IDF to use (see above).

# Operations

- `transform(mach, Xnew)`: Based on the vocabulary, IDF, and mean word counts learned in
  training, return the matrix of BM25 scores for `Xnew`, a vector of the same form as `X`
  above. The matrix has size `(n, p)`, where `n = length(Xnew)` and `p` the size of the
  vocabulary. Tokens/ngrams not appearing in the learned vocabulary are scored zero.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `vocab`: A vector containing the string used in the transformer's vocabulary.

- `idf_vector`: The transformer's calculated IDF vector.

- `mean_words_in_docs`: The mean number of words in each document.

$(doc_examples(:BM25Transformer))

See also [`TfidfTransformer`](@ref), [`CountTransformer`](@ref)

"""
BM25Transformer
