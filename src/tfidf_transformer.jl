"""
    TfidfTransformer()

The following is taken largely from scikit-learn's documentation:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/feature_extraction/text.py

Convert a collection of raw documents to a matrix of TF-IDF features.

"TF" means term-frequency while "TF-IDF" means term-frequency times
inverse document-frequency.  This is a common term weighting scheme in
information retrieval, that has also found good use in document
classification.

The goal of using TF-IDF instead of the raw frequencies of occurrence
of a token in a given document is to scale down the impact of tokens
that occur very frequently in a given corpus and that are hence
empirically less informative than features that occur in a small
fraction of the training corpus.

The formula that is used to compute the TF-IDF for a term `t` of a
document `d` in a document set is `tf_idf(t, d) = tf(t, d) *
idf(t)`. Assuming `smooth_idf=false`, `idf(t) = log [ n / df(t) ] + 1`
where `n` is the total number of documents in the document set and
`df(t)` is the document frequency of `t`. The document frequency is
the number of documents in the document set that contain the term
`t`. The effect of adding “1” to the idf in the equation above is that
terms with zero idf, i.e., terms that occur in all documents in a
training set, will not be entirely ignored. (Note that the idf formula
above differs from that appearing in standard texts, `idf(t) = log [ n
/ (df(t) + 1) ])`.

If `smooth_idf=true` (the default), the constant “1” is added to the
numerator and denominator of the idf as if an extra document was seen
containing every term in the collection exactly once, which prevents
zero divisions: `idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1`.

The parameters `max_doc_freq` and `min_doc_freq` restrict the vocabulary
that the transformer will consider.  `max_doc_freq` indicates that terms in only
up to the specified percentage of documents will be considered.  For example, if
`max_doc_freq` is set to 0.9, terms that are in more than 90% of documents
will be removed.  Similarly, the `min_doc_freq` parameter restricts terms in the
other direction.  A value of 0.01 means that only terms that are at least in 1% of
documents will be included.
"""
mutable struct TfidfTransformer <: AbstractTextTransformer
    max_doc_freq::Float64
    min_doc_freq::Float64
    smooth_idf::Bool
end

function TfidfTransformer(; max_doc_freq::Float64 = 1.0, min_doc_freq::Float64 = 0.0, smooth_idf::Bool = true)    
    transformer = TfidfTransformer(max_doc_freq, min_doc_freq, smooth_idf)
    message = MMI.clean!(transformer)
    isempty(message) || @warn message
    return transformer
end

struct TfidfTransformerResult
    vocab::Vector{String}
    idf_vector::Vector{Float64}
end

get_result(::TfidfTransformer, idf::Vector{Float64}, vocab::Vector{String}) = TfidfTransformerResult(vocab, idf)

function build_tfidf!(doc_term_mat::SparseMatrixCSC{T},
                      tfidf::SparseMatrixCSC{F},
                      idf_vector::Vector{F}) where {T <: Real, F <: AbstractFloat}
    rows = rowvals(doc_term_mat)
    dtmvals = nonzeros(doc_term_mat)
    tfidfvals = nonzeros(tfidf)
    @assert size(dtmvals) == size(tfidfvals)

    p, n = size(doc_term_mat)

    # TF tells us what proportion of a document is defined by a term
    words_in_documents = F.(sum(doc_term_mat; dims=1))
    oneval = one(F)

    @inbounds for i = 1:n
        for j in nzrange(doc_term_mat, i)
            row = rows[j]
            tfidfvals[j] = dtmvals[j] / max(words_in_documents[i], oneval) * idf_vector[row]
        end
    end

    return tfidf
end

function _transform(::TfidfTransformer, 
                    result::TfidfTransformerResult,
                    v::Corpus)
    dtm_matrix = build_dtm(v, result.vocab)
    tfidf = similar(dtm_matrix.dtm, eltype(result.idf_vector))
    build_tfidf!(dtm_matrix.dtm, tfidf, result.idf_vector)

    # here we return the `adjoint` of our sparse matrix to conform to 
    # the `n x p` dimensions throughout MLJ
    return adjoint(tfidf)
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::TfidfTransformer, fitresult)
    vocab = fitresult.vocab
    idf_vector = fitresult.idf_vector
    return (vocab = vocab, idf_vector = idf_vector)
end


## META DATA

MMI.metadata_pkg(TfidfTransformer,
             name="$PKG",
             uuid="7876af07-990d-54b4-ab0e-23690620f79a",
             url="https://github.com/JuliaAI/MLJText.jl",
             is_pure_julia=true,
             license="MIT",
             is_wrapper=false
)

MMI.metadata_model(TfidfTransformer,
               input_scitype = Union{
                   AbstractVector{<:AbstractVector{STB.Textual}},
                   AbstractVector{<:STB.Multiset{<:ScientificNGram}},
                   AbstractVector{<:STB.Multiset{STB.Textual}}
                   },
               output_scitype = AbstractMatrix{STB.Continuous},
               docstring = "Build TF-IDF matrix from raw documents",
               path = "MLJText.TfidfTransformer"
               )
