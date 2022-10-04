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

get_result(::TfidfTransformer, idf::Vector{<:AbstractFloat}, vocab::Vector{String}, ::SparseMatrixCSC) =
    TfidfTransformerResult(vocab, idf)

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
    doc_terms = build_dtm(v, result.vocab)
    tfidf = similar(doc_terms.dtm, eltype(result.idf_vector))
    build_tfidf!(doc_terms.dtm, tfidf, result.idf_vector)

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
               human_name="TF-IFD transformer",
               path = "MLJText.TfidfTransformer",
               )


# # DOCUMENT STRING

"""
$(MMI.doc_header(TfidfTransformer))

The transformer converts a collection of documents, tokenized or pre-parsed as bags of
words/ngrams, to a matrix of [TF-IDF
scores](https://en.wikipedia.org/wiki/Tf–idf#Inverse_document_frequency_2). Here "TF"
means term-frequency while "IDF" means inverse document frequency (defined below). The
TF-IDF score is the product of the two. This is a common term weighting scheme in
information retrieval, that has also found good use in document classification. The goal
of using TF-IDF instead of the raw frequencies of occurrence of a token in a given
document is to scale down the impact of tokens that occur very frequently in a given
corpus and that are hence empirically less informative than features that occur in a small
fraction of the training corpus.

$DOC_IDF

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

$DOC_TRANSFORMER_INPUTS

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `max_doc_freq=1.0`: Restricts the vocabulary that the transformer will consider.  Terms
  that occur in `> max_doc_freq` documents will not be considered by the transformer. For
  example, if `max_doc_freq` is set to 0.9, terms that are in more than 90% of the
  documents will be removed.

- `min_doc_freq=0.0`: Restricts the vocabulary that the transformer will consider.  Terms
  that occur in `< max_doc_freq` documents will not be considered by the transformer. A
  value of 0.01 means that only terms that are at least in 1% of the documents will be
  included.

- `smooth_idf=true`: Control which definition of IDF to use (see above).

# Operations

- `transform(mach, Xnew)`: Based on the vocabulary and IDF learned in training, return the
  matrix of TF-IDF scores for `Xnew`, a vector of the same form as `X` above. The matrix
  has size `(n, p)`, where `n = length(Xnew)` and `p` the size of the
  vocabulary. Tokens/ngrams not appearing in the learned vocabulary are scored zero.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `vocab`: A vector containing the strings used in the transformer's vocabulary.

- `idf_vector`: The transformer's calculated IDF vector.


$(doc_examples(:TfidfTransformer))

See also [`CountTransformer`](@ref), [`BM25Transformer`](@ref)

"""
TfidfTransformer
