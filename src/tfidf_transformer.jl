import MLJModelInterface
import ScientificTypesBase
using SparseArrays, TextAnalysis

const PKG = "MLJText"          # substitute model-providing package name
const MMI = MLJModelInterface
const STB = ScientificTypesBase

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

"""
MMI.@mlj_model mutable struct TfidfTransformer <: MLJModelInterface.Unsupervised
    max_doc_freq::Float64 = 1.0
    min_doc_freq::Float64 = 0.0
    smooth_idf::Bool = true
end

const NGram{N} = NTuple{<:Any,<:AbstractString}

struct TfidfTransformerResult
    vocab::Vector{String}
    idf_vector::Vector{Float64}
end

function limit_features(doc_term_matrix::DocumentTermMatrix,
                        high::Int,
                        low::Int)
    doc_freqs = vec(sum(doc_term_matrix.dtm, dims=2))

    # build mask to restrict terms
    mask = trues(length(doc_freqs))
    if high < 1
        mask .&= (doc_freqs .<= high)
    end
    if low > 0
        mask .&= (doc_freqs .>= low)
    end

    new_terms = doc_term_matrix.terms[mask]

    return (doc_term_matrix.dtm[mask, :], new_terms)
end

_convert_bag_of_words(X::Dict{<:NGram, <:Integer}) = 
    Dict(join(k, " ") => v for (k, v) in X)

build_corpus(X::Vector{<:Dict{<:NGram, <:Integer}}) = 
    build_corpus(_convert_bag_of_words.(X))
build_corpus(X::Vector{<:Dict{S, <:Integer}}) where {S <: AbstractString} = 
    Corpus(NGramDocument.(X))
build_corpus(X) = Corpus(TokenDocument.(X))

# based on https://github.com/zgornel/StringAnalysis.jl/blob/master/src/dtm.jl
# and https://github.com/JuliaText/TextAnalysis.jl/blob/master/src/dtm.jl
build_dtm(docs::Corpus) = build_dtm(docs, sort(collect(keys(lexicon(docs)))))
function build_dtm(docs::Corpus, terms::Vector{T}) where {T}
    # we are flipping the orientation of this matrix
    # so we get the `columnindices` from the TextAnalysis API
    row_indices = TextAnalysis.columnindices(terms)

    m = length(terms) # terms are rows
    n = length(docs)  # docs are columns

    rows = Vector{Int}(undef, 0) # terms
    columns = Vector{Int}(undef, 0) # docs
    values = Vector{Int}(undef, 0)
    for i in eachindex(docs.documents)
        doc = docs.documents[i]
        ngs = ngrams(doc)
        for ngram in keys(ngs)
            j = get(row_indices, ngram, 0)
            v = ngs[ngram]
            if j != 0
                push!(columns, i)
                push!(rows, j)
                push!(values, v)
            end
        end
    end
    if length(rows) > 0
        dtm = sparse(rows, columns, values, m, n)
    else
        dtm = spzeros(Int, m, n)
    end
    DocumentTermMatrix(dtm, terms, row_indices)
end

MMI.fit(transformer::TfidfTransformer, verbosity::Int, X) = 
    _fit(transformer, verbosity, build_corpus(X))

function _fit(transformer::TfidfTransformer, verbosity::Int, X::Corpus)
    transformer.max_doc_freq < transformer.min_doc_freq && 
        error("Max doc frequency cannot be less than Min doc frequency!")

    # process corpus vocab
    update_lexicon!(X)
    dtm_matrix = build_dtm(X)
    n = size(dtm_matrix.dtm, 2) # docs are columns

    # calculate min and max doc freq limits
    if transformer.max_doc_freq < 1 || transformer.min_doc_freq > 0
        high = round(Int, transformer.max_doc_freq * n)
        low = round(Int, transformer.min_doc_freq * n)
        new_dtm, vocab = limit_features(dtm_matrix, high, low)
    else
        new_dtm = dtm_matrix.dtm
        vocab = dtm_matrix.terms
    end

    # calculate IDF
    smooth_idf = Int(transformer.smooth_idf)
    documents_containing_term = vec(sum(new_dtm .> 0, dims=2)) .+ smooth_idf
    idf = log.((n + smooth_idf) ./ documents_containing_term) .+ 1

    # prepare result
    fitresult = TfidfTransformerResult(vocab, idf)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

function build_tfidf!(dtm::SparseMatrixCSC{T},
                      tfidf::SparseMatrixCSC{F},
                      idf_vector::Vector{F}) where {T <: Real, F <: AbstractFloat}
    rows = rowvals(dtm)
    dtmvals = nonzeros(dtm)
    tfidfvals = nonzeros(tfidf)
    @assert size(dtmvals) == size(tfidfvals)

    p, n = size(dtm)

    # TF tells us what proportion of a document is defined by a term
    words_in_documents = F.(sum(dtm, dims=1))
    oneval = one(F)

    for i = 1:n
        for j in nzrange(dtm, i)
            row = rows[j]
            tfidfvals[j] = dtmvals[j] / max(words_in_documents[i], oneval) * idf_vector[row]
        end
    end

    return tfidf
end

MMI.transform(transformer::TfidfTransformer, result::TfidfTransformerResult, v) = 
    _transform(transformer, result, build_corpus(v))

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

const ScientificNGram{N} = NTuple{<:Any,STB.Textual}

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