mutable struct CountTransformer <: AbstractTextTransformer
    max_doc_freq::Float64
    min_doc_freq::Float64
end

function CountTransformer(; max_doc_freq::Float64 = 1.0, min_doc_freq::Float64 = 0.0)
    transformer = CountTransformer(max_doc_freq, min_doc_freq)
    message = MMI.clean!(transformer)
    isempty(message) || @warn message
    return transformer
end

struct CountTransformerResult
    vocab::Vector{String}
end

function _fit(transformer::CountTransformer, verbosity::Int, X::Corpus)
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
    fitresult = CountTransformerResult(vocab)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

function _transform(::CountTransformer,
                    result::CountTransformerResult,
                    v::Corpus)
    dtm_matrix = build_dtm(v, result.vocab)

    # here we return the `adjoint` of our sparse matrix to conform to
    # the `n x p` dimensions throughout MLJ
    return adjoint(dtm_matrix.dtm)
end

# for returning user-friendly form of the learned parameters:
function MMI.fitted_params(::CountTransformer, fitresult::CountTransformerResult)
    vocab = fitresult.vocab
    return (vocab = vocab,)
end

## META DATA

MMI.metadata_pkg(CountTransformer,
             name="$PKG",
             uuid="7876af07-990d-54b4-ab0e-23690620f79a",
             url="https://github.com/JuliaAI/MLJText.jl",
             is_pure_julia=true,
             license="MIT",
             is_wrapper=false
)

MMI.metadata_model(CountTransformer,
               input_scitype = Union{
                   AbstractVector{<:AbstractVector{STB.Textual}},
                   AbstractVector{<:STB.Multiset{<:ScientificNGram}},
                   AbstractVector{<:STB.Multiset{STB.Textual}}
                   },
               output_scitype = AbstractMatrix{STB.Continuous},
               path = "MLJText.CountTransformer"
               )

# # DOCUMENT STRING

"""
$(MMI.doc_header(CountTransformer))

The transformer converts a collection of documents, tokenized or pre-parsed as bags of
words/ngrams, to a matrix of term counts.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    mach = machine(model, X)

$DOC_TRANSFORMER_INPUTS

Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

- `max_doc_freq=1.0`: Restricts the vocabulary that the transformer will consider. Terms
  that occur in `> max_doc_freq` documents will not be considered by the transformer. For
  example, if `max_doc_freq` is set to 0.9, terms that are in more than 90% of the
  documents will be removed.

- `min_doc_freq=0.0`: Restricts the vocabulary that the transformer will consider. Terms
  that occur in `< max_doc_freq` documents will not be considered by the transformer. A
  value of 0.01 means that only terms that are at least in 1% of the documents will be
  included.

# Operations

- `transform(mach, Xnew)`: Based on the vocabulary learned in training, return the matrix
  of counts for `Xnew`, a vector of the same form as `X` above. The matrix has size `(n,
  p)`, where `n = length(Xnew)` and `p` the size of the vocabulary. Tokens/ngrams not
  appearing in the learned vocabulary are scored zero.

# Fitted parameters

The fields of `fitted_params(mach)` are:

- `vocab`: A vector containing the string used in the transformer's vocabulary.

$(doc_examples(:CountTransformer))

See also
[`TfidfTransformer`](@ref), [`BM25Transformer`](@ref)
"""
CountTransformer
