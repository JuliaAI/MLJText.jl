abstract type AbstractTextTransformer <: MMI.Unsupervised end

function MMI.clean!(transformer::AbstractTextTransformer)
    warning = ""
    if transformer.min_doc_freq < 0.0
        warning *= "Need min_doc_freq ≥ 0. Resetting min_doc_freq=0. "
        transformer.min_doc_freq = 0.0
    end

    if transformer.max_doc_freq > 1.0
        warning *= "Need max_doc_freq ≤ 1. Resetting max_doc_freq=1. "
        transformer.max_doc_freq = 1.0
    end

    if transformer.max_doc_freq < transformer.min_doc_freq
        warning *= "max_doc_freq cannot be less than min_doc_freq, resetting to defaults. "
        transformer.min_doc_freq = 0.0
        transformer.max_doc_freq = 1.0
    end
    return warning
end

## General method to fit text transformer models ##
MMI.fit(transformer::AbstractTextTransformer, verbosity::Int, X) = 
    _fit(transformer, verbosity, build_corpus(X))

function _fit(transformer::AbstractTextTransformer, verbosity::Int, X::Corpus)
    # process corpus vocab
    update_lexicon!(X)
    dtm_matrix = build_dtm(X)
    n = size(dtm_matrix.dtm, 2) # docs are columns

    # calculate min and max doc freq limits
    if transformer.max_doc_freq < 1 || transformer.min_doc_freq > 0
        high = round(Int, transformer.max_doc_freq * n)
        low = round(Int, transformer.min_doc_freq * n)
        new_doc_term_mat, vocab = limit_features(dtm_matrix, high, low)
    else
        new_doc_term_mat = dtm_matrix.dtm
        vocab = dtm_matrix.terms
    end

    # calculate IDF
    idf = compute_idf(transformer.smooth_idf, new_doc_term_mat)

    # prepare result
    fitresult = get_result(transformer, idf, vocab, new_doc_term_mat)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

## General method to transform using text transformer models ##
MMI.transform(transformer::AbstractTextTransformer, result, v) = 
    _transform(transformer, result, build_corpus(v))