abstract type AbstractTextTransformer <: MMI.Unsupervised end

## General method to fit text transformer models ##
MMI.fit(transformer::AbstractTextTransformer, verbosity::Int, X) = 
    _fit(transformer, verbosity, build_corpus(X))

function _fit(transformer::AbstractTextTransformer, verbosity::Int, X::Corpus)
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
    idf = compute_idf(transformer.smooth_idf, new_dtm)

    # prepare result
    fitresult = get_result(transformer, idf, vocab)
    cache = nothing

    return fitresult, cache, NamedTuple()
end

## General method to transform using text transformer models ##
MMI.transform(transformer::AbstractTextTransformer, result, v) = 
    _transform(transformer, result, build_corpus(v))