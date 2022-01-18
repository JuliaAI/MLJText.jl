function limit_features(doc_terms::DocumentTermMatrix,
                        high::Int,
                        low::Int)
    doc_freqs = vec(sum(doc_terms.dtm, dims=2))

    # build mask to restrict terms
    mask = trues(length(doc_freqs))
    if high < 1
        mask .&= (doc_freqs .<= high)
    end
    if low > 0
        mask .&= (doc_freqs .>= low)
    end

    new_terms = doc_terms.terms[mask]

    return (doc_terms.dtm[mask, :], new_terms)
end

## Helper functions to build Corpus ##
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
        doc_term_mat = sparse(rows, columns, values, m, n)
    else
        doc_term_mat = spzeros(Int, m, n)
    end
    DocumentTermMatrix(doc_term_mat, terms, row_indices)
end

## General method to calculate IDF vector ##
function compute_idf(smooth_idf::Bool, doc_term_mat::SparseMatrixCSC{T}) where {T <: Real}
    n = size(doc_term_mat, 2) # docs are columns

    smooth_idf = Int(smooth_idf)
    documents_containing_term = vec(sum(doc_term_mat .> 0, dims=2)) .+ smooth_idf
    idf = log.((n + smooth_idf) ./ documents_containing_term) .+ 1

    return idf
end
