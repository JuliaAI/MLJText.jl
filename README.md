# MLJText.jl

An [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)
extension providing tools and models for text analysis.

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLJText.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJText.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLJText.jl/branch/dev/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJText.jl?branch=dev) |


The goal of this package is to provide an interface to various Natural Language Processing (NLP) resources for `MLJ` via such existing packages like [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl)

Currently, we have a TF-IDF Transformer which converts a collection of raw documents into a TF-IDF matrix.  We also have a similar way of representing documents using the Okapi Best Match 25 algorithm - this works in a similar fashion to TF-IDF but introduces the probability that a term is relevant in a particular document.  See more here: [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25).  Finally, there is also a simple Bag-of-Word representation available.

## TF-IDF Transformer
"TF" means term-frequency while "TF-IDF" means term-frequency times inverse document-frequency.  This is a common term weighting scheme in information retrieval, that has also found good use in document classification.

The goal of using TF-IDF instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

### Usage
The TF-IDF Transformer accepts a variety of inputs for the raw documents that one wishes to convert into a TF-IDF matrix.

Raw documents can simply be provided as tokenized documents.

```julia
using MLJ, MLJText, TextAnalysis

docs = ["Hi my name is Sam.", "How are you today?"]
tfidf_transformer = TfidfTransformer()
mach = machine(tfidf_transformer, tokenize.(docs))
MLJ.fit!(mach)

tfidf_mat = transform(mach, tokenize.(docs))
```

The resulting matrix looks like:
```
2×11 adjoint(::SparseArrays.SparseMatrixCSC{Float64, Int64}) with eltype Float64:
 0.234244  0.0       0.234244  0.0       0.234244  0.0       0.234244  0.234244  0.234244  0.0       0.0
 0.0       0.281093  0.0       0.281093  0.0       0.281093  0.0       0.0       0.0       0.281093  0.281093
```

Functionality similar to Scikit-Learn's implementation with N-Grams can easily be implemented using features from `TextAnalysis`.  Then the N-Grams themselves (either as a dictionary of Strings or dictionary of Tuples) can be passed into the transformer.  We will likely introduce an additional transformer to handle these types of conversions in a future update to `MLJText`.
```julia

# this will create unigrams and bigrams
corpus = Corpus(NGramDocument.(docs, 1, 2))

ngram_docs = ngrams.(corpus)
tfidf_transformer = TfidfTransformer()
mach = machine(tfidf_transformer, ngram_docs)
MLJ.fit!(mach)

tfidf_mat = transform(mach, ngram_docs)
```

## BM25 Transformer
BM25 is an approach similar to that of TF-IDF in terms of representing documents in a vector space.  The BM25 scoring function uses both term frequency (TF) and inverse document frequency (IDF) so that, for each term in a document, its relative concentration in the document is scored (like TF-IDF).  However, BM25 improves upon TF-IDF by incorporating probability - particularly, the probability that a user will consider a search result relevant based on the terms in the search query and those in each document.

### Usage
This transformer is used in much the same way as the `TfidfTransformer`.

```julia
using MLJ, MLJText, TextAnalysis

docs = ["Hi my name is Sam.", "How are you today?"]
bm25_transformer = BM25Transformer()
mach = machine(bm25_transformer, tokenize.(docs))
MLJ.fit!(mach)

bm25_mat = transform(mach, tokenize.(docs))
```

The resulting matrix looks like:
```
2×11 adjoint(::SparseArrays.SparseMatrixCSC{Float64, Int64}) with eltype Float64:
 0.676463  0.0      0.676463  0.0      0.676463  0.0      0.676463  0.676463  0.676463  0.0      0.0
 0.0       0.81599  0.0       0.81599  0.0       0.81599  0.0       0.0       0.0       0.81599  0.81599
```

You will note that this transformer has some additional parameters compared to the `TfidfTransformer`:
```
BM25Transformer(
    max_doc_freq = 1.0,
    min_doc_freq = 0.0,
    κ = 2,
    β = 0.75,
    smooth_idf = true)
```
Please see [http://ethen8181.github.io/machine-learning/search/bm25_intro.html](http://ethen8181.github.io/machine-learning/search/bm25_intro.html) for more details about how these parameters affect the matrix that is generated.

## Bag-of-Words Transformer
The `MLJText` package also offers a way to represent documents using the simpler bag-of-words representation.  This returns a document-term matrix (as you would get in `TextAnalysis`) that consists of the count for every word in the corpus for each document in the corpus.

### Usage
```julia
using MLJ, MLJText, TextAnalysis

docs = ["Hi my name is Sam.", "How are you today?"]
bagofwords_transformer = BagOfWordsTransformer()
mach = machine(bagofwords_transformer, tokenize.(docs))
MLJ.fit!(mach)

bagofwords_mat = transform(mach, tokenize.(docs))
```

The resulting matrix looks like:
```
2×11 adjoint(::SparseArrays.SparseMatrixCSC{Int64, Int64}) with eltype Int64:
 1  0  1  0  1  0  1  1  1  0  0
 0  1  0  1  0  1  0  0  0  1  1
```