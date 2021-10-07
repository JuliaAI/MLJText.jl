# MLJText.jl

An [MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/)
extension providing tools and model for text analysis.

| Linux | Coverage |
| :------------ | :------- |
| [![Build Status](https://github.com/JuliaAI/MLJText.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJText.jl/actions) | [![Coverage](https://codecov.io/gh/JuliaAI/MLJText.jl/branch/master/graph/badge.svg)](https://codecov.io/github/JuliaAI/MLJText.jl?branch=master) |


The goal of this package is to provide an interface to various Natural Language Processing (NLP) resources for `MLJ` via such existing packages like [TextAnalysis](https://github.com/JuliaText/TextAnalysis.jl)

Currently, we have  TF-IDF Transformer which converts a collection of raw documents into a TF-IDF matrix.

## TF-IDF Transformer
"TF" means term-frequency while "TF-IDF" means term-frequency times inverse document-frequency.  This is a common term weighting scheme in information retrieval, that has also found good use in document classification.

The goal of using TF-IDF instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

### Uses
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