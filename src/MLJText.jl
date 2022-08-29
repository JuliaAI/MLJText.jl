module MLJText

import MLJModelInterface
using ScientificTypesBase
import ScientificTypes: DefaultConvention
import CorpusLoaders
using SparseArrays
using TextAnalysis
using Statistics

const MMI = MLJModelInterface
const STB = ScientificTypesBase
const CL = CorpusLoaders

const PKG = "MLJText"          # substitute model-providing package name

const ScientificNGram{N} = NTuple{<:Any,STB.Textual}
const NGram{N} = NTuple{<:Any,<:AbstractString}

include("scitypes.jl")
include("utils.jl")
include("abstract_text_transformer.jl")
include("tfidf_transformer.jl")
include("count_transformer.jl")
include("bm25_transformer.jl")

export TfidfTransformer, BM25Transformer, CountTransformer

"""
$(MMI.doc_header(TfidfTransformer))


`TfidfTransformer`: Convert a collection of raw documents to a matrix of TF-IDF features.
"TF" means term-frequency while "TF-IDF" means term-frequency times inverse
document-frequency.  This is a common term weighting scheme in information retrieval, that
has also found good use in document classification. The goal of using TF-IDF instead of the
raw frequencies of occurrence of a token in a given document is to scale down the impact of
tokens that occur very frequently in a given corpus and that are hence empirically less
informative than features that occur in a small fraction of the training corpus.The formula
that is used to compute the TF-IDF for a term `t` of a document `d` in a document set is
`tf_idf(t, d) = tf(t, d) * idf(t)`.


# Training data


In MLJ or MLJBase, bind an instance `model` to data with

mach = machine(model, X)

Where

- `X`: is any matrix of input features whose items are of scitype
  `ScientificTypesBase.Textual`, `ScientificTypesBase.{Multiset{<:ScientificNGram}}`, or
  `ScientificTypesBase.Multiset{.ScientificTypesBase.Textual}`; check the scitype with
  `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters


- `max_doc_freq=1.0`: Restricts the vocabulary that the transformer will consider.
  Terms that occur in `> max_doc_freq` documents will not be considered by the
  transformer. For example, if `max_doc_freq` is set to 0.9, terms that are in more than
  90% of the documents will be removed
- `min_doc_freq=0.0`: Restricts the vocabulary that the transformer will consider.
  Terms that occur in `< max_doc_freq` documents will not be considered by the
  transformer. A value of 0.01 means that only terms that are at least in 1% of the
  documents will be included
- `smooth_idf=true`: Assuming `smooth_idf` is false, IDF is calculated using the equation
  `idf(t) = log [ n / df(t) ] + 1`, with term `d`, `n` documents, and document frequency
  `df(t)`. The `1` term outside of the logarithm has the effect that terms with zero idf
  (i.e. they occur in all documents) will not be entirely ignored. If `smooth_idf` is
  true, another `1` term is added, giving: `idf(t) = log [ (1 + n) / (1 + df(t)) ] + 1`.
  These `1`'s have the same affect as adding an extra document which contains every term
  in the collection exactly once, preventing division by 0
  matrix


# Operations


- `transform(mach, Xnew)`: Return a transformed matrix of scitype
  `Continuous` given new features `Xnew`.


# Fitted parameters


The fields of `fitted_params(mach)` are:

- `vocab`: A vector containing the string used in the transformer's vocabulary.
- `idf_vector`: The transformer's calculated IDF vector.


# Examples


`TfidfTransformer` accepts a variety of inputs. In the example below, we use simple
tokenized documents:

```julia
using MLJ, MLJText, TextAnalysis

docs = ["Hi my name is Sam.", "How are you today?"]
tfidf_transformer = TfidfTransformer()
mach = machine(tfidf_transformer, tokenize.(docs))
MLJ.fit!(mach)

fitted_params(mach)

tfidf_mat = transform(mach, tokenize.(docs))
```

We can also use the `TextAnalysis` package to implement funcionality similar to SciKit
Learn's N-grams:

```julia
using MLJ, MLJText, TextAnalysis

docs = ["Hi my name is Sam.", "How are you today?"]
corpus = Corpus(NGramDocument.(docs, 1, 2))
ngram_docs = ngrams.(corpus)

tfidf_transformer = TfidfTransformer()
mach = machine(tfidf_transformer, ngram_docs)
MLJ.fit!(mach)
fitted_params(mach)

tfidf_mat = transform(mach, ngram_docs)
```

See also
[`GaussianNBClassifier`](@ref)

"""

end # module
