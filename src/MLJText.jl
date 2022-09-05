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

include("docstring_helpers.jl")
include("scitypes.jl")
include("utils.jl")
include("abstract_text_transformer.jl")
include("tfidf_transformer.jl")
include("count_transformer.jl")
include("bm25_transformer.jl")

export TfidfTransformer, BM25Transformer, CountTransformer


end # module
