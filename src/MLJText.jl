module MLJText

import MLJModelInterface
using ScientificTypesBase
import ScientificTypes: DefaultConvention
import CorpusLoaders
using SparseArrays
using TextAnalysis

const MMI = MLJModelInterface
const STB = ScientificTypesBase
const CL = CorpusLoaders

const PKG = "MLJText"          # substitute model-providing package name

include("scitypes.jl")
include("tfidf_transformer.jl")

export TfidfTransformer

end # module
