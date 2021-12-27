const STB = ScientificTypesBase

# aliases not exported:
const PlainNGram{N}  = NTuple{N,<:AbstractString}
const TaggedNGram{N} = NTuple{N,<:CL.TaggedWord}

# This can be made less of a hack once ScientificTypes #155 is sorted.

type2scitype(T::Type) = STB.Scitype(T, DefaultConvention())
type2scitype(::Type{<:AbstractVector{T}}) where T =
    AbstractVector{type2scitype(T)}
type2scitype(::NTuple{N,T}) where {N,T} = NTuple{type2scitype{T}}

STB.scitype(::CL.TaggedWord, ::DefaultConvention) = Annotated{Textual}
STB.scitype(::CL.Document{<:AbstractVector{T}}, ::DefaultConvention) where T =
    Annotated{AbstractVector{type2scitype(T)}}
STB.scitype(::AbstractDict{<:AbstractString,<:Integer},
           ::DefaultConvention) = Multiset{Textual}
STB.scitype(::AbstractDict{<:CL.TaggedWord,<:Integer},
           ::DefaultConvention) = Multiset{Annotated{Textual}}
STB.scitype(::AbstractDict{<:Union{CL.TaggedWord,AbstractString},<:Integer},
           ::DefaultConvention) =
               Multiset{Union{Textual,Annotated{Textual}}}
STB.scitype(::AbstractDict{<:PlainNGram{N}}, ::DefaultConvention) where N =
    Multiset{NTuple{N,Textual}}
STB.scitype(::AbstractDict{<:TaggedNGram{N}}, ::DefaultConvention) where N =
    Multiset{NTuple{N,Annotated{Textual}}}
STB.scitype(::AbstractDict{<:PlainNGram}, ::DefaultConvention) =
    Multiset{NTuple{<:Any,Textual}}
STB.scitype(::AbstractDict{<:TaggedNGram}, ::DefaultConvention) =
    Multiset{NTuple{<:Any,Annotated{Textual}}}

STB.Scitype(::Type{<:CL.TaggedWord}, ::DefaultConvention) =
    Annotated{Textual}
