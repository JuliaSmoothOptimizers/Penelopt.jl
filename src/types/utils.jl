export find_model

"""
    find_model(::Type{M}, nlp)

Unwrap `nlp` through any number of `get_model`-defined wrappers until an
instance of `M` is found. Returns `nothing` if none is found.
"""
find_model(::Type{M}, nlp::M) where {M<:AbstractNLPModel} = nlp
find_model(::Type{M}, nlp::AbstractNLPModel) where {M<:AbstractNLPModel} =
  applicable(get_model, nlp) ? find_model(M, get_model(nlp)) : nothing

# outermost model first, down to the unwrapped one
layers(nlp) = applicable(get_model, nlp) ? (nlp, layers(get_model(nlp))...) : (nlp,)