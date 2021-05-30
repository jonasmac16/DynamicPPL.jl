struct SimpleVarInfo{NT,T} <: AbstractVarInfo
    θ::NT
    logp::Base.RefValue{T}
end

SimpleVarInfo{T}(θ) where {T<:Real} = SimpleVarInfo{typeof(θ),T}(θ, Ref(zero(T)))
SimpleVarInfo(θ) = SimpleVarInfo{Float64}(θ)

function setlogp!(vi::SimpleVarInfo, logp)
    vi.logp[] = logp
    return vi
end

function acclogp!(vi::SimpleVarInfo, logp)
    vi.logp[] += logp
    return vi
end

getindex(vi::SimpleVarInfo, spl::SampleFromPrior) = vi.θ
getindex(vi::SimpleVarInfo, spl::SampleFromUniform) = vi.θ
getindex(vi::SimpleVarInfo, spl::Sampler) = vi.θ

# Overload at highest level to just replace the value.
function tilde_assume!(ctx, right, vn, inds, vi::SimpleVarInfo{<:NamedTuple})
    value = _getvalue(vi.θ, getsym(vn), inds)
    _, logp = tilde_assume(ctx, right, value, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

AbstractPPL.getsym(vns::AbstractArray{<:VarName{sym}}) where {sym} = sym

function dot_tilde_assume!(ctx, right, left, vn, inds, vi)
    value = _getvalue(vi.θ, getsym(vn), inds)
    _, logp = dot_tilde_assume(ctx, right, value, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# Context implementations
# Only evaluation makes sense for `SimpleVarInfo`, so we only implement this.
function assume(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    left,
    vn,
    inds,
    vi::SimpleVarInfo{<:NamedTuple},
)
    return left, Distributions.loglikelihood(dist, left)
end


function dot_assume(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    vns::AbstractVector{<:VarName},
    var::AbstractMatrix,
    vi::SimpleVarInfo,
)
    @assert length(dist) == size(var, 1)
    lp = sum(Distributions.loglikelihood(dist, var))
    return var, lp
end

function dot_assume(
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    vns::AbstractArray{<:VarName},
    var::AbstractArray,
    vi::SimpleVarInfo,
)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Distributions.logpdf.(dists, var))
    return var, lp
end
