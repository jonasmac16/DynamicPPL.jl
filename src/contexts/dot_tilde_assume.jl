"""
    dot_tilde_assume(context, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value.
"""
dot_tilde_assume(context::AbstractContext, right, left, vn, inds, vi)

"""
    dot_tilde_assume(context::SamplingContext, right, left, vn, inds, vi)

Handle broadcasted assumed variables, e.g., `x .~ MvNormal()` (where `x` does not occur in the
model inputs), accumulate the log probability, and return the sampled value for a context
associated with a sampler.

Falls back to
```julia
dot_tilde_assume(context.rng, context.context, context.sampler, right, left, vn, inds, vi)
```
if the context `context.context` does not call any other context, as indicated by
[`unwrap_childcontext`](@ref). Otherwise, calls `dot_tilde_assume(c, right, left, vn, inds, vi)`
where `c` is a context in which the order of the sampling context and its child are swapped.
"""
function dot_tilde_assume(context::SamplingContext, right, left, vn, inds, vi)
    c, reconstruct_context = unwrap_childcontext(context)
    child_of_c, reconstruct_c = unwrap_childcontext(c)
    return if child_of_c === nothing
        dot_tilde_assume(context.rng, c, context.sampler, right, left, vn, inds, vi)
    else
        dot_tilde_assume(reconstruct_c(reconstruct_context(child_of_c)), right, left, vn, inds, vi)
    end
end

# leaf contexts
function dot_tilde_assume(::JointContext, right, left, vn, inds, vi)
    value, logp = dot_assume(right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, ::JointContext, sampler, right, left, vn, inds, vi
)
    value, logp = dot_assume(rng, sampler, right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function dot_tilde_assume(
    context::LikelihoodContext{<:NamedTuple}, right, left, vn, inds, vi
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(LikelihoodContext(), _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(LikelihoodContext(), right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(
    rng::Random.AbstractRNG,
    context::LikelihoodContext{<:NamedTuple},
    sampler,
    right,
    left,
    vn,
    inds,
    vi,
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(rng, LikelihoodContext(), sampler, _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(rng, LikelihoodContext(), sampler, right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(context::LikelihoodContext, right, left, vn, inds, vi)
    value, logp = dot_assume(NoDist.(right), left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, context::LikelihoodContext, sampler, right, left, vn, inds, vi
)
    value, logp = dot_assume(rng, sampler, NoDist.(right), left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function dot_tilde_assume(context::PriorContext{<:NamedTuple}, right, left, vn, inds, vi)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(PriorContext(), _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(PriorContext(), right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(
    rng::Random.AbstractRNG,
    context::PriorContext{<:NamedTuple},
    sampler,
    right,
    left,
    vn,
    inds,
    vi,
)
    return if haskey(context.vars, getsym(vn))
        var = _getindex(getfield(context.vars, getsym(vn)), inds)
        _right, _left, _vns = unwrap_right_left_vns(right, var, vn)
        set_val!(vi, _vns, _right, _left)
        settrans!.(Ref(vi), false, _vns)
        dot_tilde_assume(rng, PriorContext(), sampler, _right, _left, _vns, inds, vi)
    else
        dot_tilde_assume(rng, PriorContext(), sampler, right, left, vn, inds, vi)
    end
end
function dot_tilde_assume(context::PriorContext, right, left, vn, inds, vi)
    value, logp = dot_assume(right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function dot_tilde_assume(
    rng::Random.AbstractRNG, context::PriorContext, sampler, right, left, vn, inds, vi
)
    value, logp = dot_assume(rng, sampler, right, left, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# minibatches
function dot_tilde_assume(context::MiniBatchContext, right, left, vn, inds, vi)
    return dot_tilde_assume(context.context, right, left, vn, inds, vi)
end

# prefixes
function dot_tilde_assume(context::PrefixContext, right, left, vn, inds, vi)
    return dot_tilde_assume(context.context, right, prefix.(Ref(context), vn), inds, vi)
end

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function dot_assume(
    right::Union{MultivariateDistribution,AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vn::AbstractVector{<:VarName},
    inds,
    vi,
)
    return throw(DimensionMismatch(AMBIGUITY_MSG))
end
function dot_assume(
    rng,
    sampler::AbstractSampler,
    right::Union{MultivariateDistribution,AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vn::AbstractVector{<:VarName},
    inds,
    vi,
)
    return throw(DimensionMismatch(AMBIGUITY_MSG))
end

function dot_assume(
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    inds,
    vi,
)
    @assert length(dist) == size(var, 1)
    lp = sum(zip(vns, eachcol(var))) do vn, ri
        return Bijectors.logpdf_with_trans(dist, ri, istrans(vi, vn))
    end
    return var, lp
end
function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dist::MultivariateDistribution,
    var::AbstractMatrix,
    vns::AbstractVector{<:VarName},
    inds,
    vi,
)
    @assert length(dist) == size(var, 1)
    r = get_and_set_val!(rng, vi, vns, dist, spl)
    lp = sum(Bijectors.logpdf_with_trans(dist, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end

function dot_assume(
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    inds,
    vi,
)
    # Make sure `var` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, var, istrans(vi, vns[1])))
    return var, lp
end
function dot_assume(
    rng,
    spl::Union{SampleFromPrior,SampleFromUniform},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    var::AbstractArray,
    vns::AbstractArray{<:VarName},
    inds,
    vi,
)
    r = get_and_set_val!(rng, vi, vns, dists, spl)
    # Make sure `r` is not a matrix for multivariate distributions
    lp = sum(Bijectors.logpdf_with_trans.(dists, r, istrans(vi, vns[1])))
    var .= r
    return var, lp
end

function get_and_set_val!(
    rng,
    vi,
    vns::AbstractVector{<:VarName},
    dist::MultivariateDistribution,
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    n = length(vns)
    r = init(rng, dist, spl, n)
    for i in 1:n
        vn = vns[i]
        if haskey(vi, vn)
            vi[vn] = vectorize(dist, r[:, i])
            setorder!(vi, vn, get_num_produce(vi))
        else
            push!(vi, vn, r[:, i], dist, spl)
        end
        settrans!(vi, false, vn)
    end
    return r
end

function get_and_set_val!(
    rng,
    vi,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    spl::Union{SampleFromPrior,SampleFromUniform},
)
    r = broadcast(vns, dists) do vn, dist
        init(rng, dist, spl)
    end
    if haskey(vi, vns[1])
        for i in eachindex(vns)
            vn = vns[i]
            dist = dists isa AbstractArray ? dists[i] : dists
            vi[vn] = vectorize(dist, r[i])
            settrans!(vi, false, vn)
            setorder!(vi, vn, get_num_produce(vi))
        end
    else
        push!.(Ref(vi), vns, r, dists, Ref(spl))
        settrans!.(Ref(vi), false, vns)
    end
    return r
end

function set_val!(
    vi, vns::AbstractVector{<:VarName}, dist::MultivariateDistribution, val::AbstractMatrix
)
    @assert size(val, 2) == length(vns)
    foreach(enumerate(vns)) do (i, vn)
        vi[vn] = val[:, i]
    end
    return val
end
function set_val!(
    vi,
    vns::AbstractArray{<:VarName},
    dists::Union{Distribution,AbstractArray{<:Distribution}},
    val::AbstractArray,
)
    @assert size(val) == size(vns)
    foreach(CartesianIndices(val)) do ind
        dist = dists isa AbstractArray ? dists[ind] : dists
        vi[vns[ind]] = vectorize(dist, val[ind])
    end
    return val
end
