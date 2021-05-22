_getindex(x, inds::Tuple) = _getindex(x[first(inds)...], Base.tail(inds))
_getindex(x, inds::Tuple{}) = x

"""
    tilde_assume(context, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value.
"""
tilde_assume(context::AbstractContext, right, vn, inds, vi)

"""
    tilde_assume(context::SamplingContext, right, vn, inds, vi)

Handle assumed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the sampled value with a context associated
with a sampler.

Falls back to
```julia
tilde_assume(context.rng, context.context, context.sampler, right, vn, inds, vi)
```
if the context `context.context` does not call any other context, as indicated by
[`unwrap_childcontext`](@ref). Otherwise, calls `tilde_assume(c, right, vn, inds, vi)`
where `c` is a context in which the order of the sampling context and its child are swapped.
"""
function tilde_assume(context::SamplingContext, right, vn, inds, vi)
    c, reconstruct_context = unwrap_childcontext(context)
    child_of_c, reconstruct_c = unwrap_childcontext(c)
    return if child_of_c === nothing
        tilde_assume(context.rng, c, context.sampler, right, vn, inds, vi)
    else
        tilde_assume(reconstruct_c(reconstruct_context(child_of_c)), right, vn, inds, vi)
    end
end

# special leaf contexts
function tilde_assume(::JointContext, right, vn, inds, vi)
    value, logp = assume(right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function tilde_assume(rng::Random.AbstractRNG, ::JointContext, sampler, right, vn, inds, vi)
    value, logp = assume(rng, sampler, right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function tilde_assume(context::LikelihoodContext{<:NamedTuple}, right, vn, inds, vi)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(LikelihoodContext(), right, vn, inds, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG,
    context::LikelihoodContext{<:NamedTuple},
    sampler,
    right,
    vn,
    inds,
    vi,
)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(rng, LikelihoodContext(), sampler, right, vn, inds, vi)
end
function tilde_assume(::LikelihoodContext, right, vn, inds, vi)
    value, logp = assume(NoDist(right), vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function tilde_assume(
    rng::Random.AbstractRNG, ::LikelihoodContext, sampler, right, vn, inds, vi
)
    value, logp = assume(rng, sampler, NoDist(right), vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

function tilde_assume(context::PriorContext{<:NamedTuple}, right, vn, inds, vi)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(PriorContext(), right, vn, inds, vi)
end
function tilde_assume(
    rng::Random.AbstractRNG,
    context::PriorContext{<:NamedTuple},
    sampler,
    right,
    vn,
    inds,
    vi,
)
    if haskey(context.vars, getsym(vn))
        vi[vn] = vectorize(right, _getindex(getfield(context.vars, getsym(vn)), inds))
        settrans!(vi, false, vn)
    end
    return tilde_assume(rng, PriorContext(), sampler, right, vn, inds, vi)
end
function tilde_assume(::PriorContext, right, vn, inds, vi)
    value, logp = assume(right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end
function tilde_assume(rng::Random.AbstractRNG, ::PriorContext, sampler, right, vn, inds, vi)
    value, logp = assume(rng, sampler, right, vn, inds, vi)
    acclogp!(vi, logp)
    return value
end

# minibatches
function tilde_assume(context::MiniBatchContext, right, vn, inds, vi)
    return tilde_assume(context.context, right, vn, inds, vi)
end

# prefixes
function tilde_assume(context::PrefixContext, right, vn, inds, vi)
    return tilde_assume(context.context, right, prefix(context, vn), inds, vi)
end

# fallback without sampler
function assume(dist::Distribution, vn::VarName, inds, vi)
    if !haskey(vi, vn)
        error("variable $vn does not exist")
    end
    r = vi[vn]
    return r, Bijectors.logpdf_with_trans(dist, vi[vn], istrans(vi, vn))
end

# SampleFromPrior and SampleFromUniform
function assume(
    rng::Random.AbstractRNG,
    sampler::Union{SampleFromPrior,SampleFromUniform},
    dist::Distribution,
    vn::VarName,
    inds,
    vi,
)
    # Always overwrite the parameters with new ones.
    r = init(rng, dist, sampler)
    if haskey(vi, vn)
        vi[vn] = vectorize(dist, r)
        setorder!(vi, vn, get_num_produce(vi))
    else
        push!(vi, vn, r, dist, sampler)
    end
    settrans!(vi, false, vn)
    return r, Bijectors.logpdf_with_trans(dist, r, istrans(vi, vn))
end
