"""
    dot_tilde_observe(context, right, left, vname, vinds, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `dot_tilde_observe(context, right, left, vi)` ignoring the information about variable
name and indices.
"""
function dot_tilde_observe(context::AbstractContext, right, left, vname, vinds, vi)
    return dot_tilde_observe(context, right, left, vi)
end

"""
    dot_tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)

Handle broadcasted observed values, e.g., `x .~ MvNormal()` (where `x` does occur the model inputs),
accumulate the log probability, and return the observed value for a context associated with a
sampler.

Falls back to `dot_tilde_observe(context.context, right, left, vname, vinds, vi)` ignoring the
sampler.
"""
function dot_tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)
    return dot_tilde_observe(context.context, right, left, vname, vinds, vi)
end

"""
    dot_tilde_observe(context, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value.
"""
dot_tilde_observe(context::AbstractContext, right, left, vi)

"""
    dot_tilde_observe(context::SamplingContext, right, left, vi)

Handle broadcasted observed constants, e.g., `[1.0] .~ MvNormal()`, accumulate the log
probability, and return the observed value for a context associated with a sampler.

Falls back to `dot_tilde_observe(context.context, right, left, vi) ignoring the sampler.
"""
function dot_tilde_observe(context::SamplingContext, right, left, vi)
    return dot_tilde_observe(context.context, right, left, vname, vinds, vi)
end

# special leaf contexts
function dot_tilde_observe(context::JointContext, right, left, vi)
    logp = dot_observe(right, left, vi)
    acclogp!(vi, logp)
    return logp
end
dot_tilde_observe(::PriorContext, right, left, vi) = 0
function dot_tilde_observe(::LikelihoodContext, right, left, vi)
    logp = dot_observe(right, left, vi)
    acclogp!(vi, logp)
    return logp
end

# minibatches
function dot_tilde_observe(context::MiniBatchContext, right, left, vname, vinds, vi)
    currentlogp = getlogp(vi)
    logp =
        context.loglike_scalar *
        dot_tilde_observe(context.context, right, left, vname, vinds, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function dot_tilde_observe(context::MiniBatchContext, right, left, vi)
    currentlogp = getlogp(vi)
    logp = context.loglike_scalar * dot_tilde_observe(context.context, right, left, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function dot_tilde_observe(
    context::SamplingContext{<:Any,<:MiniBatchContext}, right, left, vname, vinds, vi
)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    currentlogp = getlogp(vi)
    logp =
        context.loglike_scalar * dot_tilde_observe(_context, right, left, vname, vinds, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function dot_tilde_observe(
    context::SamplingContext{<:Any,<:MiniBatchContext}, right, left, vi
)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    currentlogp = getlogp(vi)
    logp = context.loglike_scalar * dot_tilde_observe(_context, right, left, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end

# prefixes
function dot_tilde_observe(context::PrefixContext, right, left, vname, vinds, vi)
    return dot_tilde_observe(context.context, right, left, vname, vinds, vi)
end
function dot_tilde_observe(context::PrefixContext, right, left, vi)
    return dot_tilde_observe(context.context, right, left, vi)
end
function dot_tilde_observe(
    context::SamplingContext{<:Any,<:PrefixContext}, right, left, vname, vinds, vi
)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    return dot_tilde_observe(_context, right, left, vname, vinds, vi)
end
function dot_tilde_observe(context::SamplingContext{<:Any,<:PrefixContext}, right, left, vi)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    return dot_tilde_observe(_context, right, left, vi)
end

# fallbacks

# Ambiguity error when not sure to use Distributions convention or Julia broadcasting semantics
function dot_observe(
    right::Union{MultivariateDistribution,AbstractVector{<:MultivariateDistribution}},
    left::AbstractMatrix{>:AbstractVector},
    vi,
)
    return throw(DimensionMismatch(AMBIGUITY_MSG))
end

function dot_observe(dist::MultivariateDistribution, value::AbstractMatrix, vi)
    increment_num_produce!(vi)
    @debug "dist = $dist"
    @debug "value = $value"
    return Distributions.loglikelihood(dist, value)
end
function dot_observe(dists::Distribution, value::AbstractArray, vi)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return Distributions.loglikelihood(dists, value)
end
function dot_observe(dists::AbstractArray{<:Distribution}, value::AbstractArray, vi)
    increment_num_produce!(vi)
    @debug "dists = $dists"
    @debug "value = $value"
    return sum(Distributions.loglikelihood.(dists, value))
end
