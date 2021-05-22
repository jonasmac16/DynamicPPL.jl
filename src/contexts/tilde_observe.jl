"""
    tilde_observe(context, right, left, vname, vinds, vi)

Handle observed variables, e.g., `x ~ Normal()` (where `x` does occur in the model inputs),
accumulate the log probability, and return the observed value.

Falls back to `tilde_observe(context, right, left, vi)` ignoring the information about
variable name and indices.
"""
function tilde_observe(context::AbstractContext, right, left, vname, vinds, vi)
    return tilde_observe(context, right, left, vi)
end

"""
    tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)

Handle observed variables with a `context` associated with a sampler.

Falls back to `tilde_observe(context.context, right, left, vname, vinds, vi)` ignoring
the information about the sampler.
"""
function tilde_observe(context::SamplingContext, right, left, vname, vinds, vi)
    return tilde_observe(context.context, right, left, vname, vinds, vi)
end

"""
    tilde_observe(context, right, left, vi)

Handle observed constants, e.g., `1.0 ~ Normal()`, accumulate the log probability, and
return the observed value.
"""
tilde_observe(context::AbstractContext, right, left, vi)

"""
    tilde_observe(context::SamplingContext, right, left, vi)

Handle observed constants with a `context` associated with a sampler.

Falls back to `tilde_observe(context.context, right, left, vi)` ignoring
the information about the sampler.
"""
function tilde_observe(context::SamplingContext, right, left, vi)
    return tilde_observe(context.context, right, left, vi)
end

# special leaf contexts
function tilde_observe(::JointContext, right, left, vi)
    logp = observe(right, left, vi)
    acclogp!(vi, logp)
    return logp
end
tilde_observe(::PriorContext, right, left, vi) = 0
function tilde_observe(::LikelihoodContext, right, left, vi)
    logp = observe(right, left, vi)
    acclogp!(vi, logp)
    return logp
end

# minibatches
function tilde_observe(context::MiniBatchContext, right, left, vname, vinds, vi)
    currentlogp = getlogp(vi)
    logp =
        context.loglike_scalar *
        tilde_observe(context.context, right, left, vname, vinds, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function tilde_observe(context::MiniBatchContext, right, left, vi)
    currentlogp = getlogp(vi)
    logp = context.loglike_scalar * tilde_observe(context.context, right, left, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function tilde_observe(
    context::SamplingContext{<:Any,<:MiniBatchContext}, right, left, vname, vinds, vi
)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    currentlogp = getlogp(vi)
    logp = context.loglike_scalar * tilde_observe(_context, right, left, vname, vinds, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end
function tilde_observe(context::SamplingContext{<:Any,<:MiniBatchContext}, right, left, vi)
    _context = SamplingContext(context.rng, context.sampler, context.context.context)
    currentlogp = getlogp(vi)
    logp = context.loglike_scalar * tilde_observe(_context, right, left, vi)
    setlogp!(vi, currentlogp + logp)
    return logp
end

# prefixes
function tilde_observe(context::PrefixContext, right, left, vname, vinds, vi)
    prefixcontext = context.context
    return tilde_observe(
        prefixcontext, right, prefix(prefixcontext, left), vname, vinds, vi
    )
end
function tilde_observe(context::PrefixContext, right, left, vi)
    prefixcontext = context.context
    return tilde_observe(prefixcontext, right, prefix(prefixcontext, left), vi)
end
function tilde_observe(
    context::SamplingContext{<:Any,<:PrefixContext}, right, left, vname, vinds, vi
)
    prefixcontext = context.context
    _context = SamplingContext(context.rng, context.sampler, prefixcontext.context)
    _left = prefix(prefixcontext, left)
    return tilde_observe(_context, right, _left, vname, vinds, vi)
end
function tilde_observe(context::SamplingContext{<:Any,<:PrefixContext}, right, left, vi)
    prefixcontext = context.context
    _context = SamplingContext(context.rng, context.sampler, prefixcontext.context)
    _left = prefix(prefixcontext, left)
    return tilde_observe(_context, right, _left, vi)
end

# default fallback (used e.g. by `SampleFromPrior` and `SampleUniform`)
function observe(right::Distribution, left, vi)
    increment_num_produce!(vi)
    return Distributions.loglikelihood(right, left)
end
