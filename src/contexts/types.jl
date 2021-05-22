"""
    unwrap_childcontext(context::AbstractContext)

Return a tuple of the child context of a `context`, or `nothing` if the context does
not wrap any other context, and a function `f(c::AbstractContext)` that constructs
an instance of `context` in which the child context is replaced with `c`.

Falls back to `(nothing, _ -> context)`.
"""
function unwrap_childcontext(context::AbstractContext)
    reconstruct_context(@nospecialize(x)) = context
    return nothing, reconstruct_context
end

"""
    SamplingContext(rng, sampler, context)

Create a context that allows you to sample parameters with the `sampler` when running the model.

The `context` determines how the returned log density is computed when running the model.

See also: [`JointContext`](@ref), [`LoglikelihoodContext`](@ref), [`PriorContext`](@ref)
"""
struct SamplingContext{S<:AbstractSampler,C<:AbstractContext,R} <: AbstractContext
    rng::R
    sampler::S
    context::C
end

function unwrap_childcontext(context::SamplingContext)
    child = context.context
    function reconstruct_samplingcontext(c::AbstractContext)
        return SamplingContext(context.rng, context.sampler, c)
    end
    return child, reconstruct_samplingcontext
end

"""
    JointContext()

Create a context that allows you to compute the log joint probability of the data and parameters
when running the model.
"""
struct JointContext <: AbstractContext end

"""
    PriorContext(vars=nothing)

The `PriorContext` enables the computation of the log prior of the parameters `vars` when 
running the model.
"""
struct PriorContext{Tvars} <: AbstractContext
    vars::Tvars
end
PriorContext() = PriorContext(nothing)

"""
    LikelihoodContext(vars=nothing)

Create a context that allows you to compute the log likelihood of the parameters when
running the model.

If `vars` is `nothing` (the default), the parameter values in the provided `VarInfo` object
will be used for the computation. Otherwise, the log likelihood is evaluated for the given
values of the model's parameters.
"""
struct LikelihoodContext{Tvars} <: AbstractContext
    vars::Tvars
end
LikelihoodContext() = LikelihoodContext(nothing)

"""
    MiniBatchContext(context=JointContext(); batch_size, npoints)

Create a context that allows you to compute `logprior + s * loglikelihood` when running the
model, where `s = npoints / batch_size` is a scalar with which the log likelihood is scaled.

This context is useful in batch-based stochastic gradient descent algorithms to be optimizing 
`log(prior) + log(likelihood of all the data points)` in the expectation. Here `npoints` is
equal to the number of all data points and `batch_size` denotes the size of a batch.
"""
struct MiniBatchContext{Tctx,T} <: AbstractContext
    context::Tctx
    loglike_scalar::T
end
function MiniBatchContext(context=JointContext(); batch_size, npoints)
    return MiniBatchContext(context, npoints / batch_size)
end

function unwrap_childcontext(context::MiniBatchContext)
    child = context.context
    function reconstruct_minibatchcontext(c::AbstractContext)
        return MiniBatchContext(c, context.loglike_scalar)
    end
    return child, reconstruct_minibatchcontext
end

"""
    PrefixContext{Prefix}(context)

Create a context that allows you to use the wrapped `context` when running the model and
adds the `Prefix` to all parameters.

This context is useful in nested models to ensure that the names of the parameters are
unique.

See also: [`@submodel`](@ref)
"""
struct PrefixContext{Prefix,C} <: AbstractContext
    context::C
end
function PrefixContext{Prefix}(context::AbstractContext) where {Prefix}
    return PrefixContext{Prefix,typeof(context)}(context)
end

const PREFIX_SEPARATOR = Symbol(".")

function PrefixContext{PrefixInner}(
    context::PrefixContext{PrefixOuter}
) where {PrefixInner,PrefixOuter}
    if @generated
        :(PrefixContext{$(QuoteNode(Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)))}(
            context.context
        ))
    else
        PrefixContext{Symbol(PrefixOuter, PREFIX_SEPARATOR, PrefixInner)}(context.context)
    end
end

function prefix(::PrefixContext{Prefix}, vn::VarName{Sym}) where {Prefix,Sym}
    if @generated
        return :(VarName{$(QuoteNode(Symbol(Prefix, _prefix_seperator, Sym)))}(vn.indexing))
    else
        VarName{Symbol(Prefix, PREFIX_SEPARATOR, Sym)}(vn.indexing)
    end
end

function unwrap_childcontext(context::PrefixContext{P}) where {P}
    child = context.context
    function reconstruct_prefixcontext(c::AbstractContext)
        return PrefixContext{P}(c)
    end
    return child, reconstruct_prefixcontext
end
