struct ContextualModel{Ctx<:AbstractContext,M<:Model} <: AbstractModel
    context::Ctx
    model::M
end

function contextualize(model::AbstractModel, context::AbstractContext)
    return ContextualModel(context, model)
end

# TODO: What do we do for other contexts? Could handle this in general if we had a
# notion of wrapper-, primitive-context, etc.
function (cmodel::ContextualModel{<:ConditionContext})(
    varinfo::AbstractVarInfo, context::AbstractContext
)
    # Wrap `context` in the model-associated `ConditionContext`, but now using `context` as
    # `ConditionContext` child.
    return cmodel.model(varinfo, ConditionContext(cmodel.context.values, context))
end

condition(model::AbstractModel, values) = contextualize(model, ConditionContext(values))
condition(model::AbstractModel; values...) = condition(model, (; values...))
function condition(cmodel::ContextualModel{<:ConditionContext}, values)
    return contextualize(cmodel.model, ConditionContext(values, cmodel.context))
end

decondition(model::AbstractModel, args...) = model
function decondition(cmodel::ContextualModel{<:ConditionContext}, syms...)
    return contextualize(cmodel.model, decondition(cmodel.context, syms...))
end
