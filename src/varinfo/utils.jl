"""
    empty!(meta::Metadata)

Empty the fields of `meta`.

This is useful when using a sampling algorithm that assumes an empty `meta`, e.g. `SMC`.
"""
function empty!(meta::Metadata)
    empty!(meta.idcs)
    empty!(meta.vns)
    empty!(meta.ranges)
    empty!(meta.vals)
    empty!(meta.dists)
    empty!(meta.gids)
    empty!(meta.orders)
    for k in keys(meta.flags)
        empty!(meta.flags[k])
    end

    return meta
end

# Removes the first element of a NamedTuple. The pairs in a NamedTuple are ordered, so this is well-defined.
if VERSION < v"1.1"
    _tail(nt::NamedTuple{names}) where names = NamedTuple{Base.tail(names)}(nt)
else
    _tail(nt::NamedTuple) = Base.tail(nt)
end

"""
    getmetadata(vi::VarInfo, vn::VarName)

Return the metadata in `vi` that belongs to `vn`.
"""
getmetadata(vi::VarInfo, vn::VarName) = vi.metadata
getmetadata(vi::TypedVarInfo, vn::VarName) = getfield(vi.metadata, getsym(vn))

"""
    getidx(vi::VarInfo, vn::VarName)

Return the index of `vn` in the metadata of `vi` corresponding to `vn`.
"""
getidx(vi::VarInfo, vn::VarName) = getmetadata(vi, vn).idcs[vn]

"""
    getrange(vi::VarInfo, vn::VarName)

Return the index range of `vn` in the metadata of `vi`.
"""
getrange(vi::VarInfo, vn::VarName) = getmetadata(vi, vn).ranges[getidx(vi, vn)]

"""
    getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})

Return the indices of `vns` in the metadata of `vi` corresponding to `vn`.
"""
function getranges(vi::AbstractVarInfo, vns::Vector{<:VarName})
    return mapreduce(vn -> getrange(vi, vn), vcat, vns, init=Int[])
end

"""
    getdist(vi::VarInfo, vn::VarName)

Return the distribution from which `vn` was sampled in `vi`.
"""
getdist(vi::VarInfo, vn::VarName) = getmetadata(vi, vn).dists[getidx(vi, vn)]

"""
    getgid(vi::VarInfo, vn::VarName)

Return the set of sampler selectors associated with `vn` in `vi`.
"""
getgid(vi::VarInfo, vn::VarName) = getmetadata(vi, vn).gids[getidx(vi, vn)]

"""
    syms(vi::VarInfo)

Returns a tuple of the unique symbols of random variables sampled in `vi`.
"""
syms(vi::UntypedVarInfo) = Tuple(unique!(map(getsym, vi.metadata.vns)))  # get all symbols
syms(vi::TypedVarInfo) = keys(vi.metadata)

# Get all indices of variables belonging to SampleFromPrior:
#   if the gid/selector of a var is an empty Set, then that var is assumed to be assigned to
#   the SampleFromPrior sampler
@inline function _getidcs(vi::UntypedVarInfo, ::SampleFromPrior)
    return filter(i -> isempty(vi.metadata.gids[i]) , 1:length(vi.metadata.gids))
end
# Get a NamedTuple of all the indices belonging to SampleFromPrior, one for each symbol
@inline function _getidcs(vi::TypedVarInfo, ::SampleFromPrior)
    return _getidcs(vi.metadata)
end
@generated function _getidcs(metadata::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findinds(metadata.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get all indices of variables belonging to a given sampler
@inline function _getidcs(vi::AbstractVarInfo, spl::Sampler)
    # NOTE: 0b00 is the sanity flag for
    #         |\____ getidcs   (mask = 0b10)
    #         \_____ getranges (mask = 0b01)
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    # Checks if cache is valid, i.e. no new pushes were made, to return the cached idcs
    # Otherwise, it recomputes the idcs and caches it
    #if haskey(spl.info, :idcs) && (spl.info[:cache_updated] & CACHEIDCS) > 0
    #    spl.info[:idcs]
    #else
        #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHEIDCS
        idcs = _getidcs(vi, spl.selector, Val(getspace(spl)))
        #spl.info[:idcs] = idcs
    #end
    return idcs
end
@inline _getidcs(vi::UntypedVarInfo, s::Selector, space) = findinds(vi.metadata, s, space)
@inline _getidcs(vi::TypedVarInfo, s::Selector, space) = _getidcs(vi.metadata, s, space)
# Get a NamedTuple for all the indices belonging to a given selector for each symbol
@generated function _getidcs(metadata::NamedTuple{names}, s::Selector, ::Val{space}) where {names, space}
    exprs = []
    # Iterate through each varname in metadata.
    for f in names
        # If the varname is in the sampler space
        # or the sample space is empty (all variables)
        # then return the indices for that variable.
        if inspace(f, space) || length(space) == 0
            push!(exprs, :($f = findinds(metadata.$f, s, Val($space))))
        end
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end
@inline function findinds(f_meta, s, ::Val{space}) where {space}
    # Get all the idcs of the vns in `space` and that belong to the selector `s`
    return filter((i) ->
        (s in f_meta.gids[i] || isempty(f_meta.gids[i])) &&
        (isempty(space) || inspace(f_meta.vns[i], space)), 1:length(f_meta.gids))
end
@inline function findinds(f_meta)
    # Get all the idcs of the vns
    return filter((i) -> isempty(f_meta.gids[i]), 1:length(f_meta.gids))
end

# Get all vns of variables belonging to spl
_getvns(vi::AbstractVarInfo, spl::Sampler) = _getvns(vi, spl.selector, Val(getspace(spl)))
_getvns(vi::AbstractVarInfo, spl::Union{SampleFromPrior, SampleFromUniform}) = _getvns(vi, Selector(), Val(()))
_getvns(vi::UntypedVarInfo, s::Selector, space) = view(vi.metadata.vns, _getidcs(vi, s, space))
function _getvns(vi::TypedVarInfo, s::Selector, space)
    return _getvns(vi.metadata, _getidcs(vi, s, space))
end
# Get a NamedTuple for all the `vns` of indices `idcs`, one entry for each symbol
@generated function _getvns(metadata, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = metadata.$f.vns[idcs.$f]))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end

# Get the index (in vals) ranges of all the vns of variables belonging to spl
@inline function _getranges(vi::AbstractVarInfo, spl::Sampler)
    ## Uncomment the spl.info stuff when it is concretely typed, not Dict{Symbol, Any}
    #if ~haskey(spl.info, :cache_updated) spl.info[:cache_updated] = CACHERESET end
    #if haskey(spl.info, :ranges) && (spl.info[:cache_updated] & CACHERANGES) > 0
    #    spl.info[:ranges]
    #else
        #spl.info[:cache_updated] = spl.info[:cache_updated] | CACHERANGES
        ranges = _getranges(vi, spl.selector, Val(getspace(spl)))
        #spl.info[:ranges] = ranges
        return ranges
    #end
end
# Get the index (in vals) ranges of all the vns of variables belonging to selector `s` in `space`
@inline function _getranges(vi::AbstractVarInfo, s::Selector, space)
    return _getranges(vi, _getidcs(vi, s, space))
end
@inline function _getranges(vi::UntypedVarInfo, idcs::Vector{Int})
    mapreduce(i -> vi.metadata.ranges[i], vcat, idcs, init=Int[])
end
@inline _getranges(vi::TypedVarInfo, idcs::NamedTuple) = _getranges(vi.metadata, idcs)

@generated function _getranges(metadata::NamedTuple, idcs::NamedTuple{names}) where {names}
    exprs = []
    for f in names
        push!(exprs, :($f = findranges(metadata.$f.ranges, idcs.$f)))
    end
    length(exprs) == 0 && return :(NamedTuple())
    return :($(exprs...),)
end
@inline function findranges(f_ranges, f_idcs)
    return mapreduce(i -> f_ranges[i], vcat, f_idcs, init=Int[])
end

"""
    set_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `true` in `vi`.
"""
function set_flag!(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = true
end

"""
    empty!(vi::VarInfo)

Empty the fields of `vi.metadata` and reset `vi.logp[]` and `vi.num_produce[]` to
zeros.

This is useful when using a sampling algorithm that assumes an empty `vi`, e.g. `SMC`.
"""
function empty!(vi::VarInfo)
    _empty!(vi.metadata)
    resetlogp!(vi)
    reset_num_produce!(vi)
    return vi
end
@inline _empty!(metadata::Metadata) = empty!(metadata)
@generated function _empty!(metadata::NamedTuple{names}) where {names}
    expr = Expr(:block)
    for f in names
        push!(expr.args, :(empty!(metadata.$f)))
    end
    return expr
end

# Functions defined only for UntypedVarInfo
"""
    keys(vi::UntypedVarInfo)

Return an iterator over all `vns` in `vi`.
"""
keys(vi::UntypedVarInfo) = keys(vi.metadata.idcs)

"""
    setgid!(vi::VarInfo, gid::Selector, vn::VarName)

Add `gid` to the set of sampler selectors associated with `vn` in `vi`.
"""
setgid!(vi::VarInfo, gid::Selector, vn::VarName) = push!(getmetadata(vi, vn).gids[getidx(vi, vn)], gid)

"""
    getlogp(vi::VarInfo)

Return the log of the joint probability of the observed data and parameters sampled in
`vi`.
"""
getlogp(vi::AbstractVarInfo) = vi.logp[]

"""
    setlogp!(vi::VarInfo, logp)

Set the log of the joint probability of the observed data and parameters sampled in
`vi` to `logp`.
"""
function setlogp!(vi::VarInfo, logp)
    vi.logp[] = logp
    return vi
end

"""
    acclogp!(vi::VarInfo, logp)

Add `logp` to the value of the log of the joint probability of the observed data and
parameters sampled in `vi`.
"""
function acclogp!(vi::VarInfo, logp)
    vi.logp[] += logp
    return vi
end

"""
    resetlogp!(vi::AbstractVarInfo)

Reset the value of the log of the joint probability of the observed data and parameters
sampled in `vi` to 0.
"""
resetlogp!(vi::AbstractVarInfo) = setlogp!(vi, zero(getlogp(vi)))

"""
    get_num_produce(vi::VarInfo)

Return the `num_produce` of `vi`.
"""
get_num_produce(vi::VarInfo) = vi.num_produce[]

"""
    set_num_produce!(vi::VarInfo, n::Int)

Set the `num_produce` field of `vi` to `n`.
"""
set_num_produce!(vi::VarInfo, n::Int) = vi.num_produce[] = n

"""
    increment_num_produce!(vi::VarInfo)

Add 1 to `num_produce` in `vi`.
"""
increment_num_produce!(vi::VarInfo) = vi.num_produce[] += 1

"""
    reset_num_produce!(vi::AbstractVarInfo)

Reset the value of `num_produce` the log of the joint probability of the observed data
and parameters sampled in `vi` to 0.
"""
reset_num_produce!(vi::AbstractVarInfo) = set_num_produce!(vi, 0)

"""
    isempty(vi::VarInfo)

Return true if `vi` is empty and false otherwise.
"""
isempty(vi::UntypedVarInfo) = isempty(vi.metadata.idcs)
isempty(vi::TypedVarInfo) = _isempty(vi.metadata)
@generated function _isempty(metadata::NamedTuple{names}) where {names}
    expr = Expr(:&&, :true)
    for f in names
        push!(expr.args, :(isempty(metadata.$f.idcs)))
    end
    return expr
end

"""
    tonamedtuple(vi::VarInfo)

Convert a `vi` into a `NamedTuple` where each variable symbol maps to the values and 
indexing string of the variable.

For example, a model that had a vector of vector-valued
variables `x` would return

```julia
(x = ([1.5, 2.0], [3.0, 1.0], ["x[1]", "x[2]"]), )
```
"""
function tonamedtuple(vi::VarInfo)
    return tonamedtuple(vi.metadata, vi)
end
@generated function tonamedtuple(metadata::NamedTuple{names}, vi::VarInfo) where {names}
    length(names) === 0 && return :(NamedTuple())
    expr = Expr(:tuple)
    map(names) do f
        push!(expr.args, Expr(:(=), f, :(getindex.(Ref(vi), metadata.$f.vns), string.(metadata.$f.vns))))
    end
    return expr
end

@inline function findvns(vi, f_vns)
    if length(f_vns) == 0
        throw("Unidentified error, please report this error in an issue.")
    end
    return map(vn -> vi[vn], f_vns)
end

function Base.eltype(vi::AbstractVarInfo, spl::Union{AbstractSampler, SampleFromPrior})
    return eltype(Core.Compiler.return_type(getindex, Tuple{typeof(vi), typeof(spl)}))
end

"""
    haskey(vi::VarInfo, vn::VarName)

Check whether `vn` has been sampled in `vi`.
"""
haskey(vi::VarInfo, vn::VarName) = haskey(getmetadata(vi, vn).idcs, vn)
function haskey(vi::TypedVarInfo, vn::VarName)
    metadata = vi.metadata
    Tmeta = typeof(metadata)
    return getsym(vn) in fieldnames(Tmeta) && haskey(getmetadata(vi, vn).idcs, vn)
end

"""
    setorder!(vi::VarInfo, vn::VarName, index::Int)

Set the `order` of `vn` in `vi` to `index`, where `order` is the number of `observe
statements run before sampling `vn`.
"""
function setorder!(vi::VarInfo, vn::VarName, index::Int)
    metadata = getmetadata(vi, vn)
    if metadata.orders[metadata.idcs[vn]] != index
        metadata.orders[metadata.idcs[vn]] = index
    end
    return vi
end

"""
    is_flagged(vi::VarInfo, vn::VarName, flag::String)

Check whether `vn` has a true value for `flag` in `vi`.
"""
function is_flagged(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)]
end

"""
    unset_flag!(vi::VarInfo, vn::VarName, flag::String)

Set `vn`'s value for `flag` to `false` in `vi`.
"""
function unset_flag!(vi::VarInfo, vn::VarName, flag::String)
    return getmetadata(vi, vn).flags[flag][getidx(vi, vn)] = false
end

"""
    set_retained_vns_del_by_spl!(vi::VarInfo, spl::Sampler)

Set the `"del"` flag of variables in `vi` with `order > vi.num_produce[]` to `true`.
"""
function set_retained_vns_del_by_spl!(vi::UntypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a vector
    gidcs = _getidcs(vi, spl)
    if get_num_produce(vi) == 0
        for i = length(gidcs):-1:1
          vi.metadata.flags["del"][gidcs[i]] = true
        end
    else
        for i in 1:length(vi.orders)
            if i in gidcs && vi.orders[i] > get_num_produce(vi)
                vi.metadata.flags["del"][i] = true
            end
        end
    end
    return nothing
end
function set_retained_vns_del_by_spl!(vi::TypedVarInfo, spl::Sampler)
    # Get the indices of `vns` that belong to `spl` as a NamedTuple, one entry for each symbol
    gidcs = _getidcs(vi, spl)
    return _set_retained_vns_del_by_spl!(vi.metadata, gidcs, get_num_produce(vi))
end
@generated function _set_retained_vns_del_by_spl!(metadata, gidcs::NamedTuple{names}, num_produce) where {names}
    expr = Expr(:block)
    for f in names
        f_gidcs = :(gidcs.$f)
        f_orders = :(metadata.$f.orders)
        f_flags = :(metadata.$f.flags)
        push!(expr.args, quote
            # Set the flag for variables with symbol `f`
            if num_produce == 0
                for i = length($f_gidcs):-1:1
                    $f_flags["del"][$f_gidcs[i]] = true
                end
            else
                for i in 1:length($f_orders)
                    if i in $f_gidcs && $f_orders[i] > num_produce
                        $f_flags["del"][i] = true
                    end
                end
            end
        end)
    end
    return expr
end

"""
    updategid!(vi::VarInfo, vn::VarName, spl::Sampler)

Set `vn`'s `gid` to `Set([spl.selector])`, if `vn` does not have a sampler selector linked
and `vn`'s symbol is in the space of `spl`.
"""
function updategid!(vi::AbstractVarInfo, vn::VarName, spl::Sampler)
    if inspace(vn, getspace(spl))
        setgid!(vi, spl.selector, vn)
    end
end
