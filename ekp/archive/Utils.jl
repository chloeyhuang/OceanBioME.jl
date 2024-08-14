using JLD2, Plots
#include("FastSimulations.jl")
include("PZ.jl") # Include the functions defined in PZ.jl
include("output_writer.jl")

output_path = "jld2_out.jld2"

#function set_filepath!(path::String)
#   global output_path = path
#end

function get_params(bgc; params = nothing, excluded = nothing, float_only = true)
    T = bgc
    if isdefined(bgc, :underlying_biogeochemistry) == true
        T = bgc.underlying_biogeochemistry
    end
    param_names = fieldnames(typeof(T))
    param_vals = []

    if params !== nothing
        param_intersect = intersect(propertynames(T), params)
        if isempty(param_intersect)
        throw(ArgumentError("the parameters provided are not defined in the model."))
        end

        param_names = param_intersect
        for name in param_names
            push!(param_vals, getfield(T, name))
        end 
    end 
    
    if excluded !== nothing
        param_names = []
        for (name, typ) in zip(propertynames(T), typeof(T).types)
            if float_only == true
                if  name ∉ excluded && typ == Float64
                    push!(param_names, name)
                    push!(param_vals, getfield(T, name))
                end
            elseif float_only == false
                if  name ∉ excluded
                    push!(param_names, name)
                    push!(param_vals, getfield(T, name))
                end
            end
        end 
    end

    if isnothing(params) && isnothing(excluded) && float_only == true
        param_names = []
        for (name, typ) in zip(propertynames(T), typeof(T).types)
            if  typ == Float64
                push!(param_names, name)
                push!(param_vals, getfield(T, name))
            end
        end 
    elseif isnothing(params) && isnothing(excluded) && float_only == false
        for name in param_names
            push!(param_vals, getfield(T, name))
        end
    end

    return NamedTuple{Tuple(param_names)}(Tuple(param_vals))
end

function get_bgc(model)
    T = model.biogeochemistry
    if isdefined(model.biogeochemistry, :underlying_biogeochemistry) == true
        T = model.biogeochemistry.underlying_biogeochemistry
    end
    x = nameof(typeof(T))
    bgc_name = getfield(Main, x)
    return bgc_name
end

function set_bgc(bgc, params::NamedTuple) # sets the biogeochemistry for simpler models with no grid & light_attenuation specifications (eg. PZ)
    x = nameof(typeof(bgc))
    meth = getfield(Main, x)

    F(x) = meth(x...)
    list = []
    allfields = propertynames(bgc)
    names = keys(params)

    for name in allfields
        if name ∈ names
            push!(list, params[name])
        else
            push!(list, getfield(bgc, name))
        end
    end
    bgc_vars = NamedTuple{allfields}(Tuple(list))
    return F(bgc_vars)
end

function set_bgc(bgc; grid, params::NamedTuple) # sets the biogeochemistry with the params changed in the namedtuple automatically
    bgc_underlying = bgc.underlying_biogeochemistry
    PAR_func = bgc.light_attenuation.fields[1].func
    clock = Clock(time = Float64(0))
    
    PAR = FunctionField{Center, Center, Center}(PAR_func, grid; clock)

    x = nameof(typeof(bgc_underlying))
    meth = getfield(Main, x)

    F(x) = meth(; grid = grid, light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR), x...)
    list = []
    allfields = propertynames(bgc_underlying)
    changed_vars = keys(params)
    FTnames = []
    
    for name in allfields 
        if typeof(getfield(bgc_underlying, name)) == Float64
            push!(FTnames, name)
            if name ∈ changed_vars 
                push!(list, params[name])
            else
                push!(list, getfield(bgc_underlying, name))
            end
        end 
    end
    bgc_vars = NamedTuple{Tuple(FTnames)}(Tuple(list))
    return F(bgc_vars)
end

function set_model(model; params::NamedTuple, initial_conditions = nothing)
    bgc = model.biogeochemistry
    forcing = model.forcing
    grid = model.grid

    new_bgc = 0
    
    if typeof(model.biogeochemistry) <: PhytoplanktonZooplankton{}
        new_bgc = set_bgc(bgc, params)
    else
        new_bgc = set_bgc(bgc; grid = grid, params = params)
    end

    x = nameof(typeof(model))
    model_type = getfield(Main, x)
    new_model = model_type(; biogeochemistry = new_bgc, forcing = forcing, clock = new_bgc.light_attenuation.fields[1].clock)

    if isnothing(initial_conditions)
        for tracer in keys(model.fields)
            set!(new_model.fields[tracer], model.fields[tracer])
        end
    else 
        for tracer in keys(model.fields)
            set!(new_model.fields[tracer], getfield(initial_conditions, tracer))
        end
    end

    return new_model
end

##################################
# some useful util functions

# generate a random cov matrix 
function rand_cov(n::Int)
    X = rand(n,n)
    A = X'*X
    return A
end

function plot_timeseries(times, timeseries, timeseries_est)
    function get_tracers()
        t = []
        for key in keys(timeseries)
            if minimum(timeseries_est[key]) !== maximum(timeseries_est)
                push!(t, key)
            end
        end
        return t
    end
    tracers = get_tracers()
    plot_array = []
    for key in tracers
        push!(plot_array, plot(times, [timeseries[key] timeseries_est[key]], xlabel = "time", label = [String(key) String(key) * " est. "]))
    end
    
    return plot(plot_array..., size = (1200, 200*length(keys(tracers))+100), layout = (length(tracers), 1))
end

function scale_parameters(list) #scales a list of numbers to something between 1 and 10 and returns the amount scaled by 
    scaling = -floor.(log10.(abs.(list)))
    scaled_list = nancheck.(list .* 10.0.^(scaling))
    return scaled_list, scaling
end

function scale_list(list, scaling)
    if length(list) !== length(scaling)
        throw("scaling failed: length of list and scaling arrays not equal")
    end
    return nancheck.(list.* 10.0.^(scaling))
end

function nancheck(x) 
    if isnan(x)
        return 0.0
    else 
        return x 
    end
end

function zeroinfcheck(x, a = 1.0)
    if iszero(x) || !isfinite(x)
        return a
    else
        return x 
    end
end

function negcheck(x, a = 9999999999.9)
    if x > 0 
        return x
    else 
        return a
    end
end

function filter_nt_fields(list, nt) 
    f(x) = ifelse(x ∈ list, false, true)
    return NamedTuple{filter(f, keys(nt))}(nt)
end

#removes the prescribed tracers (:PAR, :T) from timeseries; works for OceanBioME 0.10.5+ 
function remove_prescribed_tracers(m, tseries) 
    PT = keys(m.prescribed_tracers)
    tnames = filter(x -> x ∉ PT, keys(tseries)) 
    timeseries = NamedTuple{tnames}((getproperty(tseries, name) for name in tnames))
    return timeseries
end
##################

function max_period(times, timeseries) #finds the distance between two local maximums
    firstmax = argmax(timeseries)
    period = 0
    if (abs(timeseries[argmax(timeseries[1:firstmax-1])]- timeseries[firstmax])) < abs(timeseries[argmax(timeseries[firstmax+1:end])]- times[firstmax])
        period = abs(times[argmax(timeseries[1:firstmax-1])]- times[firstmax])
    else 
        period = abs(times[argmax(timeseries[firstmax+1:end])]- times[firstmax])
    end
    
    if period > 0
        return period
    else 
        return 10000
    end
end
