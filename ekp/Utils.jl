using Plots
using OceanBioME.Models: teos10_density, teos10_polynomial_approximation
using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi

#   gets the parameter names and values in a NamedTuple (doesn't work with nested NamedTuples)
@inline function get_params(bgc; params = nothing, excluded = nothing, float_only = true)
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
#   returns bgc name as a callable function
@inline function get_bgc(model)
    T = model
    if isdefined(model, :biogeochemistry)
        T = model.biogeochemistry
        if isdefined(model.biogeochemistry, :underlying_biogeochemistry) == true
            T = model.biogeochemistry.underlying_biogeochemistry
        end
    end
    x = nameof(typeof(T))
    bgc_name = getfield(Main, x)
    return bgc_name
end

 #  sets the biogeochemistry for simpler models with no grid & light_attenuation specifications (eg. PZ)
 @inline function set_bgc(bgc, params::NamedTuple)
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
#   sets the biogeochemistry with the params as per the NamedTuple kwarg params
@inline function set_bgc(bgc; grid, params::NamedTuple) 
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

#   returns a model of the same type with the params and inital conditions set as per the kwargs
@inline function set_model(model; params::NamedTuple, initial_conditions = nothing)
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

#   set model for carbon chemistry; takes u as a list because nested NamedTuples are annoying to work with 
@inline function set_model(; u, excluded_vars, excluded_eqns, return_model = true)
    i = 1
    vals = []
    eqc_names = (:K0, :K1, :K2, :KB, :KW, :KS, :KF, :KP1, :KP2, :KP3, :KSi)
    eqc_modified = setdiff(eqc_names, excluded_eqns)

    getfn(x) = getfield(Main, x)
    
    for c_name in eqc_names
        c = getfn(c_name)
        if c_name ∈ excluded_eqns
            df = get_params(c())
            push!(vals, df)
        else
            names = keys(get_params(c(); excluded = excluded_vars))
            len = length(names)
            ilen = i + len - 1
            new_c = NamedTuple{names}(Tuple(u[i:ilen]))
            i = ilen + 1
            push!(vals, new_c)
        end
    end
    new_eqc = NamedTuple{eqc_names}(Tuple(vals))
    if return_model == true 
        return CarbonChemistry(; 
                        solubility = K0(; new_eqc.K0...), 
                        carbonic_acid = (K1 = K1(; new_eqc.K1...), K2 = K2(; new_eqc.K2...)),
                        boric_acid = KB(; new_eqc.KB...),
                        water = KW(; new_eqc.KW...),
                        sulfate = KS(; new_eqc.KS...),
                        fluoride = KF(; new_eqc.KF...),
                        phosphoric_acid = (KP1 = KP1(; new_eqc.KP1...), KP2 = KP2(; new_eqc.KP2...), KP3 = KP3(; new_eqc.KP3...)),
                        silicic_acid = KSi(; new_eqc.KSi...)) 
    else    
        return new_eqc
    end
end

#   gets raw params as a vector from CarbonChemistry model
@inline function get_cc_params_raw(model; excluded_terms = (:inverse_T, :log_T, :T²), excluded_constants = (:ionic_strength, :density_function, :calcite_solubility))
    u = []
    for key in propertynames(model)
        if key ∉ excluded_constants
            equation = getproperty(model, key)
            try length(equation)
                for eqname in keys(equation)
                    eq = equation[eqname]
                    for name in propertynames(eq)
                        if name ∉ excluded_terms && typeof(getproperty(eq, name)) == Float64
                            push!(u, getproperty(eq, name))
                        end
                    end
                end
            catch 
                for name in propertynames(equation)
                    if name ∉ excluded_terms && typeof(getproperty(equation, name)) == Float64
                        push!(u, getproperty(equation, name))
                    end
                end
            end
        end
    end
    return u
end


##################################
# some useful util functions

# generate a random cov matrix 
function rand_cov(n::Int)
    X = rand(n,n)
    A = X'*X
    return A
end

#plots two timeseries
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

#scales a list of numbers to something between 1 and 10 and returns the amount scaled by as powers of 10
function scale_parameters(list) 
    scaling = -floor.(log10.(abs.(list)))
    scaled_list = nancheck.(list .* 10.0.^(scaling))
    return scaled_list, scaling
end

#scales list with the provided scaling as powers of 10
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

@inline function filter_nt_fields(list::Tuple, nt::NamedTuple)
    return Base.structdiff(nt, NamedTuple{list})
end

#removes the prescribed tracers (:PAR, :T) from timeseries; works for OceanBioME 0.10.5+ 
function remove_prescribed_tracers(m, tseries) 
    PT = keys(m.prescribed_tracers)
    tnames = filter(x -> x ∉ PT, keys(tseries)) 
    timeseries = NamedTuple{tnames}((getproperty(tseries, name) for name in tnames))
    return timeseries
end
