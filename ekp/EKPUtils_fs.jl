using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.ParameterDistributions

using OceanBioME.Models: teos10_density, teos10_polynomial_approximation
using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi

include("Utils.jl")
include("FastSimulations.jl")

#=
- define the outputs of the 'observations' from diff input of vars
- takes the scaled parameters and unscales them to feed into the model, then takes the output and scales it 
- scales everything to be between 1 and 10 

- note that the user defined function G takes the model and other required parameters (eg Δt, stop_time) variables
- was picked this way so that the user can entirely specify the output and type of model, as well as the 
format/processing of the raw data (such as measurements made) without input from the model
=#

# requires G as a function of the model, with Δt, stop_time and also optionally data
function calculate_observations(m, 
                                params::NamedTuple, 
                                G; 
                                data=nothing, 
                                Δt = 0.05, 
                                stop_time = 50, 
                                initial_conditions, 
                                input_scaling, 
                                output_scaling) 
    
    rescaled_values = scale_list(collect(params), -input_scaling)
    rescaled_params = NamedTuple{keys(params)}(Tuple(rescaled_values))

    model = set_model(m; params = rescaled_params, initial_conditions = initial_conditions)
    try 
        if isnothing(output_scaling)
            return  G(model; data, Δt, stop_time)
        else 
            scaled_output = nancheck.(scale_list(G(model; data, Δt, stop_time), output_scaling))
            return scaled_output
        end
    catch e
        @warn "An error occured while running the model, returned NaN to trigger failure handler"
        @error e

        return NaN 
    end       
end

#   the same function but for the CarbonChemistry model specifically, requires G as a function of model and data
function calculate_observations(u, G; data = nothing, input_scaling, output_scaling, excluded_vars)
    rescaled_u = scale_list(u, -input_scaling)
    model = set_model(; u = rescaled_u, excluded_vars = excluded_vars, return_model = true)
    
    try
        if isnothing(output_scaling)
            return G(model; data)
        else 
            return scale_list(G(model; data), output_scaling)
        end
    catch e
        #@error e
        @warn "params returned a root error in the CarbonChemistry model, returned NaN to trigger failure handler" 
        return NaN
    end
end

##########################################################################################
#defines a struct that contains all the things needed to perform EKP optimisation
@kwdef struct EKPObject{FT, MD, D, DT}
    model :: MD
    Δt :: FT = 0
    stop_time :: FT = 0
    initial_conditions :: NamedTuple 
    
    prior :: D
    excluded_vars = nothing
    mutable_vars = nothing
    input_scaling = []
    output_scaling = []

    iterations :: Int64 = 12
    observation :: Function

    data :: DT 
    unparamaterised_prior :: NamedTuple
    additional_fns = nothing
end

#   EKPObject function for models such as LOBSTER, NPZD, PISCES, etc...
function EKPObject(; 
                    model, 
                    G, 
                    Δt, 
                    stop_time, 
                    mutable_vars, 
                    iterations, 
                    prior_mean, 
                    prior_std, 
                    constraints::NamedTuple = NamedTuple{}(),
                    data = nothing,
                    kwargs...)
    st = now()
    initial_values = []
    tracers = keys(model.fields)

    #   sets constraints to be either what is specified or to [-Inf, 0], [0, Inf] or [-Inf, Inf] depending on sign
    lims = zeros(length(mutable_vars), 2)
    for i in 1:length(mutable_vars)
        if mutable_vars[i] ∈ keys(constraints)
            lims[i, :] = constraints[mutable_vars[i]]
        elseif prior_mean[i] > 0.01
            lims[i, :] = [0, Inf]
        elseif prior_mean[i] < -0.01
            lims[i, :] = [-Inf, 0]
        else 
            lims[i, :] = [-Inf, Inf]
        end
    end

    #checks that initial conditions have been set and are not all zero
    for tracer in tracers
        if getfield(model.fields, tracer) !== 0
            push!(initial_values, model.fields[tracer][1])
        end
    end
    if initial_values == []
        return throw("Initial conditions not set. Please set initial conditions for the model before passing this function.")
    end 
    initial_conditions = NamedTuple{tracers}(Tuple(initial_values))

    j(x) = join([String(x), initial_conditions[x]], " = ")
    init_cond_display = join([j(x) for x in keys(initial_conditions)], ", ")
    @info "Initial conditions have been set. \nInitial conditions are \n" * init_cond_display
    println("")

    @info "Simulation parameters are 
            \nΔt = " * string(Δt) * ", 
            \nstop time = " * string(stop_time) * ", 
            \niterations = " * string(iterations) * ", 
            \noptimised variables = " * join(mutable_vars, ", \n") * "\n"
    println("")

    #   scales prior mean and std so that all prior means are between 1 and 10 and stds are some proportion of the means
    input_scaling = scale_parameters(zeroinfcheck.(prior_mean, 0))[2]
    scaled_mean = zeroinfcheck.(scale_parameters(prior_mean)[1], 0.0)
    scaled_std = zeroinfcheck.(scale_list(prior_std, input_scaling), 0.1)
    scaled_lim_min = scale_list(lims[:, 1], input_scaling)
    scaled_lim_max =  scale_list(lims[:, 2], input_scaling)

    #@info "Prior scaled means: \n" * join(round.(scaled_mean; digits = 4), ", ")
    #@info "Prior scaled stds: \n" * join(round.(scaled_std; digits = 4), ", ")

    #   check that the std + mean are not too close to or above upper bound; if it is then reduce std (see EKP docs)
    for i in eachindex(scaled_mean)
        sum = scaled_mean[i] + 1.2*scaled_std[i]
        diff = scaled_mean[i] + 1.2*scaled_std[i]
        if sum > scaled_lim_max[i] || diff < scaled_lim_min[i]
            new_std = 0.9 * min(abs(scaled_lim_max[i]-scaled_mean[i])/(1.2), abs(scaled_lim_min[i]-scaled_mean[i])/(1.2))

            @info("std of var $i rescaled from " * string(round(scaled_std[i]; digits = 3))
            *" to " *string(round(new_std; digits = 3)) * " due to μ + σ being too close to upper/lower bound")
            println("")
            
            scaled_std[i] = new_std
        end
    end

    #   generates scaled prior distribution
    prior_dis = [constrained_gaussian("$i", 
                    scaled_mean[i], 
                    scaled_std[i], 
                    scaled_lim_min[i], scaled_lim_max[i]) for i in 1:length(scaled_mean)]
    prior = combine_distributions(prior_dis)

    #   scales output to also be between 1 and 10
    unscaled_output = calculate_observations(model, 
                                            NamedTuple{Tuple(mutable_vars)}(Tuple(scaled_mean)), 
                                            G; 
                                            data = data,
                                            Δt = Δt, 
                                            stop_time = stop_time, 
                                            initial_conditions = initial_conditions, 
                                            input_scaling = input_scaling,
                                            output_scaling = nothing)                                      
    output_scaling = scale_parameters(unscaled_output)[2]
    #@info "unscaled output: \n"  * join(round.(unscaled_output; digits = 4), ", ")

    #defines functions so that they are only in terms of the model
    F(u)= calculate_observations(model, 
                                    NamedTuple{Tuple(mutable_vars)}(Tuple(u)), 
                                    G; 
                                    data = data,
                                    Δt = Δt, 
                                    stop_time = stop_time, 
                                    initial_conditions = initial_conditions, 
                                    input_scaling = input_scaling,
                                    output_scaling = output_scaling)

    
    additional_fns = values(kwargs)
    if isempty(additional_fns) == false
        @info "additional functions have been stored. They are \n"
        for key in keys(additional_fns)
            println(string(key) * "(x) = " * string(additional_fns[key]))
        end
    end

    unparamaterised_prior = (mean = scaled_mean, std = scaled_std, left_lim = scaled_lim_min, right_lim = scaled_lim_max)

    et = now()
    @info "Initialisation time: " * string(et - st) * "\n"

    return EKPObject(model, 
                    Δt, 
                    stop_time, 
                    initial_conditions,
                    prior,
                    nothing,
                    mutable_vars,
                    input_scaling,
                    output_scaling,
                    iterations,
                    u -> F(u),
                    data,
                    unparamaterised_prior,
                    additional_fns)
end


function CarbonChemistryEKPObject(; G,
                                    data = nothing, 
                                    excluded_vars, 
                                    iterations, 
                                    prior_mean, 
                                    prior_std,
                                    kwargs...)
    st = now()

    @info "Simulation parameters are 
            \niterations = " * string(iterations) * ", 
            \nexcluded variables = " * join(excluded_vars, ", ") * 
            "\n number of parameters to optimise: " * string(length(prior_mean)) * "\n"
    
    input_scaling = scale_parameters(zeroinfcheck.(prior_mean, 0))[2]
    scaled_mean = zeroinfcheck.(scale_parameters(prior_mean)[1], 0.0)
    scaled_std = zeroinfcheck.(scale_list(prior_std, input_scaling), 0.001^2)
    lims = zeros(length(scaled_mean), 2)
    
    for i in eachindex(scaled_mean)
        if scaled_mean[i] > 0.0 
            lims[i, :] = [0, Inf]
        elseif scaled_mean[i] < -0.0
            lims[i, :] = [-Inf, 0]
        else
            lims[i, :] = [-Inf, Inf]
        end
    end

    for i in eachindex(scaled_mean)
        sum = scaled_mean[i] + 1.2*scaled_std[i]
        diff = scaled_mean[i] + 1.2*scaled_std[i]
        if sum > lims[i, 2] || diff < lims[i, 1]
            new_std = 0.9 * min(abs(lims[i, 2]-scaled_mean[i])/(1.2), abs(lims[i, 1]-scaled_mean[i])/(1.2))

            @info("std of var $i rescaled from " * string(round(scaled_std[i]; digits = 3))
            *" to " *string(round(new_std; digits = 3)) * " due to μ + σ being too close to upper/lower bound")
            println("")
            
            scaled_std[i] = new_std
        end
    end

    #   generates scaled prior distribution
    prior_dis = [constrained_gaussian("$i", 
                    scaled_mean[i], 
                    scaled_std[i], 
                    lims[i, 1], lims[i, 2]) for i in 1:length(scaled_mean)]
    prior = combine_distributions(prior_dis)

    unscaled_output = calculate_observations(scaled_mean, G; 
                                            data, 
                                            input_scaling, 
                                            output_scaling = nothing, 
                                            excluded_vars)
    output_scaling = scale_parameters(unscaled_output)[2]

    #@info "unscaled output: \n"  * join(round.(unscaled_output; digits = 4), ", ")

    F(u) = calculate_observations(u, G; 
                                    data,
                                    input_scaling,
                                    output_scaling,
                                    excluded_vars)
    
    
    #   checks for additional functions defined if there are any
    additional_fns = values(kwargs)
    if isempty(additional_fns) == false
        @info "additional functions have been stored. They are \n"
        for key in keys(additional_fns)
            println(string(key) * "(x) = " * string(additional_fns[key]))
        end
    end
    lim_min = lims[:, 1]
    lim_max = lims[:, 2]

    unparamaterised_prior = (mean = scaled_mean, std = scaled_std, left_lim = lim_min, right_lim = lim_max)

    et = now()
    @info "Initialisation time: " * string(et - st) * "\n"

    return EKPObject(CarbonChemistry(),
                    0,
                    0,
                    NamedTuple{()}(()),
                    prior,
                    excluded_vars,
                    nothing,
                    input_scaling,
                    output_scaling,
                    iterations,
                    u -> F(u),
                    data,
                    unparamaterised_prior,
                    additional_fns)
end
##########################################################################################

function optimise_parameters!(obj::EKPObject, truth::Observation)
    unscale_in(list) = scale_list(list, -1 * obj.input_scaling)
    unscale_out(list) = scale_list(list, -1 * obj.output_scaling)

    G(u) = obj.observation(u)
    start_time = now()

    #   basic settings for EKP: picked α_reg = 1.0 and update_freq = 1 to minimise ensemble collapse / divergence
    α_reg =  1.0
    update_freq = 1
    N_iter = obj.iterations
    prior = obj.prior

    #   checks that constrained_gaussian hasn't incorrectly parameterised the distribution and 
    #   changes the distribution to unconstrained (i.e set bounds on that particular variable to (-Inf, Inf))
    scaled_prior_info = obj.unparamaterised_prior
    scaled_mean = scaled_prior_info.mean
    scaled_std = scaled_prior_info.std
    prior_mean = unscale_in(scaled_prior_info.mean)
    scaled_left_lim = scaled_prior_info.left_lim
    scaled_right_lim = scaled_prior_info.right_lim

    dim_input = length(mean(prior))
    dim_output = length(truth.samples[1])

    process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq)
    uki_obj = EnsembleKalmanProcess(truth.mean, truth.obs_noise_cov, process, failure_handler_method = SampleSuccGauss())
    err = zeros(N_iter)

    diff = unscale_in(get_ϕ_mean_final(prior, uki_obj)) .- prior_mean

    broken_vars = []

    for i in eachindex(diff)
        if abs(diff[i]/prior_mean[i]) > 0.01
            println(i)
            println("diff: ",round(100*diff[i]/prior_mean[i]; digits = 4), "% | constrained mean: ", unscale_in(get_ϕ_mean_final(prior, uki_obj))[i], "| real mean: ",   prior_mean[i])
            @warn("prior mean and optimised constrained mean is not the same, likely an error with constrained_gaussian")
            if abs(diff[i]/prior_mean[i]) > 0.05
                push!(broken_vars, i)
            end
        end
    end
    for i in broken_vars
        scaled_left_lim[i] = -Inf
        scaled_right_lim[i] = Inf
    end
    @info("constraints of vars " * join(broken_vars, ", ")* " have been changed to [-Inf, Inf] to prevent incorrect sampling of the intended distribution")

    prior_dis = [constrained_gaussian("$i", 
                    scaled_mean[i], 
                    scaled_std[i], 
                    scaled_left_lim[i], scaled_right_lim[i]) for i in 1:length(scaled_mean)]
    prior = combine_distributions(prior_dis)

    #   finally runs EKP 
    st = now()

    @info "elapsed initialisation time: " * string(st - start_time)
    process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq)
    uki_obj = EnsembleKalmanProcess(truth.mean, truth.obs_noise_cov, process, failure_handler_method = SampleSuccGauss())

    #println(unscale_in(get_ϕ_mean_final(prior, uki_obj)))

    @info "EKP starting...."

    current_best = 0
    min_err = 0 
        try 
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, uki_obj)
                J =  size(params_i)[2]

                G_ens = zeros(Float64, dim_output, J)
                Threads.@threads for j in 1:J
                    G_ens[:, j] .= G(params_i[:, j])
                end
                
                EKP.update_ensemble!(uki_obj, G_ens)

                err[i] = get_error(uki_obj)[end]
                @info ("\n" * 
                "Iteration: " * string(i) *
                ", Error: " * string(err[i]) *
                " norm(Cov):" * string(norm(uki_obj.process.uu_cov[i])) * "\n")
                #display(unscale_in(get_ϕ_mean_final(prior, uki_obj)))
                if err[i] == min(negcheck.(err[1:i])...)
                    current_best = unscale_in(get_ϕ_mean_final(prior, uki_obj))
                    min_err = err[i]
                    @info ("new min of " * string(min_err))
                end
            end
            if err[end] > min_err
                @info "returned params and eqc are not from the last iteration but instead the best (min err). " * 
                "\n best params had error " * string(min_err)
            end

            #   special case for CarbonChemistry model which has a different structure to set model / get params
            if typeof(obj.model) <: CarbonChemistry
                final_ensemble = get_ϕ_final(prior, uki_obj)
                final_params = unscale_in(get_ϕ_mean_final(prior, uki_obj))
                final_cov = get_u_cov_final(uki_obj)

                final_eqc = set_model(; u = final_params, excluded_vars = obj.excluded_vars, return_model = true)
                final_params = set_model(; u = final_params, excluded_vars = obj.excluded_vars, return_model = false)
                error_std = unscale_in([sqrt(final_cov[i, i]) for i in 1:dim_input])
                et = now()
                @info "total elapsed: " * string(et - st)

                #   returns the best model (where best = lowest error)
                #   chose for it to return the lowest error model as opposed to last model in case of divergence of EKP 
                return (final_ensemble = final_ensemble, 
                        best_params = set_model(; u = current_best, excluded_vars = obj.excluded_vars, return_model = false), 
                        best_model = set_model(; u = current_best, excluded_vars = obj.excluded_vars, return_model = true), 
                        error_std = error_std, 
                        final_error = err[end])
            else 
                final_ensemble = get_ϕ_final(prior, uki_obj)
                final_params = unscale_in(get_ϕ_mean_final(prior, uki_obj))
                final_cov = get_u_cov_final(uki_obj)
                param_names = obj.mutable_vars

                final = NamedTuple{Tuple(param_names)}((final_params))
                final_model = set_model(obj.model; params = final, initial_conditions = obj.initial_conditions)

                best = NamedTuple{Tuple(param_names)}((current_best))
                best_model = set_model(obj.model; params = best, initial_conditions = obj.initial_conditions)
                error_std = unscale_in([sqrt(final_cov[i, i]) for i in 1:dim_input])
                errors = (NamedTuple{Tuple(param_names)}(Tuple(error_std)))

                et = now()
                @info "elapsed: " * string(et - st)
                #   returns the best model (where best = lowest error)
                #   chose for it to return the lowest error model as opposed to last model in case of divergence of EKP 
                return  (final_ensemble = final_ensemble, 
                        best_params = best, 
                        best_model = best_model, 
                        errors = errors, 
                        final_error = err[end])
            end
        catch e 
            @error e
            @info "min error run had error " * string(min_err)
            return current_best
        end 
end