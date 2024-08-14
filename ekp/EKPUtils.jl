using EnsembleKalmanProcesses.Observations

include("Utils.jl")

function RunBoxModel(m; Δt = 0.05, stop_time = 50) #runs a box model; takes the model as values and returns the timeseries
    model = m
    output_path = "output/"*string(get_bgc(model)) * "_output.jld2"
    
    # Runs simulation 
    simulation = Simulation(model; Δt = Δt, stop_time = stop_time, verbose = false)
    simulation.callbacks[:output] = Callback(SpeedyOutput(output_path), IterationInterval(1);)

    run!(simulation)

    # Formats and processes output
    vars = keys(model.fields)
    file = jldopen(output_path)
    rounding = length(string(simulation.Δt))
    
    file = jldopen(output_path)
    len = parse(Int64, keys(file["timeseries/t"])[end])

    times = [file["timeseries/t/$i"] for i in 1:len]
    vars = keys(model.fields)

    timeseries = NamedTuple{vars}(ntuple(t -> zeros(len), length(vars)))
    for i in 1:len
        for tracer in vars
            getproperty(timeseries, tracer)[i] = file["timeseries/$tracer/$i"]
        end
    end

    close(file)
    close(file)

    return times, timeseries
end

function calculate_observations(m, params, G; Δt = 0.05, stop_time = 50, initial_conditions, input_scaling, output_scaling) 
    # define the outputs of the 'observations' from diff input of vars
    # takes the scaled parameters and unscales them to feed into the model,
    # then takes the output and scales it 
    # scales everything to be between 0.1 and 1.0 
    
    rescaled_values = scale_list(values(params), -input_scaling)
    rescaled_params = NamedTuple{keys(params)}(Tuple(rescaled_values))
    
    model = set_model(m; params = rescaled_params, initial_conditions = initial_conditions)
    data = RunBoxModel(model; Δt = Δt, stop_time = stop_time)

    times = data[1]
    timeseries_all = data[2]
    timeseries = remove_prescribed_tracers(m, timeseries_all)
    
    if output_scaling == nothing
        return  G(times, timeseries)
    else 
        scaled_output = nancheck.(scale_list(G(times, timeseries), output_scaling))
        return scaled_output
    end
end

##########################################################################################

#defines a struct that contains all the things needed to perform EKP optimisation
@kwdef struct EKPObject{FT, MD, NT, D}
           base_model :: MD
                   Δt :: FT = 0.05
            stop_time :: FT = 50
          mutable_vars = [] 
           iterations :: Int64 = 12
   initial_conditions :: NT
                prior :: D
           prior_mean = []
        input_scaling = []
       output_scaling = []
         observations :: Function
          RunBoxModel :: Function
end

function EKPObject(base_model, G; Δt, stop_time, mutable_vars, iterations, prior_mean::Array, prior_std::Array)
    st = now()
    initial_values = []
    tracers = keys(base_model.fields)
    
    #checks that initial conditions have been set and are not all zero
    for tracer in tracers
        if getfield(base_model.fields, tracer) !== 0
            push!(initial_values, base_model.fields[tracer][1])
        end
    end
    if initial_values == []
        return throw("Initial conditions not set. Please set initial conditions for the model before passing this function.")
    end 
    initial_conditions = NamedTuple{tracers}(Tuple(initial_values))

    j(x) = join([String(x), initial_conditions[x]], " = ")
    init_cond_display = join([j(x) for x in keys(initial_conditions)], ", ")
    @info "Initial conditions have been set. \nInitial conditions are \n" * init_cond_display
    @info "Simulation parameters are 
            \nΔt = " * string(Δt) * ", 
            \nstop time = " * string(stop_time) * ", 
            \niterations = " * string(iterations) * ", 
            \noptimised variables = " * join(mutable_vars, ", \n")

    input_scaling = scale_parameters(prior_mean)[2]
    scaled_mean = scale_parameters(prior_mean)[1]
    scaled_std = scale_list(prior_std, input_scaling)

    #@info "Prior scaled means: \n" * join(round.(scaled_mean; digits = 4), ", ")
    #@info "Prior scaled stds: \n" * join(round.(scaled_std; digits = 4), ", ")

    #generates scaled prior distribution
    prior_dis = [constrained_gaussian(String(mutable_vars[i]), 
                    scaled_mean[i], 
                    scaled_std[i], 
                    0, Inf) for i in 1:length(mutable_vars)]
    prior = combine_distributions(prior_dis)

    unscaled_output = calculate_observations(base_model, 
                                            NamedTuple{Tuple(mutable_vars)}(Tuple(scaled_mean)), 
                                            G; 
                                            Δt = Δt, 
                                            stop_time = stop_time, 
                                            initial_conditions = initial_conditions, 
                                            input_scaling = input_scaling,
                                            output_scaling = nothing)
    output_scaling = scale_parameters(unscaled_output)[2]

    #defines functions so that they are only in terms of the model
    F(u)= calculate_observations(base_model, 
                                    NamedTuple{Tuple(mutable_vars)}(Tuple(u)), 
                                    G; 
                                    Δt = Δt, 
                                    stop_time = stop_time, 
                                    initial_conditions = initial_conditions, 
                                    input_scaling = input_scaling,
                                    output_scaling = output_scaling)
    H(m) = RunBoxModel(m; Δt = Δt, stop_time = stop_time)

    et = now()

    @info "Initialisation time: " * string(et - st) * "\n"

    return EKPObject(base_model, 
                    Δt, 
                    stop_time, 
                    mutable_vars, 
                    iterations, 
                    initial_conditions, 
                    prior, 
                    prior_mean, 
                    input_scaling, 
                    output_scaling, 
                    u -> F(u), 
                    m -> H(m))
end

##########################################################################################

function optimise_parameters!(obj::EKPObject, truth::Observation)
    unscale_in(list) = scale_list(list, -1 * obj.input_scaling)
    unscale_out(list) = scale_list(list, -1 * obj.output_scaling)

    G(u) = obj.observations(u)
    RunBoxModel(m) = obj.RunBoxModel(m)

    α_reg =  1.0
    update_freq = 0
    N_iter = obj.iterations
    prior = obj.prior
    
    dim_input = length(obj.mutable_vars)
    dim_output = length(truth.samples[1])

    process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq)
    uki_obj = EnsembleKalmanProcess(truth.mean, truth.obs_noise_cov, process, failure_handler_method = SampleSuccGauss())

    err = zeros(N_iter)

    @info "EKP starting...."

    for i in 1:N_iter
        params_i = get_ϕ_final(prior, uki_obj)
        J =  size(params_i)[2]

        G_ens = zeros(Float64, dim_output, J)
        for j in 1:J
            G_ens[:, j] = G(params_i[:, j])
        end
        
        EKP.update_ensemble!(uki_obj, G_ens)

        err[i] = get_error(uki_obj)[end]
        println(
        "Iteration: " * string(i) *
        ", Error: " * string(err[i]) *
        " norm(Cov):" * string(norm(uki_obj.process.uu_cov[i])) * "\n")
    end

    final_ensemble = get_ϕ_final(prior, uki_obj)
    final_params = unscale_in(get_ϕ_mean_final(prior, uki_obj))
    final_cov = get_u_cov_final(uki_obj)

    final = NamedTuple{Tuple(param_names)}((final_params))
    final_model = set_model(obj.base_model; params = final, initial_conditions = obj.initial_conditions)
    error_std = unscale_in([sqrt(final_cov[i, i]) for i in 1:dim_input])
    errors = (NamedTuple{Tuple(param_names)}(Tuple(error_std)))

    return (final_ensemble = final_ensemble, final_params = final, final_model = final_model, errors = errors, final_error = err[end])
end