using OceanBioME, Oceananigans
using OceanBioME.Models: NPZDModel, LOBSTERModel
using Oceananigans.Units
using Oceananigans.Fields: FunctionField

using Dates: now
using JLD2
using Plots, LinearAlgebra
using Statistics, Distributions, Random
using DataFrames
using ProfileView, BenchmarkTools

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

include("EKPUtils.jl")
include("archive/PZ.jl")
include("FastSimulations.jl")

#=   
-   for the moment this still uses a weird timestepper for the BoxModel in FastSimulations.jl but can be easily 
    changed when defining G
-   the timestepper is fast but should use the new timestepper which is only slightly slower / the same specified

-   otherwise require defining two functions: 
    -   G(model; data, Δt, stop_time)
    -   generate_data (returns ObservationSeries)
-   PZ.jl is included because I was mainly testing with that at the very start 
    and now have something defined in Utils
=#

year = years = 365day

#config for EKP, prior noise / stds proportional to mean guesses
prior_noise = 0.1
observation_noise = 0.001
priorstds = 0.2

function G(model; data = nothing, Δt = 0.05, stop_time = 50.0)
    out = RunBoxModel(model; Δt, stop_time)
    times = out[1]
    timeseries = out[2]
    n = length(times)
    observations = []
    
    for tracer in keys(timeseries)
        if tracer !== :T
            TS_max = maximum(timeseries[tracer])
            TS_min = minimum(timeseries[tracer])
            TS_rms = sqrt(1/n * sum(abs2, (timeseries[tracer].-TS_min)))
            TS_end = timeseries[tracer][end]
            T_max = argmax(timeseries[tracer])
            push!(observations, TS_max - TS_min, TS_rms, TS_end, T_max)
        end
    end
    # observations returned: max - min, rms, last point of timeseries
    # println(observations)
    return observations
end

function G_single(times, timeseries, tracer)
    n = length(times)
    observations = []
    #print(timeseries)
    TS_max = maximum(timeseries[tracer])
    T_max = argmax(timeseries[tracer])
    TS_min = minimum(timeseries[tracer])
    TS_rms = sqrt(1/n * sum(abs2, (timeseries[tracer].-TS_min)))
    TS_end = timeseries[tracer][end]
    TS_start = timeseries[tracer][1]
    push!(observations, TS_max - TS_min, TS_rms, TS_end, T_max)
    # observations returned: end - start, rms, last point of timeseries
    #println(observations)
    return observations
end

G_single(times, timeseries) = G_single(times, timeseries, :P)

function RunBoxModel(m; Δt = 0.05, stop_time = 50) #runs a box model; takes the model as values and returns the timeseries
    model = m
    output_path = pwd() * "/output/"*string(get_bgc(model)) * "_output.jld2"
    
    # Runs simulation 
    simulation = FastSimulation(model; Δt = Δt, stop_time = stop_time)
    run!(simulation, save_interval = 1, feedback_interval = Inf, save = SaveBoxModel(output_path), verbose = false)
    
    # Formats and processes output
    vars = keys(model.fields)
    file = jldopen(output_path)
    rounding = length(string(simulation.Δt))    

    times_u = parse.(Float64, keys(file["fields"]))
    times = round.(times_u; digits = rounding)

    timeseries = NamedTuple{vars}(ntuple(t -> zeros(length(times_u)), length(vars)))
    for (idx, time) in enumerate(times_u)
        values = file["fields/$time"]
        for tracer in vars
            getproperty(timeseries, tracer)[idx] = values[tracer]
        end
    end
    close(file)
    
    return times, timeseries
end
 
#  generates data from truth model with added noise
function generate_data(obj::EKPObject, true_params, n_samples::Int64, noise)
    @info "Generating samples..."
    start_t = now()

    G(u) = obj.observation(u)
    dim_output = length(G(true_params)) # dimension of output

    Γ = noise * Diagonal([1 for i in 1:dim_output])
    noise_dist = MvNormal(zeros(dim_output), Γ)

    yt = Vector{Observation}(undef, n_samples)

    for i in 1:n_samples #generates noisy samples of true system as a matrix of dimensions dim_output x n_samples
        yt[i] = Observation(G(scale_list(param_true, obj.input_scaling)) .+ rand(noise_dist), Γ, "$i")
        if i%100 == 0
            println(string("Reached ", i, " samples"))
        end
    end

    end_t = now()
    println("elapsed: " * string(end_t - start_t) * "\n")

    return ObservationSeries(
        Dict("observations" => yt, "minibatcher" => no_minibatcher())
    )
end

#   config for models
PAR⁰(t) = 60 * (1 - cos((t + 15days) * 2π / year)) * (1 / (1 + 0.2 * exp(-((mod(t, year) - 200days) / 50days)^2))) + 2
z = -10 # specify the nominal depth of the box for the PAR profile
PAR_func(t) = PAR⁰(t) * exp(0.2z) # Modify the PAR based on the nominal depth and exponential decay

clock = Clock(time = Float64(0))
grid = BoxModelGrid()
PAR = FunctionField{Center, Center, Center}(PAR_func, grid; clock)

LOBSTER_bgc = LOBSTER(; grid = BoxModelGrid(), light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
LOBSTER_model = BoxModel(; biogeochemistry = LOBSTER_bgc, clock)
set!(LOBSTER_model, NO₃ = 10.0, NH₄ = 0.1, P = 0.1, Z = 0.01)

param_names_LOBSTER = [
    :phytoplankton_preference, 
    :maximum_grazing_rate, 
    :grazing_half_saturation, 
    :light_half_saturation, 
    :nitrate_ammonia_inhibition, 
    :nitrate_half_saturation, 
    :ammonia_half_saturation, 
    :maximum_phytoplankton_growthrate, 
    :zooplankton_assimilation_fraction, 
    :zooplankton_mortality, 
    :zooplankton_excretion_rate, 
    :phytoplankton_mortality, 
    :small_detritus_remineralisation_rate, 
    :large_detritus_remineralisation_rate, 
    :phytoplankton_exudation_fraction, 
    :nitrification_rate, 
    :ammonia_fraction_of_exudate, 
    :ammonia_fraction_of_excriment, 
    :phytoplankton_redfield
]

npzd_bgc = NPZD(; grid = BoxModelGrid(), light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
npzd_model = BoxModel(; biogeochemistry = npzd_bgc, clock)
set!(npzd_model, N = 7.0, P = 0.01, Z = 0.05)

param_names_npzd = [
    :initial_photosynthetic_slope, 
    :base_maximum_growth, 
    :nutrient_half_saturation, 
    :base_respiration_rate, 
    :phyto_base_mortality_rate, 
    :maximum_grazing_rate, 
    :grazing_half_saturation, 
    :assimulation_efficiency, #restrain to be between 0 and 1
    :base_excretion_rate, 
    :zoo_base_mortality_rate, 
    :remineralization_rate
]
NPZD_lims = (assimulation_efficiency = [0, 1],)

true_bgc = npzd_bgc
true_model = npzd_model
param_names = param_names_npzd
constraints = NPZD_lims

param_true = values(get_params(true_bgc; params = param_names, float_only = false))
dim_input = length(param_true) # dimension of input

######################
#   generates a mvn for the prior guess
prior_cov = prior_noise^2 * Diagonal([i^2 for i in param_true])
prior_offset = MvNormal(zeros(dim_input), prior_cov)

#   guesses a random prior mean which is a mvn with mean true params and checks that it is within the bounds
prior_mean = param_true .+ rand(prior_offset)

if isdefined(Main, :constraints)
    for key in keys(constraints)
        idx = findfirst(isequal(key), param_names)
        val = abs(prior_mean[idx])
        lmin = constraints[key][1]
        lmax = constraints[key][2]
        if val < lmin 
            prior_mean[idx] = lmin + min(abs(lmin - val), abs((lmax - lmin)^2/(lmin-val)))
        elseif val > lmax
            prior_mean[idx] = lmax - min(abs(val-lmax), abs((lmax-lmin)^2/(val-lmax)))
        end
    end
end

#prior_mean = [1.52051e-6, 8.61683e-6, 1.38137, 2.79668e-7,6.6611e-8,3.90374e-5, 0.439785, 0.885452,1.02191e-7, 3.38861e-6, 1.08577e-6]
prior_std = priorstds .* prior_mean

#   declare EKP object with relevant parameters

NPZDEKP = EKPObject(; 
                model = npzd_model,
                G = G,
                Δt = 60minutes, 
                stop_time = 60000minutes, 
                mutable_vars = param_names_npzd, 
                iterations = 12, 
                prior_mean = prior_mean, 
                prior_std = prior_std,
                constraints = constraints
)
#=

LOBSTEREKP = EKPObject(;
            model = LOBSTER_model, 
            G = G,
            Δt = 30minutes, 
            stop_time = 30000minutes, 
            mutable_vars = param_names_LOBSTER, 
            iterations = 10, 
            prior_mean = prior_mean, 
            prior_std = prior_std)
=#
#######################

#generate 'truth' data with noise 
start_time = now()
truth = generate_data(NPZDEKP, param_true, 200, observation_noise)
EKP_result = optimise_parameters!(NPZDEKP, truth)

end_time = now()
elapsed = end_time - start_time

#   display results
final_ensemble = EKP_result.final_ensemble
best_params = EKP_result.best_params
best_model = EKP_result.best_model
error = EKP_result.errors
prior_mean = NPZDEKP.unparameterised_prior.mean
final_err = EKP_result.final_error

#=
println("------")
println("\nensemble size")
println(size(final_ensemble)[2])
println("\ntrue params")
display(pairs(NamedTuple{Tuple(param_names)}(param_true)))

println("------")
#println("\ninitial params")
#display(pairs(NamedTuple{Tuple(param_names)}(prior_mean)))
println("\nfinal params")
display(pairs(best_params))
println("------")
println("\nfinal error: " * string(final_err))
println("\nstd of error in each parameter")
display(pairs(error))
println("------")

=#

#plots true vs estimated final result
vals = RunBoxModel(true_model; Δt = NPZDEKP.Δt, stop_time = 5*NPZDEKP.stop_time)
times = vals[1]
timeseries = vals[2]

timeseries_est = RunBoxModel(best_model; Δt = NPZDEKP.Δt, stop_time = 5*NPZDEKP.stop_time)[2]

println("\ntotal time elapsed: " * string(elapsed))

display(plot_timeseries(times, 
                        remove_prescribed_tracers(true_model, timeseries), 
                        remove_prescribed_tracers(best_model, timeseries_est)))
