using OceanBioME, Oceananigans
import OceanBioME: BoxModelGrid
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
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions
#using Peaks

include("PZ.jl") # Include the functions defined in PZ.jl
include("EKPUtils_fs.jl") #Includes G function + other required 

year = years = 365day
const EKP = EnsembleKalmanProcesses

# TO DO:
# - update code to use new box model - DONE
# - speed: are type inferences slowing down the process, optimise code etc
#       - set_model: 18ms (see line 143 utils.jl) - DONE
#       - set_bgc/get_param: trivial
#       - RunBoxModel: update_state! takes a long time - not my problem 
# - figure out optimal spread of std for guesses of means
# - implement EKP for nonhydrostatic model for LOBSTER/PISCES

# NEEDS TESTING: 
# - implement EKP for box model for NPZD/LOBSTER

#config for EKP, prior noise / stds proportional to mean guesses
prior_noise = 0.3
observation_noise = 0.001
priorstds = 0.3

function G(model; data = nothing, Δt = 0.05, stop_time = 50.0)
    out = RunBoxModel(model; Δt, stop_time)
    times = out[1]
    timeseries = out[2]
    n = length(times)
    observations = []
    #print(timeseries)
    for tracer in keys(timeseries)
        TS_max = maximum(timeseries[tracer])
        TS_min = minimum(timeseries[tracer])
        TS_rms = sqrt(1/n * sum(abs2, (timeseries[tracer].-TS_min)))
        TS_end = timeseries[tracer][end]
        T_max = argmax(timeseries[tracer])
        push!(observations, TS_max - TS_min, TS_rms, TS_end, T_max)
    end
    # observations returned: end - start, rms, last point of timeseries
    #println(observations)
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
    times = [round(times_u[i]; digits = rounding) for i in 1:length(times_u)]

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

 
function generate_data(obj::EKPObject, true_params, n_samples::Int64, noise) #generates data from truth model with added noise
    @info "Generating samples..."
    start_t = now()

    G(u) = obj.observation(u)
    dim_output = length(G(true_params)) # dimension of output

    Γ = noise * Diagonal([1 for i in 1:dim_output])
    noise_dist = MvNormal(zeros(dim_output), Γ)

    yt = zeros(dim_output, n_samples)

    for i in 1:n_samples #generates noisy samples of true system as a matrix of dimensions dim_output x n_samples
        yt[:,i] = G(scale_list(param_true, obj.input_scaling)) .+ rand(noise_dist)
        if i%100 == 0
            println(string("Reached ", i, " samples"))
        end
    end

    end_t = now()
    println("elapsed: " * string(end_t - start_t) * "\n")

    return Observations.Observation(yt, Γ, ["" for i in 1:dim_output])
end

#config for models
PAR⁰(t) = 60 * (1 - cos((t + 15days) * 2π / year)) * (1 / (1 + 0.2 * exp(-((mod(t, year) - 200days) / 50days)^2))) + 2
z = -10 # specify the nominal depth of the box for the PAR profile
PAR_func(t) = PAR⁰(t) * exp(0.2z) # Modify the PAR based on the nominal depth and exponential decay

clock = Clock(time = Float64(0))
grid = BoxModelGrid
PAR = FunctionField{Center, Center, Center}(PAR_func, grid; clock)


LOBSTER_bgc = LOBSTER(; grid = BoxModelGrid, light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
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

npzd_bgc = NPZD(; grid = BoxModelGrid, light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
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

#
pz_bgc = PhytoplanktonZooplankton()
pz_model = BoxModel(; biogeochemistry = pz_bgc)
set!(pz_model, P = 0.1, Z = 0.5)

param_names_pz = [
    :phytoplankton_growth_rate,
    :grazing_rate,
    :grazing_efficiency,
    :zooplankton_mortality_rate,
    :light_decay_length
    ]
#
true_bgc = npzd_model.biogeochemistry 
true_model = npzd_model
param_names = param_names_npzd
constraints = NPZD_lims

param_true = values(get_params(true_bgc; params = param_names, float_only = false))
dim_input = length(param_true) # dimension of input

######################
#generate a mvn for the prior guess
prior_cov = prior_noise^2 * Diagonal([i^2 for i in param_true])
prior_offset = MvNormal(zeros(dim_input), prior_cov)

 #guesses a random prior mean which is a mvn with mean true params
prior_mean = param_true .+ rand(prior_offset)

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
prior_mean = [1.52051e-6, 8.61683e-6, 1.38137, 2.79668e-7,6.6611e-8,3.90374e-5, 0.439785, 0.885452,1.02191e-7, 3.38861e-6, 1.08577e-6]
prior_std = priorstds*[prior_mean[i] for i in 1:length(param_true)]

#####################
# declare EKP object with relevant parameters
"""
PZEKP = EKPObject(pz_model, G; 
            Δt = 0.05, 
            stop_time = 50.0, 
            mutable_vars = param_names_pz, 
            iterations = 12, 
            prior_mean = prior_mean, 
            prior_std = prior_std)

            
######
"""

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

"""          
######
LOBSTEREKP = EKPObject;
            model = LOBSTER_model, 
            G = G,
            Δt = 30minutes, 
            stop_time = 30000minutes, 
            mutable_vars = param_names_LOBSTER, 
            iterations = 12, 
            prior_mean = prior_mean, 
            prior_std = prior_std)
"""
#######################

function run_ekp(obj::EKPObject)
    #generate 'truth' data with noise 
    start_time = now()
    truth = generate_data(obj, param_true, 200, observation_noise)

    ######################

    EKP_result = optimise_parameters!(obj, truth)

    end_time = now()
    elapsed = end_time - start_time

    ##################
    final_ensemble = EKP_result.final_ensemble
    final_params = EKP_result.final_params
    final_model = EKP_result.final_model
    error = EKP_result.errors
    prior_mean = obj.prior_mean
    final_err = EKP_result.final_error

    println("------")
    println("\nensemble size")
    println(size(final_ensemble)[2])
    println("\ntrue params")
    display(pairs(NamedTuple{Tuple(param_names)}(param_true)))

    println("------")
    println("\ninitial params")
    display(pairs(NamedTuple{Tuple(param_names)}(prior_mean)))
    println("\nfinal params")
    display(pairs(final_params))
    println("------")
    println("\nfinal error: " * string(final_err))
    println("\nstd of error in each parameter")
    display(pairs(error))
    println("------")

    #plots true vs estimated final result
    vals = RunBoxModel(true_model; Δt = obj.Δt, stop_time = 10*obj.stop_time)
    times = vals[1]
    timeseries = vals[2]

    timeseries_est = RunBoxModel(final_model; Δt = obj.Δt, stop_time = 10*obj.stop_time)[2]

    println("\ntotal time elapsed: " * string(elapsed))

    display(plot_timeseries(times, 
                            remove_prescribed_tracers(true_model, timeseries), 
                            remove_prescribed_tracers(final_model, timeseries_est)))
end

println("================\n")

truth = generate_data(NPZDEKP, param_true, 100, observation_noise)
result = optimise_parameters!(NPZDEKP, truth)
