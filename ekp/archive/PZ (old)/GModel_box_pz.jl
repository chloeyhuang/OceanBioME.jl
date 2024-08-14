using OceanBioME
using OceanBioME: BoxModel
using Oceananigans

using JLD2
using Plots
using Statistics

using Distributions  # probability distributions and associated functions
using LinearAlgebra
using StatsPlots
using Random
using Dates

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
using EnsembleKalmanProcesses.ParameterDistributions

import Logging
Logging.disable_logging(Logging.Info)

export G
export box_model
export model_params

path = "/Users/huangc/Documents/GitHub/EKP_CYH/PZ/output"

include("PZ.jl") # Include the functions defined in PZ.jl

function model_params(list)
    if length(list) !== 5
        throw("incorrect number of vars")
    else 
        return PhytoplanktonZooplankton(
    phytoplankton_growth_rate = list[1], 
                 grazing_rate = list[2], 
           grazing_efficiency = list[3],  
   zooplankton_mortality_rate = list[4], 
           light_decay_length = list[5]
        )
    end
end

function box_model(u)
    @inline PAR(t) = 1  # or sin(t), etc.
    biogeochemistry = model_params(u)
    # @info biogeochemistry
    
    model = BoxModel(; biogeochemistry)

    # Set the initial conditions
    set!(model, P = 0.1, Z = 0.5)

    # ## Run the model and save the output every save_interval timesteps (should only take a few seconds)
    simulation = Simulation(model; Δt = 0.05, stop_time = 50)
    #pop!(simulation.callbacks, :nan_checker)
    simulation.output_writers[:fields] = JLD2OutputWriter(model, model.fields; dir = path, filename = "box_pz.jld2", schedule = TimeInterval(0.05), overwrite_existing = true)

    @info "Running the model..."
    run!(simulation)

    # Now, read the saved output and plot the results
    filepath = path * "/box_pz.jld2"
    times = FieldTimeSeries(filepath, "P").times
    timeseries = NamedTuple{keys(model.fields)}(FieldTimeSeries(filepath, "$field")[1, 1, 1, :] for field in keys(model.fields))    
    return times, timeseries
end

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

function G(u) #define the outputs of the 'observations' from diff input of vars
    # Define a function form for the photosynthetically available radiation (PAR), which modulates the growth rate
    data = box_model(u)
    timeseries = data[2]
    times = data[1]
    n = length(timeseries.P)
    # observations returned: P_min, P_max, P_mean, Z_min, Z_max, Z_mean
    P_max = maximum(timeseries.P)
    P_min = minimum(timeseries.P)
    P_rms = sqrt(1/n * sum(abs2, timeseries.P))

    Z_max = maximum(timeseries.Z)
    Z_min = minimum(timeseries.Z)
    Z_rms = sqrt(1/n * sum(abs2, timeseries.Z))
    #period = max_period(times, timeseries.P)
    return [P_min, P_max, P_rms, Z_min, Z_max, Z_rms]
end
##########################################################################################

true_model = PhytoplanktonZooplankton()
#get param names / true vals 
param_names = []
param_true = []
    
Tm = typeof(true_model)
for (name, typ) in zip(fieldnames(Tm), Tm.types)
    if typ == Float64 && name !== :light_amplitude
    push!(param_names, "$name")
    push!(param_true, getfield(true_model, name))
    end
end 
start_time = now()
print(box_model(param_true)[2].P)
end_time = now()

elapsed = end_time - start_time
println("\nelapsed: " * string(elapsed))
