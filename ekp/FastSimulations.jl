using Oceananigans: Clock, prettytime, CPU
using StructArrays, JLD2

import Oceananigans.Simulations: run!, Simulation
import Oceananigans: time_step!, set!

#export FastSimulation, run!, SaveBoxModel

@kwdef struct FastSimulation{ML, DT, ST}
             model :: ML
                Δt :: DT
         stop_time :: ST
    run_time_limit :: Float64
end

function FastSimulation(model; Δt = 0, stop_time = Inf, run_time_limit = Inf)
    if stop_time == Inf && run_time_limit == Inf
        @warn "This simulation will run forever as stop time " * "= run time limit = Inf."
    end
    if Δt == 0
        throw(ArgumentError("Δt must be larger than zero"))
    end
    return FastSimulation(model, Δt, stop_time, run_time_limit)
end

#########################

function run_test!(sim::FastSimulation; feedback_interval = 100000, save_interval = 1, save = nothing)
    @info("Starting simulation...")

    if save_interval == 0
        throw(ArgumentError("Save interval must be larger than zero"))
    end

    model = sim.model
    Δt = sim.Δt

    itter = 0 
    while model.clock.time < sim.stop_time
        itter % feedback_interval == 0 && @info "Reached $(prettytime(model.clock.time))"
        time_step_c!(model, Δt)
        itter += 1
        itter % save_interval == 0 && save(model)
    end

    if !isnothing(save)
        close(save.file)
    end
end

function run!(sim::FastSimulation; feedback_interval = 100000, save_interval = 1, save = nothing, verbose = true)
    if verbose == true
        @info("Starting simulation...")
    end

    if save_interval == 0
        throw(ArgumentError("Save interval must be larger than zero"))
    end

    model = sim.model
    Δt = sim.Δt
    itter = 0 

    while model.clock.time < sim.stop_time
        itter % feedback_interval == 0 && itter !== 0 && @info "Reached $(prettytime(model.clock.time))"
        itter % save_interval == 0 && save(model)
        time_step!(model, Δt)
        itter += 1
    end
    if verbose == true
        @info "Ending simulation at $(prettytime(model.clock.time))"
    end

    if !isnothing(save)
        close(save.file)
    end
end

function data_converter(model)
    data = model.fields
    vars = keys(model.fields)
    return NamedTuple{vars}(ntuple(i -> data[vars[i]][1], length(vars)))
end

struct SaveBoxModel{FP, F}
    filepath :: FP
    file :: F

    function SaveBoxModel(filepath::FP) where FP
        file = jldopen(filepath, "w")
        F = typeof(file)
        return new{FP, F}(filepath, file)
    end
end

(save::SaveBoxModel)(model) = save.file["fields/$(model.clock.time)"] = data_converter(model)

