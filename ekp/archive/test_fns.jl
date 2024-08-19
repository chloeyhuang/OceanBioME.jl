#export box_model_test_PZ, box_model_test_NPZD, box_model_test_LOBSTER, run!, G, box_model, model_params, FastSimulation, SaveBoxModel
using OceanBioME: NPZDModel, LOBSTERModel
using Oceananigans.Units

include("Utils.jl")
include("EKPUtils.jl")

year = years = 365days
nothing 

function box_model_test_NPZD(; output = false)
    st = now()
    set_filepath!("output/box_nzpd.jld2")

    @inline PAR⁰(t) = 60 * (1 - cos((t + 15days) * 2π / 365days)) * (1 / (1 + 0.2 * exp(-((mod(t, 365days) - 200days) / 50days)^2))) + 2

    z = -10 # nominal depth of the box for the PAR profile
    @inline PAR(t)::Float64 = PAR⁰(t) * exp(0.2z) # Modify the PAR based on the nominal depth and exponential decay 

    bgc = NPZD(; grid = BoxModelGrid, light_attenuation_model = nothing)
    model = BoxModel(; biogeochemistry = bgc, forcing = (; PAR))

    # Set the initial conditions
    set!(model, N = 10.0, P = 0.1, Z = 0.01)
    et = now()
    println("moddle setting elapsed: " * string(et - st))

    start_time = now()
    @info "Running the model..."
    
    data = RunBoxModel(model; Δt = 20minutes, stop_time = 20000minutes)
    times = data[1]
    timeseries = data[2]
    end_time = now()

    if output == true
        return times, timeseries, "elapsed: " * string(end_time - start_time)
    end
    return "elapsed: " * string(end_time - start_time)
end

function box_model_test_PZ(; output = false)
    set_filepath!("output/box_pz.jld2")

    biogeochemistry = PhytoplanktonZooplankton()
    model = BoxModel(; biogeochemistry = bgc, timestepper = :RungeKutta3)

    # Set the initial conditions
    set!(model, P = 0.1, Z = 0.5)

    start_time = now()
    @info "Running the model..."
    
    data = RunBoxModel(model; Δt = 0.05, stop_time = 50.0)
    times = data[1]
    timeseries = data[2]

    end_time = now()
    
    if output == true
        return times, timeseries, "elapsed: " * string(end_time - start_time)
    end
    return "elapsed: " * string(end_time - start_time)
end

function box_model_test_LOBSTER(; output = false)
    set_filepath!("output/box_LOBSTER.jld2")
    
    PAR⁰(t) = 60 * (1 - cos((t + 15days) * 2π / year)) * (1 / (1 + 0.2 * exp(-((mod(t, year) - 200days) / 50days)^2))) + 2
    z = -10 # specify the nominal depth of the box for the PAR profile
    PAR(t) = PAR⁰(t) * exp(0.2z) # Modify the PAR based on the nominal depth and exponential decay
    
    bgc = LOBSTER(; grid = BoxModelGrid, light_attenuation_model = PrescribedPhotosyntheticallyActiveRadiation(PAR))
    model = BoxModel(; biogeochemistry = LOBSTER_bgc, clock)
    set!(model, NO₃ = 10.0, NH₄ = 0.1, P = 0.1, Z = 0.01)

    start_time = now()
    @info "Running the model..."

    data = RunBoxModel(model; Δt = 20minutes, stop_time = 20000minutes)
    times = data[1]
    timeseries = data[2]

    end_time = now()

    if output == true
        return times, timeseries, "elapsed: " * string(end_time - start_time)
    end
    return "elapsed: " * string(end_time - start_time)
end

#precompile(box_model_test_PZ(), (Any, ))
#precompile(box_model_test_NPZD(), (Any, ))
#precompile(box_model_test_LOBSTER(), (Any, ))
