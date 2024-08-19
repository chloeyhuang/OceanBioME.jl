using OceanBioME, Oceananigans
import OceanBioME: BoxModelGrid
using OceanBioME.Models: NPZDModel, LOBSTERModel
using Oceananigans.Units
using Oceananigans.Fields: FunctionField

using Dates: now
using JLD2
using LinearAlgebra
using Statistics, Distributions, Random, StatsBase
using DataFrames
using ProfileView, BenchmarkTools

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

const EKP = EnsembleKalmanProcesses

include("EKPUtils_fs.jl")
include("CarbonChemistry_utils.jl")
include("glodap_cleaned_data.jl")

function generate_truth(obj::EKPObject, n_samples, G)
    @info "Generating samples..."
    start_t = now()
    
    data = obj.data
    pH_err = Normal(0.0, pH_error^2)
    pCO₂_err = Normal(0.0, pCO₂_error^2)

    true_pH = [dp.measurements.pH for dp in data]
    true_pCO₂ = [dp.measurements.pCO₂ for dp in data]
    
    n = length(true_pH)
    vh = var(true_pH)
    mh = mean(true_pH)
    vc = var(true_pCO₂)
    mc = mean(true_pCO₂)

    yt = Vector{Observation}(undef, n_samples)

    #   cov matrix of error
    Γ = diagm([#=1/n*vh, vh, vh, =#1/n*vc, vc, vc, 
    #=sqrt(1/n^3*(0.01 * (6.63-vh-mh^2)^2 + 4n*vh*(vh+mh^2)))   ,=# 
    sqrt(1/n^3*(0.01 * (6.63-vc-mc^2)^2 + 4n*vc*(vc+mc^2)))])
    #=Γ = Diagonal([1/n*vh, 
                    vh, 
                    1/n*vc, 
                    vc, 
                    sqrt(1/n^3*(0.01 * (6.63-vh-mh^2)^2 + 4n*vh*(vh+mh^2))), 
                    sqrt(1/n^3*(0.01 * (6.63-vc-mc^2)^2 + 4n*vc*(vc+mc^2)))]) 
    =#

    Threads.@threads for i in 1:(n_samples)
        if i%50 == 0
            println(string("Reached ", i, " samples"))
        end
        pH = true_pH .+ rand(pH_err)
        pCO₂ = true_pCO₂ .+ rand(pCO₂_err)
        yt[i] = Observation(scale_list(G(pH, pCO₂, true_pH, true_pCO₂), obj.output_scaling), Γ, "$i")
    end
    
    end_t = now()
    println("elapsed: " * string(end_t - start_t) * "\n")

    return ObservationSeries(
        Dict("observations" => yt, "minibatcher" => no_minibatcher())
    )
end

function generate_truth_mbatched(obj::EKPObject)
    data = obj.data
    n_samples = length(data)

    yt = Vector{Observation}(undef, n_samples)

    pH_err = Normal(0.0, pH_error^2)
    pCO₂_err = Normal(0.0, pCO₂_error^2)

    Γ = Diagonal([#=pH_error, =#pCO₂_error])
    
    Threads.@threads for i in 1:n_samples
        if i%50 == 0
            println(string("Reached ", i, " samples"))
        end
        #c = [data[i].measurements.pH + rand(pH_err), data[i].measurements.pCO₂ + rand(pCO₂_err)]
        yt[i] = Observation(scale_list([data[i].measurements.pCO₂ +  rand(pCO₂_err)], obj.output_scaling[1]), Γ, "$i")
    end

    return ObservationSeries(
        Dict("observations" => yt, "minibatcher" => RandomFixedSizeMinibatcher(obj.batch))
    )
end

@inline function G(pH, pCO₂, true_pH, true_pCO₂)
    pH_mean = mean(pH)
    pH_var = var(pH)
    pH_iqr = iqr(pH)

    pCO₂_mean = mean(pCO₂)
    pCO₂_var = var(pCO₂)
    pCO₂_iqr = iqr(pCO₂)

    n = length(pH)

    pH_rms_err =  sqrt(sum(abs2, (pH.-true_pH)))
    pCO₂_rms_err = sqrt(sum(abs2, (pCO₂.-true_pCO₂)))

    return[#=pH_mean, pH_var, pH_iqr, =#pCO₂_mean, pCO₂_var, pCO₂_iqr,#= pH_rms_err, =#pCO₂_rms_err]
end

@inline function G_model(model; data)
    true_pH = [dp.measurements.pH for dp in data]
    true_pCO₂ = [dp.measurements.pCO₂ for dp in data]

    pH = [model(; filter_pH_fields(dp.values)..., return_pH = true) for dp in data]
    pCO₂ = [model(; filter_pCO2_fields(dp.values)..., T = 20, P = 0) for dp in data]

    return G(pH, pCO₂, true_pH, true_pCO₂)
end

@inline function G_raw_model(model; data, batch)
    pH(i) = model(; filter_pH_fields(data[i].values)..., return_pH = true)
    pCO₂(i) = model(; filter_pCO2_fields(data[i].values)..., T = 20, P = 0)

    return [pCO₂(i) for i in batch]
end 

dr = get_cleaned_data()
d::Vector{@NamedTuple{values ::@NamedTuple{lat::Float64, lon::Float64, depth::Float64, T::Float64, S::Float64, DIC::Float64, Alk::Float64, P::Float64, silicate::Float64, phosphate::Float64}, measurements::@NamedTuple{pH::Float64, pCO₂::Float64}}} = dr

raw_data = load_object("output/glodap_cleaned_data.jld2")

priorstds = 0.005
pH_error = 0.01
pCO₂_error = 5.0 

excluded_vars = (:inverse_T, :log_T, :T²)

prior_mean = get_cc_params_raw(CarbonChemistry(); excluded_terms = excluded_vars, excluded_constants = (:ionic_strength, :calcite_solubility, :density_function))
prior_std = zeroinfcheck.(abs.(priorstds .* prior_mean), priorstds)

#dt = sample(d, 800; replace = false)

cc_ekp = CarbonChemistryEKPObject(; G = G_model, 
                                    data = d,
                                    excluded_vars,
                                    iterations = 15,
                                    prior_mean,
                                    #scale_output = false,
                                    prior_std)

truth = generate_truth(cc_ekp, 300, G)

println("-------------")

result = optimise_parameters!(cc_ekp, truth)
m = result.best_model
mb = result.final_model

display(plot_errors(d; model = m, ylims = [-100, 100]))

println("hello world!")

