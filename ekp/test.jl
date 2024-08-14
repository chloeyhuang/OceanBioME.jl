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
using EnsembleKalmanProcesses.Observations
using EnsembleKalmanProcesses.DataContainers
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
    
    Gt(x, y) = G(x, y, true_pH, true_pCO₂)

    output = Gt(true_pH, true_pCO₂)
    dim_out = length(output)
    n = length(true_pH)
    vh = var(true_pH)
    mh = mean(true_pH)
    vc = var(true_pCO₂)
    mc = mean(true_pCO₂)

    yt = zeros(dim_out, n_samples)

    #   cov matrix of error
    Γ = Diagonal([1/n*vh, vh, vh, 1/n*vc, vc, vc, 
    sqrt(1/n^3*(0.01 * (6.63-vh-mh^2)^2 + 4n*vh*(vh+mh^2))), 
    sqrt(1/n^3*(0.01 * (6.63-vc-mc^2)^2 + 4n*vc*(vc+mc^2)))])
    #=Γ = Diagonal([1/n*vh, 
                    vh, 
                    1/n*vc, 
                    vc, 
                    sqrt(1/n^3*(0.01 * (6.63-vh-mh^2)^2 + 4n*vh*(vh+mh^2))), 
                    sqrt(1/n^3*(0.01 * (6.63-vc-mc^2)^2 + 4n*vc*(vc+mc^2)))]) =#

    Threads.@threads for i in 1:(n_samples)
        if i%50 == 0
            println(string("Reached ", i, " samples"))
        end
        pH = true_pH .+ rand(pH_err)
        pCO₂ = true_pCO₂ .+ rand(pCO₂_err)
        yt[:, i] = scale_list(Gt(pH,pCO₂), obj.output_scaling)
    end
    
    end_t = now()
    println("elapsed: " * string(end_t - start_t) * "\n")
    #display(yt)

    return Observation(yt, Γ, ["" for n in 1:dim_out])
end

function G(pH, pCO₂, true_pH, true_pCO₂)
    pH_mean = mean(pH)
    pH_var = var(pH)
    pH_iqr = iqr(pH)

    pCO₂_mean = mean(pCO₂)
    pCO₂_var = var(pCO₂)
    pCO₂_iqr = iqr(pCO₂)

    n = length(pH)

    pH_rms_err =  sqrt(sum(abs2, (pH.-true_pH)))
    pCO₂_rms_err = sqrt(sum(abs2, (pCO₂.-true_pCO₂)))

    return[pH_mean, pH_var, pH_iqr, pCO₂_mean, pCO₂_var, pCO₂_iqr, pH_rms_err, pCO₂_rms_err]
end

function G_model(model; data)
    true_pH = [dp.measurements.pH for dp in data]
    true_pCO₂ = [dp.measurements.pCO₂ for dp in data]

    pH = [model(; filter_nt_fields([:depth], dp.values)..., return_pH = true) for dp in data]
    pCO₂ = [model(; filter_nt_fields([:depth, :T, :P], dp.values)..., T = 20, P = 0) for dp in data]
    
    return G(pH, pCO₂, true_pH, true_pCO₂)
end

d = get_cleaned_data()
raw_data = load_object("output/glodap_cleaned_data.jld2")

priorstds = 0.003
pH_error = 0.01
pCO₂_error = 5.0 

excluded_vars = (:inverse_T, :log_T, :T²)

prior_mean = get_cc_params_raw(CarbonChemistry(); excluded_terms = excluded_vars)
prior_std = zeroinfcheck.(abs.(priorstds .* prior_mean), priorstds)

cc_ekp = CarbonChemistryEKPObject(; G = G_model, 
                                    data = d,
                                    excluded_vars,
                                    iterations = 20,
                                    prior_mean,
                                    prior_std)

truth = generate_truth(cc_ekp, 300, G)

result = optimise_parameters!(cc_ekp, truth)
m = result.best_model

display(plot_errors(d; model = m, ylims = [-100, 100]))
println("hello world!")