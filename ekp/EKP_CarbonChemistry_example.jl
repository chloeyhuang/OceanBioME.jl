using OceanBioME, Oceananigans
using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KF, KB, KW, KS, KP1, KP2, KP3, KSi

using Dates: now
using JLD2
using LinearAlgebra
using Statistics, Distributions, Random, StatsBase
using DataFrames
using ProfileView, BenchmarkTools

using EnsembleKalmanProcesses
using EnsembleKalmanProcesses.ParameterDistributions

include("EKPUtils.jl")
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

    unscaled_cov = 2000*[1, 1, 5, 5, 5]
    #   cov matrix of error
    Γ = diagm(scale_list(unscaled_cov, obj.output_scaling))
    
    #=
    Γ = Diagonal([1/n*vh, 
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
    pH_error_v = pH .- true_pH
    pCO₂_error_v = pCO₂ .- true_pCO₂

    pH_MAE= mean(abs.(pH_error_v))
    pCO₂_MAE = mean(abs.(pCO₂_error_v))

    #=
    pH_var_err = var(pH_error_v)
    pCO₂_var_err = var(pCO₂_error_v)

    pH_iqr_err = iqr(pH_error_v)
    pCO₂_iqr_err = iqr(pCO₂_error_v)

    pH_mean = mean(pH)
    pH_var = var(pH)
    pH_iqr = iqr(pH)

    pCO₂_mean = mean(pCO₂)
    pCO₂_var = var(pCO₂)
    pCO₂_iqr = iqr(pCO₂)
    =#

    pCO2_large_err_count = sum(abs.(pCO₂_error_v) .> 30)
    pH_MSE =  mean(sum(abs2, (pH.-true_pH)))
    pCO₂_MSE = mean(sum(abs2, (pCO₂.-true_pCO₂)))

    return  [pH_MAE, 
            pH_MSE,
            pCO₂_MAE, 
            pCO₂_MSE,
            pCO2_large_err_count]
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

d = get_cleaned_data()

raw_data = load_object("output/glodap_cleaned_data.jld2")

priorstds = 0.0003
pH_error = 0.01
pCO₂_error = 5.0 

excluded_vars = (:inverse_T, :log_T, :T²)
excluded_constants = setdiff(propertynames(CarbonChemistry()), (:solubility, :carbonic_acid, :phosphoric_acid))
excluded_eqns = setdiff((:K0, :K1, :K2, :KB, :KW, :KS, :KF, :KP1, :KP2, :KP3, :KSi), (:K0, :K1, :K2, :KP1, :KP2, :KP3))

prior_mean = get_cc_params_raw(CarbonChemistry(); excluded_terms = excluded_vars, excluded_constants = excluded_constants)
prior_std = zeroinfcheck.(abs.(priorstds .* prior_mean), priorstds)

#dt = sample(d, 800; replace = false)

cc_ekp = CarbonChemistryEKPObject(; G = G_model, 
                                    data = d,
                                    excluded_vars,
                                    excluded_eqns,
                                    iterations = 10,
                                    prior_mean,
                                    #scale_output = false,
                                    prior_std)

truth = generate_truth(cc_ekp, 300, G)

result = optimise_parameters!(cc_ekp, truth)
m = result.best_model
mb = result.final_model

println("Original error:")
get_model_error(CarbonChemistry(), d; verbose = true)

println("\nNew error:")
display(plot_errors(d; model = m, ylims = [-100, 100], verbose = true))

display(plot_errors(d; ylims = [-100, 100]))

println("")