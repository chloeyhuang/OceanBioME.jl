using OceanBioME, Oceananigans
import OceanBioME: BoxModelGrid
using Oceananigans.Units

using OceanBioME.Models: teos10_density, teos10_polynomial_approximation
using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi

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

include("EKP_Utils_CC.jl")
include("glodap_cleaned_data.jl")

const EKP = EnsembleKalmanProcesses

function generate_truth(obj::EKPObject_cc, n_samples, G)
    @info "Generating samples..."
    start_t = now()
    dt = obj.data

    pH_e = Normal(0.0, pH_error^2)
    pCO₂_e = Normal(0.0, pCO₂_error^2)
    pH_true = [dp.measurements.pH for dp in dt]
    pCO₂_true = [dp.measurements.pCO₂ for dp in dt]

    Gt(x, y) = G(x, y, pH_true, pCO₂_true)

    output = Gt(pH_true, pCO₂_true)
    dim_out = length(output)
    n = length(pH_true)
    vh = var(pH_true)
    mh = mean(pH_true)
    vc = var(pCO₂_true)
    mc = mean(pCO₂_true)

    yt = zeros(dim_out, n_samples)
    Γ = Diagonal([1/n*vh, vh, vh, 1/n*vc, vc, vc, 
    sqrt(1/n^3*(0.01 * (6.63-vh-mh^2)^2 + 4n*vh*(vh+mh^2))), 
    sqrt(1/n^3*(0.01 * (6.63-vc-mc^2)^2 + 4n*vc*(vc+mc^2)))])

    Threads.@threads for i in 1:(n_samples)
        if i%50 == 0
            println(string("Reached ", i, " samples"))
        end
        pH = [dp.measurements.pH + rand(pH_e) for dp in dt]
        pCO₂ = [dp.measurements.pCO₂ + rand(pCO₂_e) for dp in dt]
        yt[:, i] = scale_list(Gt(pH,pCO₂), obj.output_scaling)
    end
    
    end_t = now()
    println("elapsed: " * string(end_t - start_t) * "\n")
    display(yt)
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

    return [pH_mean, pH_var, pH_iqr, pCO₂_mean, pCO₂_var, pCO₂_iqr, pH_rms_err, pCO₂_rms_err]
end

raw_data = load_object("output/glodap_cleaned_data.jld2")
d = get_cleaned_data()

priorstds = 0.004
pH_error = 0.01
pCO₂_error = 5.0 

excluded = (:inverse_T, :log_T, :T²)

eqc = (K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi)
eqc_names = (:K0, :K1, :K2, :KB, :KW, :KS, :KF, :KP1, :KP2, :KP3, :KSi)
vals = Tuple([get_params(c(); excluded = excluded) for c in eqc])
defaults = NamedTuple{eqc_names}(vals)

prior_mean = get_cc_params_raw(CarbonChemistry())
prior_std = zeros(length(prior_mean))

for i in 1:length(prior_mean)
    std = abs(prior_mean[i] * priorstds)
    if std == 0
        prior_std[i] = priorstds^2
    else 
        prior_std[i] = std
    end
end

#println(join(round.(prior_mean; digits = 4), ", \n"))
cc_ekp = EKPObject_cc(G, d; excluded_vars = excluded, iterations = 15, prior_mean = prior_mean, prior_std = prior_std)
truth = generate_truth(cc_ekp, 500, G)

#display(truth)
#optimised_params = optimise_parameters_cc!(cc_ekp, truth)

#display(pairs(optimised_params.final_params))
#println("\noriginal:")
#display(pairs(defaults))
#display(optimised_params)
