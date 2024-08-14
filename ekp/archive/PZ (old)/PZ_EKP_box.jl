# This script runs the phytoplankton-zooplankton (PZ) model as a 0-D box model using OceanBioME
using OceanBioME

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
#using Peaks

import Logging
Logging.disable_logging(Logging.Info)

include("PZ.jl") # Include the functions defined in PZ.jl
include("GModel_box_pz.jl") #Include G Function (output)

const EKP = EnsembleKalmanProcesses

prior_noise = 0.2
observation_noise = 0.05
priorstds = 0.05

true_model = PhytoplanktonZooplankton(phytoplankton_growth_rate = 1.0, grazing_efficiency = 0.7)
# To override any of the default parameters, do something like this:
# PhytoplanktonZooplankton(phytoplankton_growth_rate = 0.5)

######## TO DO 
# - implement failure_handler for EKP 

# generate a random cov matrix 
function rand_cov(n::Int)
    X = rand(n,n)
    A = X'*X
    return A
end 
nothing

start_time = now()

#get param names / true vals 
param_names = []
param_true = []
 
Tm = typeof(true_model)
for (name, typ) in zip(fieldnames(Tm), Tm.types)
    if typ == Float64 && name !== :light_amplitude
    push!(param_names, "$name")
    push!(param_true, getfield(true_model, name))
    end
end # what to do with sinking velocity/vars that are a function: try to parameterise 

#generate a noisy sample of the true system 
dim_output = length(G(param_true)) # dimension of output
dim_input = length(param_true)# dimension of input
n_samples = 300

Γ = observation_noise * Diagonal([1 for i in 1:dim_output])
noise_dist = MvNormal(zeros(dim_output), Γ)
prior_offset = MvNormal(zeros(dim_input), prior_noise*I)

yt = zeros(length(G(param_true)), n_samples)

for i in 1:n_samples
    yt[:,i] = G(param_true) .+ rand(noise_dist) #generate noisy samples of true system 
end
data_names = ["P_min", "P_max", "P_rms","Z_min", "Z_max", "Z_rms"]

truth = Observations.Observation(yt, Γ, data_names)
truth_sample = truth.mean

prior_means_negative = param_true .+ rand(prior_offset) #guesses for prior means
prior_means = [abs(prior_means_negative[i]) for i in 1:length(prior_means_negative)]
prior_stds = [priorstds for i in 1:length(param_true)] #guesses for prior stds
prior_dis = [constrained_gaussian(param_names[i], prior_means[i], prior_stds[i], 0, Inf) for i in 1:length(param_names) ]
prior = combine_distributions(prior_dis)

prior_mean = mean(prior)
prior_cov = cov(prior)

##############

α_reg =  1.0
update_freq = 0
N_iter = 12

process = Unscented(prior_mean, prior_cov; α_reg = α_reg, update_freq = update_freq)
uki_obj = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process, failure_handler_method = SampleSuccGauss())

err = zeros(N_iter)

println("EKP starting....")

for i in 1:N_iter
    println("iteration " * str(i) * " start")
    params_i = get_ϕ_final(prior, uki_obj)
    J =  size(params_i)[2]

    G_ens = zeros(Float64, dim_output, J)
    Threads.@threads for j in 1:J
        G_ens[:, j] = G(params_i[:, j]) 
    end
    #display(G_ens)
    
    EKP.update_ensemble!(uki_obj, G_ens)

    err[i] = get_error(uki_obj)[end]
    println(
    "Iteration: " * string(i) *
    ", Error: " * string(err[i]) *
    " norm(Cov):" * string(norm(uki_obj.process.uu_cov[i])) * "\n")
    
end

final_ensemble = get_ϕ_final(prior, uki_obj)
final_params = get_ϕ_mean_final(prior, uki_obj)
final_cov = get_u_cov_final(uki_obj)

println("------")
println("initial params")
println(prior_means)
println("\nfinal params")
println(final_params)
println("\ntrue params")
println(param_true)
println("------")

#plots true vs estimated final result
vals = box_model(param_true)
times = vals[1]

timeseries = vals[2]
timeseries_est = box_model(final_params)[2]

plt1 = plot(times, timeseries.P, linewidth = 2, xlabel = "time", ylabel = "P, Z", legend = :outertopleft, label = "P");
plot!(plt1, times,timeseries.Z, linewidth = 2, label="Z");
plot!(plt1, times, timeseries_est.P, linewidth = 2, label = "P_est", color = :lightcyan3);
plot!(plt1, times, timeseries_est.Z, linewidth = 2, label = "Z_est", color = :pink3);
plt2 = plot(timeseries.P, timeseries.Z, linewidth = 2, xlabel="P", ylabel="Z", linecolor = :black, legend = :none);
plot!(plt2, timeseries_est.P, timeseries_est.Z, linewidth = 2, linecolor = :grey, legend = :none);

display(plot(plt1, plt2, layout = (1,2), size = (1400, 600)))

end_time = now()

elapsed = end_time - start_time
println("\nelapsed: " * string(elapsed))

