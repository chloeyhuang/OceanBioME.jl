using CSV, DataFrames, EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions, Random, LinearAlgebra, Distributions
using EnsembleKalmanProcesses.Observations
using OceanBioME: CarbonChemistry
using OceanBioME.Models: teos10_density

# possible noise source described here: https://essd.copernicus.org/articles/13/5565/2021/essd-13-5565-2021-discussion.html

using OceanBioME.Models.CarbonChemistryModel: K0, K1, K2, KP, KS, KF, KB, KW, KSi, seawater_density

const EKP = EnsembleKalmanProcesses

# load data

file = CSV.read("GLODAPv2_pH_DIC_ALK_subset.csv", DataFrame)

T = file[(file[!,:cruise] .!= 270), :temperature]

S = file[(file[!,:cruise] .!= 270), :salinity]

density = [seawater_density(T[n], S[n]) for n in 1:length(T)]

depth = file[(file[!,:cruise] .!= 270), :depth]

DIC = file[(file[!,:cruise] .!= 270), :tco2] .* density * 1e-3

Alk = file[(file[!,:cruise] .!= 270), :talk] .* density * 1e-3

measured_pH = file[(file[!,:cruise] .!= 270), :phtsinsitutp]

#measured_phosphate = file[(file[!,:cruise] .!= 270), :phosphate] .* density * 1e-3

#measured_silicate = file[(file[!,:cruise] .!= 270), :silicate] .* density * 1e-3

# subset

subset = (depth.<Inf)#.&(S.>30)

T = T[subset]
S = S[subset]
DIC = DIC[subset]
Alk = Alk[subset]
measured_pH = measured_pH[subset]
#=
# construct the forward map

#testing_subset = rand(Bool, length(T)) .* rand(Bool, length(T))
training_subset = [!out for out in testing_subset]

function G(u; pco2 = false, test_set = false)
    carbonate_chemistry = CarbonChemistry(; #solubility = K0(u[1:7]...),
                                            carbonic_acid = (K1 = K1(u[(8:12).-7]...), K2 = K2(u[(13:17).-7]...)),
                                            #boric_acid = KB(u[(18:29).-7]...),
                                            water = KW(u[(18:24).-7]...))#u[(30:36).-7]...))

    pH = [carbonate_chemistry(DIC[n], Alk[n], T[n], S[n]; return_pH = !pco2, boron = S[n] * u[end]) for n in 1:length(T) if training_subset[n] ⊻ test_set]
    return pH
end

# construct the priors

prior_means = [#-162.8301, 218.2968 * 100, 90.9241, -1.47696 / 100^2, 0.025695, -0.025225 / 100, 0.0049867 / 100^2,
               62.008, -3670.7, -9.7944, 0.0118, -0.000116,
               -4.777, -1394.7, 0.0184, -0.000118, 0.0,
               #148.0248, -8966.90, -2890.53, -77.942, 1.728, -0.0996, 137.1942, 1.62142, -24.4344, -25.085, -0.2474, 0.053105,
               148.9652, -13847.26, -23.6521, -5.977, 118.67, 1.0495, -0.01615,
               0.000232 / 10.811 / 1.80655]


prior = combine_distributions([constrained_gaussian("u$n", mean, ifelse(mean == 0, 0.0001, abs(mean * 0.01)), ifelse(n == length(prior_means), 0, -Inf), Inf) for (n, mean) in enumerate(prior_means)])
# boron ratio can't be negative
# make the EKP

N_iterations = 100
N_ensemble = 4*1028

rng_seed = 42
rng = Random.MersenneTwister(rng_seed)

pH_precision = 0.01 # the origional model had an error of ~ -0.007 ± 0.020 so within instrument precisions anyway..., maybe if we can get the offset to 0?

y = measured_pH[training_subset]
Γ = pH_precision * I

n_samples = 64
#=
y_t = zeros(length(y), n_samples)

Γy = convert(Array, Diagonal(pH_precision .* ones(length(y))))
μ = zeros(length(y))

# Add noise
Threads.@threads for i in 1:n_samples
    i%10 == 0 && @info i
    y_t[:, i] = y .+ rand(MvNormal(μ, Γy))
end=#
#=
@info "Generated truth samples"

truth = Observations.Observation(y_t, Γy, ["" for n in 1:length(y)])
truth_sample = truth.mean

α_reg = 1.0
update_freq = 0

process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq)
ensemble_kalman_process = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process; failure_handler_method = SampleSuccGauss())
=#

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Sampler(mean(prior), cov(prior));#Inversion(); 
                                                    rng, failure_handler_method = IgnoreFailures())
# error = 488 after 10

                                   

G_ens = zeros(length(y), N_ensemble)#size(get_ϕ_final(prior, ensemble_kalman_process), 2))#

finite_or_zero(x) = ifelse(isfinite(x), x, 0)

# run the inversion
for i in 1:N_iterations
    @info "Running generation $i"
    params_i = get_ϕ_final(prior, ensemble_kalman_process)

    @info maximum(finite_or_zero, abs.(1 .-  get_ϕ_mean_final(prior, ensemble_kalman_process) ./ prior_means)), get_ϕ_mean_final(prior, ensemble_kalman_process)[10]

    Threads.@threads for i in 1:N_ensemble#size(params_i, 2)#
        G_ens[:, i] = G(params_i[:, i])
    end

    @info "updating ensemble"

    update_ensemble!(ensemble_kalman_process, G_ens, additive_inflation = true)

    @info get_error(ensemble_kalman_process)[end]
end

# REWRITE G for GPU!!!

#=
julia> mean(optimised_pH_final .- y)
0.000377271699448902

julia> mean(unoptimised_pH .- y)
-0.005011471476222631

julia> std(optimised_pH_final .- y)
0.02412175089864384

julia> std(unoptimised_pH .- y)
0.038421737360646344
=#=#