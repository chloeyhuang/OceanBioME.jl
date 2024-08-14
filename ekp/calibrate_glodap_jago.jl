using CSV, DataFrames, Random, CairoMakie
using OceanBioME: CarbonChemistry

using OceanBioME.Models: teos10_density, teos10_polynomial_approximation
using OceanBioME.Models.CarbonChemistryModel: IonicStrength, K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi


function plot_errors(; subset = :,
                       to_plot = pH_error,
                       title = "ΔpH",
                       color_name = S_name,
                       color_marker = data[:, color_name][subset],
                       limit_y = false,
                       ylim = 100)
    fig = Figure();

    ax = Axis(fig[1, 1], title = "DIC")#, aspect = DataAspect(), title = "Local grid")
    ax2 = Axis(fig[1, 2], title = "Alk")
    ax3 = Axis(fig[1, 3], title = "pH")
    ax4 = Axis(fig[2, 1], title = "T")
    ax5 = Axis(fig[2, 2], title = "S")
    ax6 = Axis(fig[2, 3], title = "Depth")

    sc=scatter!(ax, filter_column(data[:, DIC_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax2, filter_column(data[:, Alk_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax3, filter_column(data[:, pH_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax4, filter_column(data[:, T_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax5, filter_column(data[:, S_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax6, filter_column(data[:, depth_name][subset]), to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    
    if limit_y
        [ylims!(a, -ylim, ylim) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    end

    Colorbar(fig[1:2, 4], sc)

    Label(fig[0, :], title)

    return fig
end

filter_column(col) = [ifelse(v == -9999, NaN, v) for v in col]


if !(@isdefined T)||isnothing(T) # turns out this doesn't actually stop it reloading the data
    # get the glodap data - downloaded "merged" file from https://www.ncei.noaa.gov/data/oceans/ncei/ocads/data/0283442/
    data = CSV.read("validation/GLODAP.csv", DataFrame)

    DIC_name = "G2tco2"
    Alk_name = "G2talk"
    T_name = "G2temperature"
    S_name = "G2salinity"
    P_name = "G2pressure"
    Si_name = "G2silicate"
    PO_name = "G2phosphate"
    pH_name = "G2phtsinsitutp"
    depth_name = "G2depth"
    cruise_name = "G2cruise"
    fCO2_name = "G2fco2"
    lon_name = "G2longitude"
    lat_name = "G2latitude"

    DIC_name, Alk_name, T_name, S_name, P_name, Si_name, PO_name, pH_name

    # remove low quality data and non QCed
    # and drop a weird point from cruise with a massive errors (only 2 points)
    #=data = filter(row -> any([row[name*"f"] == 2 for name in [DIC_name, Alk_name, S_name]]) && 
                        !any([row[name] == -9999 for name in [DIC_name, Alk_name, T_name, S_name]]) &&
                        #row[cruise_name] != 5005 &&
                        #row[cruise_name] != 345 &&
                        row[Alk_name] > 1500 &&
                        row["G2phtsqc"] == 1,  data)
    =#
    # if we don't have Si and PO₄ the error gets a lot bigger but not in the surface values (as much)
    data = filter(row -> all([row[name*"f"] == 2 for name in [DIC_name, Alk_name, S_name]]) && 
                        !any([row[name] == -9999 for name in [DIC_name, Alk_name, T_name, S_name, P_name, Si_name, PO_name]]) &&
                        row["G2expocode"] != "325020210316" &&
                        row["G2expocode"] != "33RO20071215" &&
                        row[cruise_name] != 270 &&
                        row["G2phtsqc"] == 1 &&
                        # for this, refine to the ~10000 points with measured pCO2
                        row[fCO2_name*"f"] == 2 && 
                        row[fCO2_name*"temp"] != -9999 &&
                        row[depth_name] < 100, data)

    # setup the model
    base_carbon_chemistry = CarbonChemistry()

    # pre-rescale
    for row in eachrow(data)
        row[P_name] *= 0.1

        T = row[fCO2_name*"temp"]
        S = row[S_name]
        P = 0#row[P_name]

        lon = row[lon_name]
        lat = row[lat_name]

        density = base_carbon_chemistry.density_function(T, S, P, lon, lat)

        row[DIC_name] *= density * 1e-3
        row[Alk_name] *= density * 1e-3
        row[Si_name] *= density * 1e-3
        row[PO_name] *= density * 1e-3

        row[Si_name] = ifelse(row[Si_name] == -9999, 0, row[Si_name])
        row[PO_name] = ifelse(row[PO_name] == -9999, 0, row[PO_name])
    end
end

#####
##### Functions to turn a list of the equilibrium constants to default values and names
#####

@inline function constant_parameters(F)
    f = F()
    all_properties = propertynames(f)

    return [prop for prop in all_properties if getproperty(f, prop) isa Number]
end

@inline function constant_param_values(F)
    f = F()

    all_properties = propertynames(f)

    return [getproperty(f, prop) for prop in all_properties if getproperty(f, prop) isa Number]
end

#####
##### Turn a list of constant names and values into parameters
#####

@inline function params_to_constants(u, u_functions; all_functions = (K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi))
    any([!(f in all_functions) for f in u_functions]) && throw(ArgumentError("You need to specify `all_functions` to include more things"))
    # a helpful error I know

    f_names = nameof.(all_functions)
    nf = length(f_names)
    fs = Array{Any}(nothing, nf)

    current_index = 1

    for (fn, f) in enumerate(all_functions)
        if f in u_functions
            param_names = constant_parameters(f)

            end_index = current_index + length(param_names) - 1

            params = NamedTuple{tuple(param_names...)}(u[current_index:end_index])

            if f() isa KF
                KS_index = findfirst(f_names .== :KS)
                KF_index = findfirst(f_names .== :KF)
                if isnothing(KS_index) | KS_index > KF_index
                    throw(ArgumentError("KS needs to be defined before KF"))
                else
                    @inbounds fs[fn] = f(; sulfate_constant = fs[KS_index], params...)
                end
            else
                @inbounds fs[fn] = f(; params...)
            end

            current_index = end_index + 1
        else
            @inbounds fs[fn] = f()
        end
    end

    return NamedTuple{f_names}(fs)
end

#####
##### Compute the `i`th iteration
#####
function compute_pCO₂!(i, G, u, u_functions; T, S, DIC, Alk, silicate, phosphate)# we don't need lon/lat for the density approx
    IonicStrength in u_functions && throw(ArgumentError("This script doesn't work for calibrating the `IonicStrength` model"))

    constants = params_to_constants(u[:, i], u_functions)

    carbon_chemistry = CarbonChemistry(; solubility = constants.K0,
                                         carbonic_acid = (K1 = constants.K1, K2 = constants.K2),
                                         boric_acid = constants.KB,
                                         water = constants.KW,
                                         sulfate = constants.KS,
                                         fluoride = constants.KF,
                                         phosphoric_acid = (KP1 = constants.KP1, KP2 = constants.KP2, KP3 = constants.KP3),
                                         silicic_acid = constants.KSi)

    try 
        for n in 1:size(G, 1)
            # fCO2 is reported at no pressure and at a different temp
            @inbounds(G[n, i] = carbon_chemistry(; DIC = DIC[n], Alk = Alk[n], T = T[n], S = S[n], silicate = silicate[n], phosphate = phosphate[n]))
            @inbounds(isinf(G[n, i]) && (G[n, i] = NaN))
        end
    catch # some parameter values will make the problem un-solvable and if one is NaN they all will be neglected anyway
        G[:, i] .= NaN
    end
    return nothing
end

#####
##### Setup the calibration
#####
constants_to_calibrate = (K0, K1, K2, KB, KW, KS, KF, KP1, KP2, KP3, KSi)

using EnsembleKalmanProcesses, EnsembleKalmanProcesses.ParameterDistributions, Random, LinearAlgebra, Distributions
using EnsembleKalmanProcesses.Observations

const EKP = EnsembleKalmanProcesses

prior_means = vcat([constant_param_values(f) for f in constants_to_calibrate]...)
prior_names = vcat([constant_parameters(f) for f in constants_to_calibrate]...)
prior = combine_distributions([constrained_gaussian("u$n", mean, ifelse(mean == 0, 0.00001, abs(mean * 0.01)), ifelse(prior_names[n] in (:log_S, :log_S_KS), -1/40, -Inf), Inf) for (n, mean) in enumerate(prior_means)])

N_iterations = 30#15
#N_ensemble = length(prior_means) ^ 2

#rng_seed = 42
rng = Random.MersenneTwister()#rng_seed)

pCO₂_precision = 0.1#2.0#5.0 # from their crossover analysis in https://essd.copernicus.org/articles/16/2047/2024/essd-16-2047-2024.pdf fig 3?

y = data[:, fCO2_name]
println(length(y))

#=
Γ = pCO₂_precision * I

initial_ensemble = EKP.construct_initial_ensemble(rng, prior, N_ensemble)
ensemble_kalman_process = EKP.EnsembleKalmanProcess(initial_ensemble, y, Γ, Inversion(); 
                                                    rng, failure_handler_method = SampleSuccGauss())



n_samples = 500

y_t = zeros(length(y), n_samples)

Γy = convert(Array, Diagonal(pCO₂_precision .* ones(length(y))))
μ = zeros(length(y))

noise = MvNormal(μ, Γy)

# Add noise
Threads.@threads for i in 1:n_samples
    y_t[:, i] = y .+ rand(noise)
end

@info "Generated truth samples"

truth = Observations.Observation(y_t, Γy, ["" for n in 1:length(y)])
truth_sample = truth.mean

α_reg = 0.05#1e-3#0.2
update_freq = 2

process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq, prior_mean = mean(prior), prior_cov = cov(prior))
ensemble_kalman_process = EnsembleKalmanProcess(truth_sample, truth.obs_noise_cov, process; failure_handler_method = SampleSuccGauss())

finite_or_zero(x) = ifelse(isfinite(x), x, 0)

T = data[:, fCO2_name*"temp"]
S = data[:, S_name]

DIC = data[:, DIC_name]
Alk = data[:, Alk_name]
        
silicate = data[:, Si_name]
phosphate = data[:, PO_name]

G_ens = zeros(length(y), size(get_ϕ_final(prior, ensemble_kalman_process), 2))

@info size(G_ens)

initial = zeros(length(y), 1)

compute_pCO₂!(1, initial, reshape(prior_means, length(prior_means), 1), constants_to_calibrate; T, S, DIC, Alk, silicate, phosphate)

@info "Priors result in ΔpCO₂ = ($(mean(initial.-y)) ± $(std(initial.-y))) μatm"

# run the inversion
for it in 1:N_iterations
    @info "Running generation $it"
    params_i = get_ϕ_final(prior, ensemble_kalman_process)

    @info mean(finite_or_zero, abs.(1 .-  get_ϕ_mean_final(prior, ensemble_kalman_process) ./ prior_means))

    for i in 1:size(params_i, 2)#N_ensemble
        compute_pCO₂!(i, G_ens, params_i, constants_to_calibrate; T, S, DIC, Alk, silicate, phosphate)
    end

    @info "updating ensemble"

    update_ensemble!(ensemble_kalman_process, G_ens)#, additive_inflation = true)

    @info "ΔpCO₂ = ($(mean(G_ens.-y)) ± $(std(G_ens.-y))) μatm"
end

# α = 0.2, σ = 2, works well for (K0, K1, K2, KB, KW)
# can we learn parameters to make it work just as well without PO4 and Si as with?

# α = 0.1, update_freq = 2, returns almost exactly the same error for all of the parameters

function plot_ensemble(prior, ekp, y; T = T, S = S, DIC = DIC, Alk = Alk, silicate = silicate, phosphate = phosphate, all_results = nothing, name = "pCO₂_optimisation.mp4")
    N_its = length(ekp.u)
    N_members = size(ekp.u[1].stored_data, 2)

    fig = Figure(size=(1000, 500))

    range = (minimum(y), maximum(y))

    ax1 = Axis(fig[1, 1], limits = (range..., range...))
    ax2 = Axis(fig[1, 2], limits = (range..., -50, 50))

    initial = zeros(length(y), 1)

    compute_pCO₂!(1, initial, reshape(prior_means, size(ekp.u[1].stored_data, 1), 1), constants_to_calibrate; T, S, DIC, Alk, silicate, phosphate)

    errorbars!(ax1, y, initial[:, 1], fill(pCO₂_precision, length(y)), color = :grey, direction = :x)
    lines!(ax1, [range...], [range...], color = :black, linestyle = :dash)
    errorbars!(ax2, y, initial[:, 1] .- y, fill(pCO₂_precision, length(y)), color = :grey, direction = :x)

    if isnothing(all_results)
        all_results = zeros(length(y), N_members, N_its)

        for it in 1:N_its
            gen_results = zeros(length(y), N_members)
            Threads.@threads for n in 1:N_members
               compute_pCO₂!(n, gen_results, get_ϕ(prior, ekp, it), constants_to_calibrate; T, S, DIC, Alk, silicate, phosphate)
            end
            all_results[:, :, it] .= gen_results
        end
    end

    n = Observable(1)

    generation_mean = @lift mean(all_results[:, :, $n], dims = 2)[:, 1]
    generation_std  = @lift std(all_results[:, :, $n], dims = 2)[:, 1]

    generation_mean_error = @lift mean(all_results[:, :, $n] .- y, dims = 2)[:, 1]
    generation_std_error  = @lift std(all_results[:, :, $n] .- y, dims = 2)[:, 1]

    errorbars!(ax1, y, generation_mean, generation_std, color = :black)
    errorbars!(ax2, y, generation_mean_error, generation_std_error, color = :black)
    scatter!(ax1, y, generation_mean, markersize = 3, color = S)
    scatter!(ax2, y, generation_mean_error, markersize = 3, color = S)

    record(fig, name, 1:N_its; framerate = 2) do i; 
        @info i
        n[] = i
    end
    
    return fig
end

function plot_errors(; subset = :,
                       to_plot = pH_error,
                       title = "ΔpH",
                       color = S,
                       limit_y = false,
                       ylim = 100,
                       DIC = DIC, Alk = Alk, T = T, S = S, pH = data[:, pH_name], depth = data[:, depth_name],
                       color_marker = color[subset])
    fig = Figure();

    ax = Axis(fig[1, 1], title = "DIC")#, aspect = DataAspect(), title = "Local grid")
    ax2 = Axis(fig[1, 2], title = "Alk")
    ax3 = Axis(fig[1, 3], title = "pH")
    ax4 = Axis(fig[2, 1], title = "T")
    ax5 = Axis(fig[2, 2], title = "S")
    ax6 = Axis(fig[2, 3], title = "Depth")

    sc=scatter!(ax, DIC, to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax2, Alk, to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax3, pH[subset], to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax4, T[subset], to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax5, S[subset], to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    scatter!(ax6, depth[subset], to_plot[subset], markersize = 3, alpha = 0.5, color = color_marker)
    
    if limit_y
        [ylims!(a, -ylim, ylim) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    end

    Colorbar(fig[1:2, 4], sc)

    Label(fig[0, :], title)

    return fig
end =#