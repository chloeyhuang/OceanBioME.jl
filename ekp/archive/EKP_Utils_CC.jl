using EnsembleKalmanProcesses.Observations

include("Utils.jl")
include("FastSimulations.jl")
using CairoMakie


function calculate_observations_cc(G, u, data; input_scaling, output_scaling, excluded_vars)
    rescaled_u = scale_list(u, -input_scaling)
    nm = set_model_cc(rescaled_u, excluded_vars)
    pH = []
    pCO₂ = []
    try 
        for dp in data
            conditions = filter_nt_fields([:depth], merge(dp.loc, dp.values))
            push!(pH, nm(; conditions..., return_pH = true))
            push!(pCO₂, nm(; conditions...))
        end

        if isnothing(output_scaling)
            return (G(pH, pCO₂))
        else 
            scaled_output = scale_list(G(pH, pCO₂), output_scaling)
            return scaled_output
        end
    catch 
        @warn "params returned a root error in the CarbonChemistry model" 
        return NaN
    end
end
##########################################################################################

#defines a struct that contains all the things needed to perform EKP optimisation
@kwdef struct EKPObject_cc{FT, MD, NT, D}
           base_model :: MD
                   Δt :: FT = 0.05
            stop_time :: FT = 50
         excluded_vars = [] 
           iterations :: Int64 = 12

   initial_conditions :: NT
                prior :: D
           prior_mean = []
        input_scaling = []
       output_scaling = []

                 data = nothing
         observations :: Function
          RunBoxModel :: Function
end

function EKPObject_cc(G, data; excluded_vars, iterations, prior_mean, prior_std)
    st = now()

    pH_true = [dp.measurements.pH for dp in data]
    pCO₂_true = [dp.measurements.pCO₂ for dp in data]
    
    broken_constraints= [1, 3, 7, 8, 10, 12, 13, 15, 18, 19, 22, 25, 28, 29, 30, 33, 36, 42, 43, 44, 47,48, 50, 53, 54, 57, 58, 61]

    @info "Simulation parameters are 
            \niterations = " * string(iterations) * ", 
            \nexcluded variables = " * join(excluded_vars, ", ") * "\n"

    input_scaling = scale_parameters(zeroinfcheck.(prior_mean, 0))[2]
    scaled_mean = zeroinfcheck.(scale_parameters(prior_mean)[1], 0.0)
    scaled_std = zeroinfcheck.(scale_list(prior_std, input_scaling), 0.001^2)
    lims = zeros(length(scaled_mean), 2)
    
    for i in eachindex(scaled_mean)
        if scaled_mean[i] > 0.0  && i ∉ broken_constraints
            lims[i, :] = [0, Inf]
        elseif scaled_mean[i] < -0.0  && i ∉ broken_constraints
            lims[i, :] = [-Inf, 0]
        else
            lims[i, :] = [-Inf, Inf]
        end
    end

    #@info "Prior scaled mean: \n" * join(round.(scaled_mean; digits = 4), ", ")
    #@info "Prior scaled stds1: \n" * join(round.(scaled_std; digits = 4), ", ")

    #check that the std + mean are not too close to or above upper bound; if it is then reduce std (see EKP docs)
    for i in eachindex(scaled_mean)
        sum = scaled_mean[i] + 1.2*scaled_std[i]
        diff = scaled_mean[i] + 1.2*scaled_std[i]
        if sum > lims[i, 2] || diff < lims[i, 1]
            new_std = 0.9 * min(abs(lims[i, 2]-scaled_mean[i])/(1.2), abs(lims[i, 1]-scaled_mean[i])/(1.2))
            scaled_std[i] = new_std
            println("var $i rescaled due to μ + σ being too close to upper/lower bound")
        end
    end

    #@info "Prior scaled stds: \n" * join(round.(scaled_std; digits = 4), ", ")
    #generates scaled prior distribution and checks that constrained_gaussian scales correctly
    #=
    prior_dis_precheck = [constrained_gaussian("$i", 
                    scaled_mean[i], 
                    scaled_std[i], 
                    lims[i, 1], lims[i, 2] ) 
                    for i in 1:length(collected_vals)]
    prior_precheck = combine_distributions(prior_dis_precheck)
    broken = []
    println(mean(prior_precheck))
    
    diff = mean(prior_precheck) .- scaled_mean
    println(diff)
    for i in eachindex(diff)
        if abs(diff[i]) > 0.01
            lims[i, :] = [-Inf, Inf]
            push!(broken, i)
        end
    end
    println(broken) =#

    prior_dis = [constrained_gaussian("$i", 
                    scaled_mean[i], 
                    scaled_std[i], 
                    lims[i, 1], lims[i, 2] ) 
                    for i in 1:length(scaled_mean)]
    prior = combine_distributions(prior_dis)
                
    Gt(x, y) = G(x, y, pH_true, pCO₂_true)

    unscaled_output = calculate_observations_cc(Gt, scaled_mean, data;
                                                input_scaling = input_scaling,
                                                output_scaling = nothing, 
                                                excluded_vars = excluded_vars)
    output_scaling = scale_parameters(unscaled_output)[2]

    @info "unscaled output: \n"  * join(round.(unscaled_output; digits = 4), ", ")

    #defines functions so that they are only in terms of the model
    F(u)= calculate_observations_cc(Gt, u, data;
                                    input_scaling = input_scaling,
                                    output_scaling = output_scaling, 
                                    excluded_vars = excluded_vars)
    H(m) = 0 
    #@info "scaled output: \n" * join(round.(F(scaled_mean); digits = 4), ", ")
    #@info  "output scaling: \n" * join(output_scaling, ", ")
    #mutable_vars = NamedTuple{keys(defaults)}(Tuple([keys(x) for x in collect(defaults)]))

    et = now()
    @info "Initialisation time: " * string(et - st) * "\n"

    return EKPObject_cc(nothing, 
                    Inf, 
                    Inf, 
                    excluded_vars, 
                    iterations, 
                    nothing, 
                    prior, 
                    prior_mean, 
                    input_scaling, 
                    output_scaling,
                    data, 
                    u -> F(u), 
                    m -> H(m))
end

##########################################################################################

function optimise_parameters!(obj::EKPObject_cc, truth)
    unscale_in(list) = scale_list(list, -1 * obj.input_scaling)
    unscale_out(list) = scale_list(list, -1 * obj.output_scaling)

    G(u) = obj.observations(u)

    α_reg =  1.0
    update_freq = 1
    N_iter = obj.iterations
    prior = obj.prior
    
    dim_input = length(obj.prior_mean)
    dim_output = length(truth.samples[1])

    process = Unscented(mean(prior), cov(prior); α_reg = α_reg, update_freq = update_freq)
    uki_obj = EnsembleKalmanProcess(truth.mean, truth.obs_noise_cov, process, failure_handler_method = SampleSuccGauss())
    err = zeros(N_iter)

    diff = unscale_in(get_ϕ_mean_final(prior, uki_obj)) .- obj.prior_mean

    for i in eachindex(diff)
        if abs(diff[i]/obj.prior_mean[i]) > 0.01
            println(i)
            println("diff: ",round(100*diff[i]/obj.prior_mean[i]; digits = 4), "% | constrained mean: ", unscale_in(get_ϕ_mean_final(prior, uki_obj))[i], "| real mean: ",   obj.prior_mean[i])
            throw("prior mean and optimised constrained mean is not the same, see constrained_gaussian")
        end
    end

    @info "EKP starting...."

    current_best = 0
    min_err = 0 
        try 
            for i in 1:N_iter
                params_i = get_ϕ_final(prior, uki_obj)
                J =  size(params_i)[2]
                #display(params_i[38:39, :])

                G_ens = zeros(Float64, dim_output, J)
                Threads.@threads for j in 1:J
                    G_ens[:, j] .= G(params_i[:, j])
                end
                
                EKP.update_ensemble!(uki_obj, G_ens)

                err[i] = get_error(uki_obj)[end]
                @info ("\n" * 
                "Iteration: " * string(i) *
                ", Error: " * string(err[i]) *
                " norm(Cov):" * string(norm(uki_obj.process.uu_cov[i])) * "\n")
                #display(unscale_in(get_ϕ_mean_final(prior, uki_obj)))
                display(pairs(set_model_cc(unscale_in(get_ϕ_mean_final(prior, uki_obj)), obj.excluded_vars, false)))
                if err[i] == min(negcheck.(err[1:i])...)
                    current_best = unscale_in(get_ϕ_mean_final(prior, uki_obj))
                    min_err = err[i]
                    @info ("new min of " * string(min_err))
                end
            end
            cc = CarbonChemistry()
            final_ensemble = get_ϕ_final(prior, uki_obj)
            final_params = unscale_in(get_ϕ_mean_final(prior, uki_obj))
            final_cov = get_u_cov_final(uki_obj)

            final_eqc = set_model(cc; u = final_params, excluded_vars = obj.excluded_vars, return_model = true)
            final_params = set_model(cc; u = final_params, excluded_vars = obj.excluded_vars, return_model = false)
            error_std = unscale_in([sqrt(final_cov[i, i]) for i in 1:dim_input])

            if negcheck.(err[end]) > min_err
                @warn "returned params and eqc are not from the last iteration but instead the best (min err). " * 
                "\n best params had error " * string(min_err)
            end

            return (final_ensemble = final_ensemble, 
                    final_eqc = set_model(cc; u = current_best, excluded_vars = obj.excluded_vars, return_model = true),
                    final_params = current_best, 
                    error_std = error_std, 
                    final_error = err[end])
        catch e
            @error e
            @info "min error run had error " * string(min_err)
            return current_best
        end
end


function get_model_error(model, data)
    pH_err = []
    pCO₂_err = []
    for dp in data
        ph_conditions = filter_nt_fields((:depth,), merge(dp.loc, dp.values))
        pco2_conditions = filter_nt_fields((:depth, :T, :P), merge(dp.loc, dp.values))
        pH_glodap = dp.measurements.pH
        pCO₂_glodap = dp.measurements.pCO₂

        phe = model(; ph_conditions..., return_pH = true) - pH_glodap
        pce = model(; pco2_conditions..., T = 20, P = 0) - pCO₂_glodap
        push!(pH_err, phe)
        push!(pCO₂_err, pce)
        if abs(pce) > 50
            #println( model(; pco2_conditions..., T = 20, P = 0), " | " , pCO₂_glodap)
        end
    end

    return (pH = pH_err, pCO₂ = pCO₂_err)
end

function plot_errors(
    data;
    to_plot = :pCO₂,
    model = CarbonChemistry(),
    color_name = :S, 
    limit_y = false)

    color_marker = [dp.values[color_name] for dp in data]
    fig = CairoMakie.Figure(; size=(1600,900))

    ax = CairoMakie.Axis(fig[1, 1], title = "DIC")#, aspect = DataAspect(), title = "Local grid")
    ax2 = CairoMakie.Axis(fig[1, 2], title = "Alk")
    ax3 = CairoMakie.Axis(fig[1, 3], title = "pH")
    ax4 = CairoMakie.Axis(fig[2, 1], title = "T")
    ax5 = CairoMakie.Axis(fig[2, 2], title = "S")
    ax6 = CairoMakie.Axis(fig[2, 3], title = "Depth")

    error = get_model_error(model, data)[to_plot]

    sc=CairoMakie.scatter!(ax, [dp.values.DIC for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)
    CairoMakie.scatter!(ax2, [dp.values.Alk for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)
    CairoMakie.scatter!(ax3, [dp.measurements.pH for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)
    CairoMakie.scatter!(ax4, [dp.values.T for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)
    CairoMakie.scatter!(ax5, [dp.values.S for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)
    CairoMakie.scatter!(ax6, [dp.loc.depth for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)

    if limit_y && to_plot == :pH
        [CairoMakie.ylims!(a, -0.2, 0.2) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    elseif limit_y && to_plot == :pCO₂
        [CairoMakie.ylims!(a, -250, 250) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    end
    
    title = "Δ" * string(to_plot)

    Colorbar(fig[1:2, 4], sc)

    Label(fig[0, :], title)

    return fig
end

