using CairoMakie

function get_model_error(model, data)
    pH_err = []
    pCO₂_err = []
    for dp in data
        ph_conditions = filter_nt_fields((:depth,), dp.values)
        pco2_conditions = filter_nt_fields((:depth, :T, :P), dp.values)
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
    ylims = nothing)

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
    CairoMakie.scatter!(ax6, [dp.values.depth for dp in data], error, markersize = 3, alpha = 0.5, color = color_marker)

    if !isnothing(ylims) && to_plot == :pH
        [CairoMakie.ylims!(a, ylims[1], ylims[2]) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    elseif !isnothing(ylims) && to_plot == :pCO₂
        [CairoMakie.ylims!(a, ylims[1], ylims[2]) for a in (ax, ax2, ax3, ax4, ax5, ax6)]
    end
    
    title = "Δ" * string(to_plot)

    Colorbar(fig[1:2, 4], sc)

    Label(fig[0, :], title)

    return fig
end

