import Oceananigans.TimeSteppers: time_step!, update_state!, next_time, store_tendencies!, compute_tendencies!
#using Oceananigans.TimeSteppers
using Oceananigans: AbstractModel, fields, prognostic_fields
import OceanBioME.BoxModels: RungeKutta3TimeStepper

export time_step!

@inline update_boxmodel_state!(model::BoxModel) = nothing

function rk3_substep!(model::BoxModel, Δt, γⁿ, ζⁿ)
    @inbounds for tracer in keys(model.fields)
        getproperty(model.fields, tracer) .+= Δt * (γⁿ * getproperty(model.timestepper.Gⁿ, tracer) + ζⁿ * getproperty(model.timestepper.G⁻, tracer))
    end
end

function rk3_substep!(model::BoxModel, Δt, γⁿ, ::Nothing)
    @inbounds for tracer in keys(model.fields)
        getproperty(model.fields, tracer) .+= Δt * γⁿ * getproperty(model.timestepper.Gⁿ, tracer)
    end
end

function store_tendencies!(model)
    @inbounds for tracer in keys(model.fields)
        getproperty(model.timestepper.G⁻, tracer) .= getproperty(model.timestepper.Gⁿ, tracer)
    end
end


function time_step!(model::BoxModel{<:Any, <:Any, <:Any, <:Any, <:RungeKutta3TimeStepper, <:Any}, Δt)
    Δt == 0 && error("Δt can not be zero")

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    # first stage
    compute_tendencies!(model, [])
    rk3_substep!(model, Δt, γ¹, nothing)

    tick!(model.clock, first_stage_Δt; stage=true)
    store_tendencies!(model)

    update_boxmodel_state!(model)

    # second stage
    compute_tendencies!(model, [])
    rk3_substep!(model, Δt, γ², ζ²)

    tick!(model.clock, second_stage_Δt; stage=true)
    store_tendencies!(model)

    update_boxmodel_state!(model)

    # third stage
    compute_tendencies!(model, [])
    rk3_substep!(model, Δt, γ³, ζ³)

    tick!(model.clock, third_stage_Δt)
    store_tendencies!(model)

    update_boxmodel_state!(model)

    return nothing
end

summary(::RungeKutta3TimeStepper{FT, TG}) where {FT, TG} = string("Runge-Kutta 3 Timetepper")
show(io::IO, model::RungeKutta3TimeStepper{FT, TG}) where {FT, TG} = print(io, summary(model))