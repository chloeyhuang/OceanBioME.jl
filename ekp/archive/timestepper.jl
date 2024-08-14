import Oceananigans.TimeSteppers: time_step!, update_state!, next_time, rk3_substep!, store_tendencies!
using Oceananigans.TimeSteppers
using Oceananigans: AbstractModel, fields, prognostic_fields

export time_step_c! 

function time_step_c!(model::AbstractModel{<:RungeKutta3TimeStepper}, Δt; callbacks=[], compute_tendencies = true)
    Δt == 0 && @warn "Δt == 0 may cause model blowup!"
    # Be paranoid and update state at iteration 0, in case run! is not used:
    model.clock.iteration == 0 && update_state!(model, callbacks)

    γ¹ = model.timestepper.γ¹
    γ² = model.timestepper.γ²
    γ³ = model.timestepper.γ³

    ζ² = model.timestepper.ζ²
    ζ³ = model.timestepper.ζ³

    first_stage_Δt  = γ¹ * Δt
    second_stage_Δt = (γ² + ζ²) * Δt
    third_stage_Δt  = (γ³ + ζ³) * Δt

    # Compute the next time step a priori to reduce floating point error accumulation
    tⁿ⁺¹ = next_time(model.clock, Δt)

    #
    # First stage
    #

    rk3_substep!(model, Δt, γ¹, nothing)

    #calculate_pressure_correction!(model, first_stage_Δt)
    #pressure_correct_velocities!(model, first_stage_Δt)

    tick!(model.clock, first_stage_Δt; stage=true)
    model.clock.last_stage_Δt = first_stage_Δt
    store_tendencies!(model)
    update_state!(model, callbacks)
    #step_lagrangian_particles!(model, first_stage_Δt)

    #
    # Second stage
    #

    rk3_substep!(model, Δt, γ², ζ²)

    #calculate_pressure_correction!(model, second_stage_Δt)
    #pressure_correct_velocities!(model, second_stage_Δt)

    tick!(model.clock, second_stage_Δt; stage=true)
    model.clock.last_stage_Δt = second_stage_Δt
    store_tendencies!(model)
    update_state!(model, callbacks)
    #step_lagrangian_particles!(model, second_stage_Δt)

    #
    # Third stage
    #
    
    rk3_substep!(model, Δt, γ³, ζ³)

    #calculate_pressure_correction!(model, third_stage_Δt)
    #pressure_correct_velocities!(model, third_stage_Δt)

    # This adjustment of the final time-step reduces the accumulation of
    # round-off error when Δt is added to model.clock.time. Note that we still use 
    # third_stage_Δt for the substep, pressure correction, and Lagrangian particles step.
    corrected_third_stage_Δt = tⁿ⁺¹ - model.clock.time
  
    tick!(model.clock, corrected_third_stage_Δt)
    model.clock.last_stage_Δt = corrected_third_stage_Δt
    model.clock.last_Δt = Δt

    update_state!(model, callbacks; compute_tendencies)
    #step_lagrangian_particles!(model, third_stage_Δt)

    return nothing
end
