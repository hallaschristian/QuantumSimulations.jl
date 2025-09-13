function flip(ϵ)
    return SVector{3, ComplexF64}(ϵ[3],ϵ[2],ϵ[1])
end

function gaussian_intensity_along_axes(r, axes)
    """1/e^2 width = 5 mm Gaussian beam """
    d2 = r[axes[1]]^2 + r[axes[2]]^2   
    return exp(-2d2/((5e-3*k)^2))
end

function define_lasers(
        states,
        s1,
        s2,
        s3,
        s4,
        Δ1,
        Δ2,
        Δ3,
        Δ4,
        pol1_x,
        pol2_x,
        pol3_x,
        pol4_x,
        s_ramp_time,
        s_ramp_factor
    )
    
    ω1 = 2π * (energy(states[end]) - energy(states[1]) + Δ1)
    ω2 = 2π * (energy(states[end]) - energy(states[1]) + Δ2)
    ω3 = 2π * (energy(states[end]) - energy(states[10]) + Δ3)
    ω4 = 2π * (energy(states[end]) - energy(states[10]) + Δ4)

    ϵ_func(ϵ) = t -> ϵ
    s_gaussian_ramp(factor, ramp_time, axes) = (r,t) -> (1 + (factor-1)/ramp_time * min(t, ramp_time)) * gaussian_intensity_along_axes(r, axes)
    
    ϕs = [2π*rand() for i ∈ 1:6]
    
    k̂ = +x̂; ϵ1 = rotate_pol(pol1_x, k̂); ϵ_func1 = ϵ_func(ϵ1); laser1 = Field(k̂, ϵ_func1, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[1])
    k̂ = -x̂; ϵ2 = rotate_pol(pol1_x, k̂); ϵ_func2 = ϵ_func(ϵ2); laser2 = Field(k̂, ϵ_func2, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[2])
    k̂ = +ŷ; ϵ3 = rotate_pol(pol1_x, k̂); ϵ_func3 = ϵ_func(ϵ3); laser3 = Field(k̂, ϵ_func3, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[3])
    k̂ = -ŷ; ϵ4 = rotate_pol(pol1_x, k̂); ϵ_func4 = ϵ_func(ϵ4); laser4 = Field(k̂, ϵ_func4, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[4])
    k̂ = +ẑ; ϵ5 = rotate_pol(flip(pol1_x), k̂); ϵ_func5 = ϵ_func(ϵ5); laser5 = Field(k̂, ϵ_func5, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[5])
    k̂ = -ẑ; ϵ6 = rotate_pol(flip(pol1_x), k̂); ϵ_func6 = ϵ_func(ϵ6); laser6 = Field(k̂, ϵ_func6, ω1, s1, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[6])

    lasers_1 = [laser1, laser2, laser3, laser4, laser5, laser6]

    k̂ = +x̂; ϵ7 = rotate_pol(pol2_x, k̂); ϵ_func7 = ϵ_func(ϵ7); laser7 = Field(k̂, ϵ_func7, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[1])
    k̂ = -x̂; ϵ8 = rotate_pol(pol2_x, k̂); ϵ_func8 = ϵ_func(ϵ8); laser8 = Field(k̂, ϵ_func8, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[2])
    k̂ = +ŷ; ϵ9 = rotate_pol(pol2_x, k̂); ϵ_func9 = ϵ_func(ϵ9); laser9 = Field(k̂, ϵ_func9, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[3])
    k̂ = -ŷ; ϵ10 = rotate_pol(pol2_x, k̂); ϵ_func10 = ϵ_func(ϵ10); laser10 = Field(k̂, ϵ_func10, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[4])
    k̂ = +ẑ; ϵ11 = rotate_pol(flip(pol2_x), k̂); ϵ_func11 = ϵ_func(ϵ11); laser11 = Field(k̂, ϵ_func11, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[5])
    k̂ = -ẑ; ϵ12 = rotate_pol(flip(pol2_x), k̂); ϵ_func12 = ϵ_func(ϵ12); laser12 = Field(k̂, ϵ_func12, ω2, s2, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[6])
    
    lasers_2 = [laser7, laser8, laser9, laser10, laser11, laser12]
    
    k̂ = +x̂; ϵ13 = rotate_pol(pol3_x, k̂); ϵ_func13 = ϵ_func(ϵ13); laser13 = Field(k̂, ϵ_func13, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[1])
    k̂ = -x̂; ϵ14 = rotate_pol(pol3_x, k̂); ϵ_func14 = ϵ_func(ϵ14); laser14 = Field(k̂, ϵ_func14, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[2])
    k̂ = +ŷ; ϵ15 = rotate_pol(pol3_x, k̂); ϵ_func15 = ϵ_func(ϵ15); laser15 = Field(k̂, ϵ_func15, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[3])
    k̂ = -ŷ; ϵ16 = rotate_pol(pol3_x, k̂); ϵ_func16 = ϵ_func(ϵ16); laser16 = Field(k̂, ϵ_func16, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[4])
    k̂ = +ẑ; ϵ17 = rotate_pol(flip(pol3_x), k̂); ϵ_func17 = ϵ_func(ϵ17); laser17 = Field(k̂, ϵ_func17, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[5])
    k̂ = -ẑ; ϵ18 = rotate_pol(flip(pol3_x), k̂); ϵ_func18 = ϵ_func(ϵ18); laser18 = Field(k̂, ϵ_func18, ω3, s3, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[6])

    lasers_3 = [laser13, laser14, laser15, laser16, laser17, laser18]

    k̂ = +x̂; ϵ19 = rotate_pol(pol4_x, k̂); ϵ_func19 = ϵ_func(ϵ19); laser19 = Field(k̂, ϵ_func19, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[1])
    k̂ = -x̂; ϵ20 = rotate_pol(pol4_x, k̂); ϵ_func20 = ϵ_func(ϵ20); laser20 = Field(k̂, ϵ_func20, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (2,3)), ϕs[2])
    k̂ = +ŷ; ϵ21 = rotate_pol(pol4_x, k̂); ϵ_func21 = ϵ_func(ϵ21); laser21 = Field(k̂, ϵ_func21, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[3])
    k̂ = -ŷ; ϵ22 = rotate_pol(pol4_x, k̂); ϵ_func22 = ϵ_func(ϵ22); laser22 = Field(k̂, ϵ_func22, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,3)), ϕs[4])
    k̂ = +ẑ; ϵ23 = rotate_pol(flip(pol4_x), k̂); ϵ_func23 = ϵ_func(ϵ23); laser23 = Field(k̂, ϵ_func23, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[5])
    k̂ = -ẑ; ϵ24 = rotate_pol(flip(pol4_x), k̂); ϵ_func24 = ϵ_func(ϵ24); laser24 = Field(k̂, ϵ_func24, ω4, s4, s_gaussian_ramp(s_ramp_factor, s_ramp_time, (1,2)), ϕs[6])

    lasers_4 = [laser19, laser20, laser21, laser22, laser23, laser24]

    lasers = [lasers_1; lasers_2; lasers_3; lasers_4]

end