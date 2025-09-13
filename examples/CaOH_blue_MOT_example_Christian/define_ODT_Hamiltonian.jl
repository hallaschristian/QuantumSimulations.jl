"""
    Compute the Hamiltonian for the molecule-ODT interaction. Assumes that H_ODT is real.
"""
function get_H_ODT(states, X_states, A_states, peak_intensity, pol, wavelength=1064e-9)
    Isat = π*h*c*Γ/(3λ^3) # saturation intensity
    s = peak_intensity / Isat
    f_ODT = c/wavelength
    
    n_states = length(states)
    G = sqrt(s)/(2 * √2) # factor?
    H_ODT = zeros(Float64, n_states, n_states)
    
    all_states = [X_states; A_states]
    d = tdms_between_states(all_states, all_states)
    fs = energy.(all_states)

    for q ∈ 1:3
        for p ∈ 1:3
            for i ∈ 1:n_states
                for i′ ∈ 1:n_states
                    for j ∈ 1:length(all_states)
                        
                        # println((1/((fs[j] - fs[i]) - f_ODT) + 1/((fs[j] - fs[i]) + f_ODT)) / h)
                        
                        H_ODT[i,i′] += (2π * Γ) * (G^2/4) * d[i,j,q] * pol[q] * d[j,i′,p] * pol[p] * (1/((fs[j] - fs[i]) - f_ODT) + 1/((fs[j] - fs[i]) + f_ODT))
                       
                    end
                end
            end
        end
    end
    
    return H_ODT
end
