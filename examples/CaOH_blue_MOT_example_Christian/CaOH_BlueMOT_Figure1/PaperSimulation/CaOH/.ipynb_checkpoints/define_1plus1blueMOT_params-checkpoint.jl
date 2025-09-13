### SIMULATION PARAMETERS: 3-FREQUENCY 1+1 BLUE MOT  ###

sim_type = Float64

# DEFINE STATES #
energies = energy.(states) .* (2π / Γ)

# DEFINE FREQUENCIES #
detuning = +7.6
δ1 = +0.00
δ2 = -0.75
# δ2 = +2.50

Δ1 = 1e6 * (detuning + δ1)
Δ2 = 1e6 * (detuning + δ2)

f1 = energy(states[end]) - energy(states[1]) + Δ1
f2 = energy(states[end]) - energy(states[10]) + Δ2

freqs = [f1, f2] .* (2π / Γ)

# DEFINE SATURATION INTENSITIES #
beam_radius = 5e-3
Isat = π*h*c*Γ/(3λ^3)
P = 0.60 * 13.1e-3 # 13.1 mW/1 V at 1.0 V, factor of 0.60 to match scattering rates
# P = 0.55 * 13.1e-3 # 13.1 mW/1 V at 1.0 V, factor of 0.60 to match scattering rates
I = 2P / (π * beam_radius^2)

total_sat = I / Isat
# s1 = 0.5625total_sat
# s2 = 0.4375total_sat
s1 = 0.5total_sat
s2 = 0.5total_sat

sats = [s1, s2]

# DEFINE POLARIZATIONS #
pols = [σ⁻, σ⁺]

k_relative = 1

# DEFINE FUNCTION TO UPDATE PARAMETERS DURING SIMULATION #
@everywhere function update_p_1plus1!(p, r, t)
    
    s = p.sim_params.total_sat
    s1_ratio = p.sim_params.s1_ratio
    s2_ratio = p.sim_params.s2_ratio
    s_scalar = min(t / p.sim_params.s_ramp_time, 1.0)
    s_factor = p.sim_params.s_factor_start + (p.sim_params.s_factor_end - p.sim_params.s_factor_start) * s_scalar
    p.sats[1] = s1_ratio * s * s_factor
    p.sats[2] = s1_ratio * s * s_factor
    
    # set ramped scale factor for B field
    B_scalar = min(t / p.sim_params.B_ramp_time, 1.0)
    B_grad = p.sim_params.B_grad_start + (p.sim_params.B_grad_end - p.sim_params.B_grad_start) * B_scalar
    p.sim_params.Bx = +r[1] * B_grad * 1e2 / k / 2
    p.sim_params.By = +r[2] * B_grad * 1e2 / k / 2
    p.sim_params.Bz = -r[3] * B_grad * 1e2 / k
    
    return nothing
end

σx_initial = 585e-6
σy_initial = 585e-6
σz_initial = 435e-6
Tx_initial = 35e-6
Ty_initial = 35e-6
Tz_initial = 35e-6

@everywhere function update_p_1plus1_diffusion!(p, r, t)
    s = p.sim_params.total_sat
    s_factor = p.sim_params.s_factor_end
    s1_ratio = p.sim_params.s1_ratio
    s2_ratio = p.sim_params.s2_ratio
    p.sats[1] = s1_ratio * s * s_factor
    p.sats[2] = s1_ratio * s * s_factor
    return nothing
end

import MutableNamedTuples: MutableNamedTuple
sim_params = MutableNamedTuple(
    Zeeman_Hx = MMatrix{size(Zeeman_x_mat)...}(sim_type.(Zeeman_x_mat)),
    Zeeman_Hy = MMatrix{size(Zeeman_y_mat)...}(sim_type.(Zeeman_y_mat)),
    Zeeman_Hz = MMatrix{size(Zeeman_z_mat)...}(sim_type.(Zeeman_z_mat)),
    
    B_ramp_time = 4e-3 / (1/Γ),
    B_grad_start = 0.,
    B_grad_end = 74., #74.,

    s_ramp_time = 4e-3 / (1/Γ),
    s_factor_start = 0.9,
    s_factor_end = 0.7,

    photon_budget = rand(Geometric(1/13500)),
    
    x_dist = Normal(0, σx_initial),
    y_dist = Normal(0, σy_initial),
    z_dist = Normal(0, σz_initial),
    
    vx_dist = Normal(0, sqrt(kB*Tx_initial/2m)),
    vy_dist = Normal(0, sqrt(kB*Ty_initial/2m)),
    vz_dist = Normal(0, sqrt(kB*Tz_initial/2m)),
    
    total_sat = total_sat,
    # s1_ratio = 0.5625,
    # s2_ratio = 0.4375,
    s1_ratio = 0.5,
    s2_ratio = 0.5,
    
    Bx = 0.,
    By = 0.,
    Bz = 0.,

    dt_diffusion = 1e-7 / (1/Γ)
)