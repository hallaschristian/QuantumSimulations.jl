### SIMULATION PROBLEMS DEFINITION: 1+2 BLUE MOT ###

# PROBLEM TO CALCULATE TRAJECTORIES #
t_start = 0.0
t_end   = 15e-3
t_span  = (t_start, t_end) ./ (1/Γ)

coupling_idxs = [[5:16,17:20],[1:4,5:16]]
p = initialize_prob_multiple(Float64, energies, freqs, sats, pols, ks, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p!, add_terms_dψ!, 16, 4, coupling_idxs)

function diffusion_kick(integrator)
    p = integrator.p
    n_states = p.n_states
    n_e = p.n_e
    @inbounds @fastmath for i ∈ 1:3
        kick = sqrt(2p.diffusion_constant[i] * p.sim_params.dt_diffusion) / p.m
        integrator.u[2n_states + n_e + 3 + i] += rand((-1,1)) * kick
    end
    return nothing
end

diffusion_times = (0:1e-7:t_end) ./ (1/Γ)
cb_diffusion = PresetTimeCallback(diffusion_times, diffusion_kick, save_positions=(false,false))
cb1 = DiscreteCallback(condition_discrete, stochastic_collapse_nokick!, save_positions=(false,false))
cb2 = DiscreteCallback(terminate_condition, terminate!)
cbs = CallbackSet(cb1, cb2, cb_diffusion)
kwargs = (alg=DP5(), reltol=1e-4, abstol=1e-5, saveat=1000, maxiters=1000000000, callback=cbs)

prob = ODEProblem(ψ_fast_multiple!, p.u0, t_span, p; kwargs...)

# PROBLEM TO COMPUTE DIFFUSION # # why inf here??? should be beam_radius
p_diffusion = initialize_prob_multiple(Float64, energies, freqs, sats, pols, ks, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_diffusion!, add_terms_dψ!, 16, 4, coupling_idxs)
cbs_diffusion = CallbackSet(cb1, cb_diffusion)
kwargs = (alg=DP5(), reltol=1e-4, abstol=1e-5, saveat=10000, maxiters=1000000000, callback=cbs_diffusion)
# prob_diffusion = ODEProblem(ψ_fast_multiple_ballistic!, p_diffusion.u0, (0, 10e-6 / (1/Γ)), p_diffusion; kwargs...) 
prob_diffusion = ODEProblem(ψ_fast_multiple_ballistic!, p_diffusion.u0, (0, 10e-6 / (1/Γ)), p_diffusion; kwargs...)

function prob_func_YO!(prob)
    prob.p.n_scatters = 0.
    prob.p.last_decay_time = 0.
    prob.p.time_to_decay = rand(Exponential(1))
    prob.p.sim_params.photon_budget = 10000000
    update_initial_position!(prob, sample_position(prob.p.sim_params) ./ (1/k))
    update_initial_velocity!(prob, sample_velocity(prob.p.sim_params) ./ (Γ/k))
    update_phases!(prob) # add back!
    return nothing
end

function prob_func_diffusion_YO!(prob)
    prob_func_YO!(prob)
    prob.p.sim_params.Bx = +x(prob.u0) * prob.p.sim_params.B_grad * 1e2 / 2
    prob.p.sim_params.By = +y(prob.u0) * prob.p.sim_params.B_grad * 1e2 / 2
    prob.p.sim_params.Bz = -z(prob.u0) * prob.p.sim_params.B_grad * 1e2
    return nothing
end

# function prob_func_diffusion_YO!(integrator)
#     prob_func_YO!(integrator)
#     integrator.p.sim_params.Bx = +x(integrator.u) * integrator.p.sim_params.B_grad * 1e2 / 2
#     integrator.p.sim_params.By = +y(integrator.u) * integrator.p.sim_params.B_grad * 1e2 / 2
#     integrator.p.sim_params.Bz = -z(integrator.u) * integrator.p.sim_params.B_grad * 1e2
#     return nothing
# end