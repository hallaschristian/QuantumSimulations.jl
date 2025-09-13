## SIMULATION PROBLEMS DEFINITION: 1+1 BLUE MOT ###

# PROBLEM TO CALCULATE TRAJECTORIES #
t_start = 0.0
t_end   = 16e-3
t_span  = (t_start, t_end) ./ (1/Γ)

p_1plus1 = initialize_prob(sim_type, energies, freqs, sats, pols, k_relative, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_1plus1!, add_terms_dψ!, 1:12, 13:16)

@everywhere function diffusion_kick(integrator)
    p = integrator.p
    @inbounds @fastmath for i ∈ 1:3
        kick = sqrt(2p.diffusion_constant[i] * p.sim_params.dt_diffusion) / p.m
        integrator.u[2p.n_states + p.n_e + 3 + i] += rand((-1,1)) * kick
    end   
    return nothing
end

diffusion_times = (0:1e-7:t_end) ./ (1/Γ)
cb_diffusion = PresetTimeCallback(diffusion_times, diffusion_kick, save_positions=(false,false))
cb1 = DiscreteCallback(condition_discrete, stochastic_collapse_nokick!, save_positions=(false,false))
cb2 = DiscreteCallback(terminate_condition, terminate!)
cbs = CallbackSet(cb1, cb2, cb_diffusion)
kwargs = (alg=DP5(), reltol=1e-4, abstol=1e-5, saveat=1000, maxiters=100000000, callback=cbs)
prob_1plus1 = ODEProblem(ψ_update!, p_1plus1.u0, sim_type.(t_span), p_1plus1; kwargs...)

# PROBLEM TO COMPUTE DIFFUSION #
p_1plus1_diffusion = initialize_prob(sim_type, energies, freqs, sats, pols, k_relative, beam_radius, d, m/(ħ*k^2/Γ), Γ, k, sim_params, update_p_1plus1_diffusion!, add_terms_dψ!, 1:12, 13:16)
cbs_diffusion = CallbackSet(cb1, cb_diffusion)
kwargs = (alg=DP5(), reltol=1e-4, abstol=1e-5, saveat=1000, maxiters=100000000, callback=cbs_diffusion)
prob_1plus1_diffusion = ODEProblem(ψ_update_ballistic!, p_1plus1_diffusion.u0, sim_type.(t_span), p_1plus1_diffusion; kwargs...)

# #
function update_phases!(prob)
    for k ∈ 1:3
        ϕ1 = exp(im * 2π * rand())
        ϕ2 = exp(im * 2π * rand())
        for q ∈ axes(prob.p.ϵs,3)
            for f ∈ axes(prob.p.ϵs,2)
                prob.p.ϵs[k,f,q] *= ϕ1
                prob.p.ϵs[k+3,f,q] *= ϕ2
            end
        end
    end
    return nothing
end

function prob_func!(prob)
    reset_prob!(prob)
    prob.p.sim_params.photon_budget = rand(Geometric(1/13500))
    update_initial_position!(prob, sample_position(prob.p.sim_params) ./ (1/k))
    update_initial_velocity!(prob, sample_velocity(prob.p.sim_params) ./ (Γ/k))
    update_phases!(prob)
    return nothing
end

function prob_func_diffusion!(prob)
    prob_func!(prob)
    prob.p.sim_params.Bx = +x(prob.u0) * prob.p.sim_params.B_grad_end * 1e2 / 2
    prob.p.sim_params.By = +y(prob.u0) * prob.p.sim_params.B_grad_end * 1e2 / 2
    prob.p.sim_params.Bz = -z(prob.u0) * prob.p.sim_params.B_grad_end * 1e2
    return nothing
end