"""
    Time step update function for stochastic Schrödinger simulation.
"""
function ψ_update!(du, u, p, t)
    
    zero_array!(du)

    normalize_u!(u, p.n_states)

    update_r!(u, p.r, p.r_idx)

    p.update_params(p, p.r, t)

    update_ψ!(p.ψ, u, p.n_states)

    update_fields!(p.denom, p.rs, p.ϕs, p.ωs, p.as, p.sats, p.kEs, p.ϵs, p.E, p.k_relative, p.r, t)
    
    update_eiωt_new!(p.eiω0ts, p.ω0s, t)

    Heisenberg_turbo_state!(p.ψ, p.eiω0ts, -1)

    update_ψq!(p.ψ_q, p.d_ge, p.d_eg, p.ψ)

    update_d_exp!(p.d_exp, p.ψ, p.ψ_q, p.n_g)
     
    update_force!(p.F, p.d_exp, p.kEs)

    update_velocity!(p.m, du, u, p.F, p.v_idx, p.F_idx)

    update_dψ!(p.dψ, p.ψ_q, p.E, p.n_g)

    p.add_terms_dψ(p.dψ, p.ψ, p, p.r, t) # custom terms to add to dψ

    Heisenberg_turbo_state!(p.dψ, p.eiω0ts, +1)

    update_du!(du, u, p.dψ, p.ψ, p.n_states, p.n_g, p.n_e, p.r_idx, p.v_idx, p.m)

    return nothing
end
export ψ_update!

function ψ_update_ballistic!(du, u, p, t)

    ψ_update!(du, u, p, t)
    
    # set force to zero
    @inbounds @fastmath for k ∈ 1:3
        du[p.v_idx+k] = zero(eltype(du))
    end

    return nothing
end
export ψ_update_ballistic!

@inline function normalize_u!(u, n_states)
    u_norm2 = zero(eltype(u))
    @turbo for i ∈ 1:2n_states
        u_norm2 += u[i]^2
    end
    u_norm = sqrt(u_norm2)
    @turbo for i ∈ 1:2n_states
        u[i] /= u_norm
    end
    return nothing
end

@inline function update_r!(u, r, r_idx)
    @inbounds @fastmath for k ∈ 1:3
        r[k] = u[r_idx+k]
    end
    return nothing
end

@inline function update_ψ!(ψ, u, n_states)
    @turbo for i ∈ eachindex(ψ)
        ψ.re[i] = u[i]
        ψ.im[i] = u[i+n_states]
    end
    return nothing
end

@inline function update_u!(u, ψ, n_states)
    @turbo for i ∈ 1:n_states
        u[i] = ψ.re[i]
        u[i+n_states] = ψ.im[i]
    end
    return nothing
end
export update_u!

@inline function update_du!(du, u, dψ, ψ, n_states, n_g, n_e, r_idx, v_idx, m)
    @turbo for i ∈ eachindex(dψ)
        du[i] = dψ.re[i]
        du[i+n_states] = dψ.im[i]
    end
    @inbounds @fastmath for k ∈ 1:3
        du[r_idx+k] = u[v_idx+k]
    end
    @turbo for i ∈ 1:n_e # integrated excited state population, combine this with loop below?
        ψ_i_pop = u[n_g+i]^2 + u[n_states+n_g+i]^2
        du[2n_states+i] = ψ_i_pop
    end
    @turbo for i ∈ 1:n_e
        # non-hermitian part of Hamiltonian, -im/2, but multiplied by -im also
        du[n_g+i] -= u[n_g+i]/2
        du[n_g+i+n_states] -= u[n_g+i+n_states]/2
    end
    return nothing
end

@inline function update_du_ballistic!(du, u, dψ, ψ, n_states, n_g, n_e, r_idx, F, v_idx, F_idx, m)
    update_du!(du, u, dψ, ψ, n_states, n_g, n_e, r_idx, F, v_idx, F_idx, m)
    @inbounds @fastmath for k ∈ 1:3
        du[v_idx+k] = 0.
    end
end

function update_velocity!(m, du, u, F, v_idx, F_idx)
    @inbounds @fastmath for k in 1:3
        du[v_idx+k] += F[k] / m
        du[F_idx+k+3] += F[k]
    end
    return nothing
end

"""
    Evalute ψ_q ≡ d_q ψ.

    Break up psi into ground and excited? Also do we really need the excited states, or can we do everything with ground states and taking conjugates?
"""
@inline function update_ψq!(ψ_q, d_ge, d_eg, ψ)
    n_g = size(d_ge,1)
    # ground states
    @turbo for q ∈ 1:3
        for i ∈ axes(d_ge,1)
            ψq_re_i = zero(eltype(ψ_q.re))
            ψq_im_i = zero(eltype(ψ_q.im))
            for j ∈ axes(d_ge,2)
                d_q_ij = d_ge[i,j,q]
                ψ_re_j = ψ.re[n_g+j]
                ψ_im_j = ψ.im[n_g+j]
                ψq_re_i += d_q_ij * ψ_re_j
                ψq_im_i += d_q_ij * ψ_im_j
            end
            ψ_q.re[i,q] = ψq_re_i
            ψ_q.im[i,q] = ψq_im_i
        end
    end
    # excited states
    @turbo for q ∈ 1:3
        for i ∈ axes(d_eg,1)
            ψq_re_i = zero(eltype(ψ_q.re))
            ψq_im_i = zero(eltype(ψ_q.im))
            for j ∈ axes(d_eg,2)
                d_q_ij = d_eg[i,j,q]
                ψ_re_j = ψ.re[j]
                ψ_im_j = ψ.im[j]
                ψq_re_i += d_q_ij * ψ_re_j
                ψq_im_i += d_q_ij * ψ_im_j
            end
            ψ_q.re[n_g+i,q] = ψq_re_i
            ψ_q.im[n_g+i,q] = ψq_im_i
        end
    end
    return nothing
end

# should this just be put directly into du? probably faster since we use one less function
@inline function update_dψ!(dψ, ψ_q, E, n_g)
    @turbo for i ∈ 1:n_g
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E.re[q]
            E_q_im = E.im[q]
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im # multiply by -im
        dψ.im[i] = -dψ_i_re
    end
    @turbo for i ∈ (n_g+1):length(dψ)
        dψ_i_re = zero(eltype(dψ.re))
        dψ_i_im = zero(eltype(dψ.im))
        for q ∈ 1:3
            E_q_re = E.re[q]
            E_q_im = -E.im[q] # conjugate for the excited states
            ψ_q_re = ψ_q.re[i,q]
            ψ_q_im = ψ_q.im[i,q]
            
            dψ_i_re += ψ_q_re * E_q_re - ψ_q_im * E_q_im
            dψ_i_im += ψ_q_re * E_q_im + ψ_q_im * E_q_re
        end
        dψ.re[i] = dψ_i_im # multiply by -im
        dψ.im[i] = -dψ_i_re
    end
    return nothing
end
export update_dψ!

# can maybe make it so that we don't have to take the sincos for all states if there are degeneracies
@inline function update_eiωt_new!(eiω0ts, ω0s, t)
    @turbo for i ∈ eachindex(eiω0ts)
        eiω0ts.im[i], eiω0ts.re[i] = sincos(ω0s[i] * t)
    end
    return nothing
end
export update_eiωt_new!

"""
    Note that this isn't actually d_exp, since we only go over the ground states.
    So this is the d^+ part of d = d^+ + d^-
"""
@inline function update_d_exp!(d_exp, ψ, ψ_q, n_g)
    @turbo for q ∈ 1:3
        re = zero(eltype(ψ.re))
        im = zero(eltype(ψ.im))
        for i ∈ 1:n_g
            ψ_re = ψ.re[i] # take conjugate
            ψ_im = -ψ.im[i]
            ψq_re = ψ_q.re[i,q]
            ψq_im = ψ_q.im[i,q]
            re += ψ_re * ψq_re - ψ_im * ψq_im
            im += ψ_re * ψq_im + ψq_re * ψ_im
        end
        d_exp.re[q] = re
        d_exp.im[q] = im
    end
    return nothing
end

@inline function update_force!(F, d_exp, kEs)
    @turbo for k ∈ 1:3
        F_k = zero(eltype(F))
        for q ∈ 1:3
            d_q_re = d_exp.re[q]
            d_q_im = d_exp.im[q]
            
            # multiply by -im
            E_kq_im = -(kEs.re[k,q] - kEs.re[k+3,q])
            E_kq_re = (kEs.im[k,q] - kEs.im[k+3,q])
            
            F_k_a_re = d_q_re * E_kq_re - d_q_im * E_kq_im

            F_k -= 2F_k_a_re

        end
        F[k] = F_k
    end
    return nothing
end
export update_force!

function stochastic_collapse_new!(integrator)

    u = integrator.u
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_e
    n_ground = p.n_g
    d_ge = p.d_ge
    ψ = p.ψ
    
    p⁺ = zero(eltype(ψ.re))
    p⁰ = zero(eltype(ψ.re))
    p⁻ = zero(eltype(ψ.re))

    @turbo for i ∈ 1:n_excited
        c_i_re = ψ.re[n_ground + i] 
        c_i_im = -ψ.im[n_ground + i] # take conjugate
        for j ∈ 1:n_excited
            c_j_re = ψ.re[n_ground + j]
            c_j_im = ψ.im[n_ground + j]
            re = c_i_re * c_j_re - c_i_im * c_j_im
            for k ∈ 1:n_ground
                p⁺ += re * d_ge[k,i,1] * d_ge[k,j,1] # assume that d is real
                p⁰ += re * d_ge[k,i,2] * d_ge[k,j,2]
                p⁻ += re * d_ge[k,i,3] * d_ge[k,j,3]
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end

    # zero ground state amplitudes
    @turbo for i ∈ 1:n_ground
        # ψ.re[i] = zero(eltype(ψ.re))
        # ψ.im[i] = zero(eltype(ψ.im))
        u[i] = zero(eltype(u))
        u[i+n_states] = zero(eltype(u))
    end

    # decay from excited to ground state
    @turbo for i ∈ 1:n_ground
        for j ∈ 1:n_excited
            d = d_ge[i,j,pol]
            # ψ.re[i] += d * ψ.re[n_ground+j]
            # ψ.im[i] += d * ψ.im[n_ground+j]
            u[i] += d * ψ.re[n_ground+j]
            u[i+n_states] += d * ψ.im[n_ground+j]
        end
    end
    
    # zero excited state amplitudes
    @turbo for i ∈ 1:n_excited
        # ψ.re[n_ground+i] = zero(eltype(ψ.re))
        # ψ.im[n_ground+i] = zero(eltype(ψ.im))
        u[n_ground+i] = zero(eltype(u))
        u[i+n_states+n_ground] = zero(eltype(u))
    end

    # zero integrated excited state populations - # add this with loop above???
    @turbo for i ∈ 1:n_excited
        u[2n_states+i] = zero(eltype(u))
    end

    # add diffusion
    time_before_decay = integrator.t - p.last_decay_time
    @inbounds @fastmath for i ∈ 1:3
        kick = sqrt( 2p.diffusion_constant[i] * time_before_decay ) / p.m
        u[2n_states + n_excited + 3 + i] += rand((-1,1)) * kick
        # u[2n_states + n_excited + 3 + i] += rand(Normal(0,kick))
    end
    p.last_decay_time = integrator.t

    # add spontaneous decay
    if p.add_spontaneous_decay_kick
        @inbounds @fastmath for i ∈ 1:3
            u[2n_states + n_excited + 3 + i] += rand((-1,1)) / p.m
        end
    end

    p.time_to_decay = rand(p.decay_dist)
    p.n_scatters += 1

    return nothing
end
export stochastic_collapse_new!

function stochastic_collapse_nokick!(integrator)

    u = integrator.u
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_e
    n_ground = p.n_g
    d_ge = p.d
    ψ = p.ψ
    
    p⁺ = zero(eltype(ψ.re))
    p⁰ = zero(eltype(ψ.re))
    p⁻ = zero(eltype(ψ.re))

    @turbo for i ∈ 1:n_excited
        c_i_re = ψ.re[n_ground + i]
        c_i_im = -ψ.im[n_ground + i] # take conjugate
        for j ∈ 1:n_excited
            c_j_re = ψ.re[n_ground + j]
            c_j_im = ψ.im[n_ground + j]
            re = c_i_re * c_j_re - c_i_im * c_j_im
            for k ∈ 1:n_ground
                p⁺ += re * d_ge[k,i+n_ground,1] * d_ge[k,j+n_ground,1] # assume that d is real
                p⁰ += re * d_ge[k,i+n_ground,2] * d_ge[k,j+n_ground,2]
                p⁻ += re * d_ge[k,i+n_ground,3] * d_ge[k,j+n_ground,3]
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end

    # zero ground state amplitudes
    @turbo for i ∈ 1:n_ground
        # ψ.re[i] = zero(eltype(ψ.re))
        # ψ.im[i] = zero(eltype(ψ.im))
        u[i] = zero(eltype(u))
        u[i+n_states] = zero(eltype(u))
    end

    # decay from excited to ground state
    @turbo for i ∈ 1:n_ground
        for j ∈ 1:n_excited
            d = d_ge[i,j+n_ground,pol]
            u[i] += d * ψ.re[n_ground+j]
            u[i+n_states] += d * ψ.im[n_ground+j]
        end
    end
    
    # zero excited state amplitudes
    @turbo for i ∈ 1:n_excited
        u[n_ground+i] = zero(eltype(u))
        u[i+n_states+n_ground] = zero(eltype(u))
    end

    # zero integrated excited state populations - # add this with loop above???
    @turbo for i ∈ 1:n_excited
        u[2n_states+i] = zero(eltype(u))
    end

    # add spontaneous decay
    if p.add_spontaneous_decay_kick
        @inbounds @fastmath for i ∈ 1:3
            u[2n_states + n_excited + 3 + i] += rand((-1,1)) / p.m
        end
    end

    p.time_to_decay = rand(p.decay_dist)
    p.n_scatters += 1

    return nothing
end
export stochastic_collapse_nokick!


@inline function condition_new(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = zero(eltype(u))
    @inbounds @fastmath for i ∈ 1:4
        p_i = u[2p.n_states+i]
        integrated_excited_pop += p_i
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition
end
export condition_new

@inline function condition_discrete(u,t,integrator)
    p = integrator.p
    integrated_excited_pop = zero(eltype(u))
    @inbounds @fastmath for i ∈ 1:p.n_e
        p_i = u[2p.n_states+i]
        integrated_excited_pop += p_i
    end
    _condition = integrated_excited_pop - p.time_to_decay
    return _condition > 0
end
export condition_discrete

function stochastic_collapse_no_diffusion!(integrator)

    u = integrator.u
    p = integrator.p
    n_states = p.n_states
    n_excited = p.n_e
    n_ground = p.n_g
    d_ge = p.d_ge
    ψ = p.ψ
    
    p⁺ = zero(eltype(ψ.re))
    p⁰ = zero(eltype(ψ.re))
    p⁻ = zero(eltype(ψ.re))

    @turbo for i ∈ 1:n_excited
        c_i_re = ψ.re[n_ground + i] 
        c_i_im = -ψ.im[n_ground + i] # take conjugate
        for j ∈ 1:n_excited
            c_j_re = ψ.re[n_ground + j]
            c_j_im = ψ.im[n_ground + j]
            re = c_i_re * c_j_re - c_i_im * c_j_im
            for k ∈ 1:n_ground
                p⁺ += re * d_ge[k,i,1] * d_ge[k,j,1] # assume that d is real
                p⁰ += re * d_ge[k,i,2] * d_ge[k,j,2]
                p⁻ += re * d_ge[k,i,3] * d_ge[k,j,3]
            end
            # note the polarization p in d[:,:,p] is defined to be m_e - m_g, 
            # whereas the polarization of the emitted photon is m_g - m_e
        end
    end

    p_norm = p⁺ + p⁰ + p⁻
    rn = rand() * p_norm
    
    pol = 0
    if 0 < rn <= p⁺ # photon is measured to have polarization σ⁺
        pol = 1
    elseif p⁺ < rn <= p⁺ + p⁰ # photon is measured to have polarization σ⁰
        pol = 2
    else # photon is measured to have polarization σ⁻
        pol = 3
    end

    # zero ground state amplitudes
    @turbo for i ∈ 1:n_ground
        # ψ.re[i] = zero(eltype(ψ.re))
        # ψ.im[i] = zero(eltype(ψ.im))
        u[i] = zero(eltype(u))
        u[i+n_states] = zero(eltype(u))
    end

    # decay from excited to ground state
    @turbo for i ∈ 1:n_ground
        for j ∈ 1:n_excited
            d = d_ge[i,j,pol]
            # ψ.re[i] += d * ψ.re[n_ground+j]
            # ψ.im[i] += d * ψ.im[n_ground+j]
            u[i] += d * ψ.re[n_ground+j]
            u[i+n_states] += d * ψ.im[n_ground+j]
        end
    end
    
    # zero excited state amplitudes
    @turbo for i ∈ 1:n_excited
        # ψ.re[n_ground+i] = zero(eltype(ψ.re))
        # ψ.im[n_ground+i] = zero(eltype(ψ.im))
        u[n_ground+i] = zero(eltype(u))
        u[i+n_states+n_ground] = zero(eltype(u))
    end

    # zero integrated excited state populations - # add this with loop above???
    @turbo for i ∈ 1:n_excited
        u[2n_states+i] = zero(eltype(u))
    end

    p.time_to_decay = rand(p.decay_dist)
    p.n_scatters += 1

    return nothing
end
export stochastic_collapse_no_diffusion!