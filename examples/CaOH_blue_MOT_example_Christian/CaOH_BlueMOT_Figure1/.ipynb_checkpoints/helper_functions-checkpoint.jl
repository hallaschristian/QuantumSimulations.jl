x(u) = real(u[32+4+1]) * (1/k)
y(u) = real(u[32+4+2]) * (1/k)
z(u) = real(u[32+4+3]) * (1/k)
vx(u) = real(u[32+4+4]) * (Γ/k)
vy(u) = real(u[32+4+5]) * (Γ/k)
vz(u) = real(u[32+4+6]) * (Γ/k)

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
; 

function prob_func!(prob)
    prob.p.n_scatters = 0.
    prob.p.last_decay_time = 0.
    prob.p.time_to_decay = rand(Exponential(1))
    prob.p.params.photon_budget = rand(Geometric(1/13500))
    update_initial_position!(prob, sample_position(prob.p.params) ./ (1/k))
    update_initial_velocity!(prob, sample_velocity(prob.p.params) ./ (Γ/k))
    update_phases!(prob)
    return nothing
end

function prob_func_diffusion!(prob)
    prob.p.n_scatters = 0.
    prob.p.last_decay_time = 0.
    prob.p.time_to_decay = rand(Exponential(1))
    prob.p.params.photon_budget = rand(Geometric(1/13500))
    update_initial_position!(prob, sample_position(prob.p.params) ./ (1/k))
    update_initial_velocity!(prob, sample_velocity(prob.p.params) ./ (Γ/k))
    update_phases!(prob)
    
    # set magnetic field
    prob.p.params.Bx = +x(prob.u0) * prob.p.params.B_grad_end * 1e2 / 2
    prob.p.params.By = +y(prob.u0) * prob.p.params.B_grad_end * 1e2 / 2
    prob.p.params.Bz = -z(prob.u0) * prob.p.params.B_grad_end * 1e2
    return nothing
end

function terminate_condition(u,t,integrator)
    p = integrator.p
    r = sqrt(x(u)^2 + y(u)^2 + z(u)^2)
    if r >= 3e-3 && (t > 1e-3 / (1/p.Γ))
        return true
    elseif p.n_scatters > p.params.photon_budget
        return true
    end
    return false
end

@everywhere function sample_position(p)
    r = (rand(p.x_dist), rand(p.y_dist), rand(p.z_dist))
end
@everywhere function sample_velocity(p)
    v = (rand(p.vx_dist), rand(p.vy_dist), rand(p.vz_dist))
end

### FITTING FUNCTIONS ###
using LsqFit

function gaussian(x, p)
    σ, x0, A = p
   return A * exp.(-(x.-x0).^2/(2σ^2))
end

function maxwell_boltzmann(v, p)
    A, temp = p
    return A * ((m/(2π*kB*temp))^(3/2) * 4π) .* v .^2 .* exp.(v .^2 .* (-m / (2*kB*temp)))
end

function maxwell_boltzmann_1D(v, p)
    A, temp = p
   return A * (m/(2π*kB*temp))^(1/2) .* exp.(v .^2 .* (-m / (2*kB*temp)))
end

# POSITION FITTING FUNCTIONS #
function σ_fit(xs)
    
    hist_data = fit(Histogram, xs, -2e-3:1e-5:2e-3)
    hist_data.isdensity = true
    v = collect(hist_data.edges[1])
    dv = v[2]-v[1]
    v = v[1:end-1] .+ dv/2
    fv = hist_data.weights ./ (sum(hist_data.weights) * dv)
    
    v_fit = curve_fit(gaussian, v, fv, [60e-6, 0., 1000])
    σ, x0, A = v_fit.param
    
    return abs(σ)
end

function σ_coord_fit(sols, coord_func)
    coords_end = [coord_func(sol.u[end]) for sol ∈ sols if survived(sol)]
    return σ_fit(coords_end)
end
σx_fit(sols) = σ_coord_fit(sols, x)
σy_fit(sols) = σ_coord_fit(sols, y)
σz_fit(sols) = σ_coord_fit(sols, z)

# TEMPERATURE FITTING FUNCTIONS #
function T_fit(vs)
    
    hist_data = fit(Histogram, vs, 0:0.01:2.0)
    hist_data.isdensity = true
    v = collect(hist_data.edges[1])
    dv = v[2]-v[1]
    v = v[1:end-1] .+ dv/2
    fv = hist_data.weights ./ (sum(hist_data.weights) * dv)

    v_fit = curve_fit(maxwell_boltzmann, v, fv, [1, 150e-6])
    A, temp = v_fit.param
    
    return temp
end

function T_fit_1D(vs)
    
    hist_data = fit(Histogram, vs, -1.0:0.03:1.0)
    hist_data.isdensity = true
    v = collect(hist_data.edges[1])
    dv = v[2]-v[1]
    v = v[1:end-1] .+ dv/2
    fv = hist_data.weights ./ (sum(hist_data.weights) * dv)

    v_fit = curve_fit(maxwell_boltzmann_1D, v, fv, [1, 150e-6])
    A, temp = v_fit.param
    
    return temp
end

function T_coord_fit(sols, vcoord_func)
    vcoords_end = [vcoord_func(sol.u[end]) for sol ∈ sols if survived(sol)]
    return T_fit_1D(vcoords_end)
end
Tx_fit(sols) = T_coord_fit(sols, vx)
Ty_fit(sols) = T_coord_fit(sols, vy)
Tz_fit(sols) = T_coord_fit(sols, vz)

function survived(sol)
    r = x(sol.u[end]), y(sol.u[end]), z(sol.u[end])
    if Int(sol.retcode) == 1 && abs(r[1]) <= 2e-3 && abs(r[2]) <= 2e-3
        return true
    else
        return false
    end
end

function σ_geom_ensemble_sol(ensemble_sol)

    xs = [x(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]
    ys = [y(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]
    zs = [z(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]

    σ_x = σ_fit(xs)
    σ_y = σ_fit(ys)
    σ_z = σ_fit(zs)
    σ_geom = (σ_x * σ_y * σ_z)^(1/3)
    
    return σ_geom
end

function σ_geom_ensemble_sol(ensemble_sol, i)

    xs = [x(sol.u[i]) for sol ∈ ensemble_sol if survived(sol)]
    ys = [y(sol.u[i]) for sol ∈ ensemble_sol if survived(sol)]
    zs = [z(sol.u[i]) for sol ∈ ensemble_sol if survived(sol)]

    σ_x = σ_fit(xs)
    σ_y = σ_fit(ys)
    σ_z = σ_fit(zs)
    σ_geom = (σ_x * σ_y * σ_z)^(1/3)
    
    return σ_geom
end

function σ_vs_time(sols)

    _, max_time_idx = findmax(sol.t for sol ∈ sols)
    times = sols[max_time_idx].t
    
    σs = zeros(length(times))
    
    for i ∈ eachindex(times)
        xs = [x(sol.u[i]) for sol ∈ sols if length(sol.t) >= i]
        ys = [y(sol.u[i]) for sol ∈ sols if length(sol.t) >= i]
        zs = [z(sol.u[i]) for sol ∈ sols if length(sol.t) >= i]
    
        σ_x = σ_fit(xs)
        σ_y = σ_fit(ys)
        σ_z = σ_fit(zs)
        σ_geom = (σ_x * σ_y * σ_z)^(1/3)
        
        σs[i] = σ_geom
    end
    
    return σs
end

function density_vs_time(sols)

    N = 5000
    
    σs = σ_vs_time(sols)
    densities = zeros(length(σs))
    
    for i ∈ eachindex(densities)
        n_survived = sum(1 for sol ∈ sols if length(sol.t) >= i) / length(sols)
        densities[i] = N * n_survived / σs[i]^3 / (2π)^(3/2) * 1e-6
    end
    
    return densities
end

function T_ensemble_sol(ensemble_sol)

    vxs = [vx(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]
    vys = [vy(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]
    vzs = [vz(sol.u[end]) for sol ∈ ensemble_sol if survived(sol)]

    vs = sqrt.(vxs.^2 .+ vys.^2 .+ vzs.^2)
    
    T = T_fit(vs)
    
    return T
end

function distributed_solve(n_trajectories, prob, prob_func, scan_func, scan_values)
    
    n_steps = n_trajectories * length(workers()) * length(scan_values)
    p = Progress(n_steps)
    channel = RemoteChannel(() -> Channel{Bool}(), 1)

    all_sols = []

    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            for (i, scan_value) ∈ enumerate(scan_values)

                sols_futures = Vector{Future}()
                for pid ∈ workers()
                    sols_future = @spawnat pid begin
                        sols_workers = []
                        scan_func(prob, scan_value)
                        for j ∈ 1:n_trajectories
                            put!(channel, true)
                            prob_func(prob)
                            sol = DifferentialEquations.solve(prob)
                            push!(sols_workers, sol)
                        end
                        sols_workers
                    end
                    push!(sols_futures, sols_future)
                end
                sols = vcat(fetch.(sols_futures)...)
                push!(all_sols, sols)
            end
            put!(channel, false)
        end
    end
    return all_sols
end

using Bootstrap
function distributed_compute_diffusion(prob, prob_func, n_trajectories, t_end, τ_total, n_times, scan_func, scan_values)
    
    diffusions = zeros(length(scan_values))
    diffusion_errors = zeros(length(scan_values))
    diffusions_over_time = Vector{Float64}[]

    n_steps = n_trajectories * length(workers()) * length(scan_values)
    p = Progress(n_steps)
    channel = RemoteChannel(() -> Channel{Bool}(), 1)
    
    @sync begin
        @async while take!(channel)
            next!(p)
        end

        @async begin
            for (i, scan_value) ∈ enumerate(scan_values)
                
                futures = Vector{Future}()
                for pid ∈ workers()
                    future = @spawnat pid begin
                        scan_func(prob, scan_value)
                        compute_diffusion(prob, prob_func, n_trajectories, t_end, τ_total, n_times, channel)
                    end
                    push!(futures, future)
                end
                rets = fetch.(futures)
                
                Cs = real.(ret[1] for ret ∈ rets)
                fτ_fts = real.(ret[2] for ret ∈ rets)
                Cs_integrated = vcat(real.(ret[3] for ret ∈ rets)...)
                fτ_fts_integrated = vcat(real.(ret[4] for ret ∈ rets)...)
                
                diffusion_over_time = mean(Cs) .- mean(fτ_fts)
                push!(diffusions_over_time, diffusion_over_time)

                diffusion = Cs_integrated .- fτ_fts_integrated
                diffusions[i] = mean(diffusion)

                bs = bootstrap(mean, diffusion, BasicSampling(1000))
                diffusion_errors[i] = stderror(bs)[1]
                
            end
            put!(channel, false)
        end
    end
    return (diffusions, diffusion_errors, diffusions_over_time)
end