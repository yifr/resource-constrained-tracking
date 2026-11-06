#=
Hierarchical multiple-object-tracking (MOT) stimulus generator.

Motion is a three-level hierarchy:

    global flow  ->  group (cluster) motion  ->  object motion

* Global flow:   one shared velocity `u_t` for the whole display (an OU process;
                 set `global_flow_std = 0` to turn it off).
* Group motion:  every group runs its *own instance* of the same global dynamics
                 model (`GlobalDynamics`): a heading random walk plus an OU process
                 on speed, bouncing off the arena walls. Group velocity is
                 `u_t + g_t`.
* Object motion: each object sits at `center + offset`. The offset has its own
                 small relative velocity (a damped spring toward the object's rest
                 offset plus noise), so objects jitter/drift within the cluster
                 while inheriting the group motion.

Observations are noisy, unlabeled, shuffled detections per frame (with optional
missed detections).
=#

using Random
using LinearAlgebra
using JSON3

using StaticArrays

const Vec2 = SVector{2,Float64}

v2(x, y) = Vec2(x, y)
vnorm(a::Vec2) = hypot(a[1], a[2])
randn2(rng) = Vec2(randn(rng), randn(rng))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

"""
Shared dynamics model that every group inherits (each group gets its own
independent sample path).
"""
Base.@kwdef struct GlobalDynamics
    speed_mean::Float64      = 2.5    # units / s, group speed the OU process reverts to
    speed_std::Float64       = 0.5    # stationary std of group speed
    speed_tau::Float64       = 2.0    # s, speed OU time constant
    heading_diffusion::Float64 = 0.8  # rad / sqrt(s), heading random-walk strength
    global_flow_std::Float64 = 0.0    # units / s, stationary std of shared flow (0 = off)
    global_flow_tau::Float64 = 3.0    # s, flow OU time constant
end

"""
Within-cluster dynamics: each object's small velocity relative to its group.
"""
Base.@kwdef struct ObjectDynamics
    rel_speed_std::Float64 = 0.35   # units / s, stationary std of relative velocity
    rel_tau::Float64       = 0.8    # s, relative-velocity OU time constant
    spring_k::Float64      = 1.5    # 1/s^2, pull back toward rest offset (keeps cluster cohesive)
end

Base.@kwdef struct ObservationModel
    noise_std::Float64 = 0.03  # Gaussian position noise
    p_detect::Float64  = 1.0   # per-object detection probability
end

Base.@kwdef struct StimulusConfig
    width::Float64         = 20.0
    height::Float64        = 20.0
    n_groups::Int          = 4
    min_group_size::Int    = 1
    max_group_size::Int    = 5
    group_sizes::Union{Nothing,Vector{Int}} = nothing  # override random sizes
    cluster_radius::Float64 = 1.2     # max rest-offset distance from group center
    min_object_sep::Float64 = 0.5     # min distance between rest offsets in a cluster
    min_group_sep::Float64  = 4.0     # min distance between initial group centers
    duration::Float64      = 10.0     # s
    fps::Int               = 30
    dynamics::GlobalDynamics = GlobalDynamics()
    object_dynamics::ObjectDynamics = ObjectDynamics()
    observation::ObservationModel = ObservationModel()
end

# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

struct Stimulus
    config::StimulusConfig
    seed::Int
    times::Vector{Float64}
    group_of::Vector{Int}                 # object index -> group index
    group_sizes::Vector{Int}
    positions::Array{Float64,3}           # (T, N, 2) object positions
    velocities::Array{Float64,3}          # (T, N, 2) object velocities
    group_centers::Array{Float64,3}       # (T, G, 2)
    global_flow::Matrix{Float64}          # (T, 2)
    observations::Vector{Matrix{Float64}} # per frame: (M_t, 2), shuffled, unlabeled
    obs_source::Vector{Vector{Int}}       # per frame: object index per detection; hidden from observer
end

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

function sample_group_sizes(rng, cfg::StimulusConfig)
    if cfg.group_sizes !== nothing
        @assert length(cfg.group_sizes) == cfg.n_groups "group_sizes length must equal n_groups"
        @assert all(cfg.min_group_size .<= cfg.group_sizes .<= cfg.max_group_size)
        return copy(cfg.group_sizes)
    end
    return rand(rng, cfg.min_group_size:cfg.max_group_size, cfg.n_groups)
end

"Rejection-sample `n` points in a box with pairwise separation >= `sep`."
function sample_separated(rng, n, sampler, sep; max_tries = 10_000)
    pts = Vec2[]
    tries = 0
    while length(pts) < n
        tries += 1
        tries > max_tries && error("Could not place $n points with separation $sep; " *
                                   "reduce n, separation, or enlarge the region.")
        p = sampler()
        all(vnorm(p - q) >= sep for q in pts) && push!(pts, p)
    end
    return pts
end

function sample_rest_offsets(rng, k, radius, sep)
    k == 1 && return [v2(0, 0)]
    disk() = begin
        r = radius * sqrt(rand(rng)); θ = 2π * rand(rng)
        v2(r * cos(θ), r * sin(θ))
    end
    offs = sample_separated(rng, k, disk, sep)
    # recentre so the group center is the cluster centroid
    c = (1 / k) * reduce(+, offs)
    return [o - c for o in offs]
end

# ---------------------------------------------------------------------------
# Dynamics helpers
# ---------------------------------------------------------------------------

# Exact discretisation of an OU process: x' = a x + sqrt(1-a^2) * σ * ξ
ou_coeffs(dt, tau) = (a = exp(-dt / tau); (a, sqrt(1 - a^2)))

"Reflect a velocity off the arena walls given position `p` and margin `m`."
function reflect(p::Vec2, v::Vec2, halfw, halfh, m)
    x, y = p
    vx, vy = v
    if x < -halfw + m && vx < 0; vx = -vx; x = 2(-halfw + m) - x; end
    if x >  halfw - m && vx > 0; vx = -vx; x = 2( halfw - m) - x; end
    if y < -halfh + m && vy < 0; vy = -vy; y = 2(-halfh + m) - y; end
    if y >  halfh - m && vy > 0; vy = -vy; y = 2( halfh - m) - y; end
    return v2(x, y), v2(vx, vy)
end

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

"""
    generate_stimulus(cfg = StimulusConfig(); seed = 1) -> Stimulus

Simulate one trial of hierarchical cluster motion and its observations.
"""
function generate_stimulus(cfg::StimulusConfig = StimulusConfig(); seed::Integer = 1)
    rng = MersenneTwister(seed)
    D, O, Obs = cfg.dynamics, cfg.object_dynamics, cfg.observation
    dt = 1 / cfg.fps
    T = round(Int, cfg.duration * cfg.fps) + 1
    halfw, halfh = cfg.width / 2, cfg.height / 2

    sizes = sample_group_sizes(rng, cfg)
    G = cfg.n_groups
    N = sum(sizes)
    group_of = reduce(vcat, [fill(g, sizes[g]) for g in 1:G])

    # Group margin: keep the whole cluster (plus some slack for jitter) inside the box.
    margin = cfg.cluster_radius + 0.5
    @assert 2margin < min(cfg.width, cfg.height) "Arena too small for cluster_radius"

    centers = sample_separated(rng, G,
        () -> v2((rand(rng) * 2 - 1) * (halfw - margin), (rand(rng) * 2 - 1) * (halfh - margin)),
        cfg.min_group_sep)

    rest = Vec2[]
    for g in 1:G
        append!(rest, sample_rest_offsets(rng, sizes[g], cfg.cluster_radius, cfg.min_object_sep))
    end

    # Group state: heading + speed (own motion, drawn from the shared model)
    heading = 2π .* rand(rng, G)
    speed = max.(0.0, D.speed_mean .+ D.speed_std .* randn(rng, G))
    flow = D.global_flow_std > 0 ? D.global_flow_std .* randn2(rng) : v2(0, 0)

    # Object state relative to group
    offset = copy(rest)
    relvel = [O.rel_speed_std .* randn2(rng) for _ in 1:N]

    a_s, b_s = ou_coeffs(dt, D.speed_tau)
    a_f, b_f = ou_coeffs(dt, D.global_flow_tau)
    a_r, b_r = ou_coeffs(dt, O.rel_tau)

    positions = zeros(T, N, 2)
    velocities = zeros(T, N, 2)
    group_centers = zeros(T, G, 2)
    global_flow = zeros(T, 2)

    gvel = [flow + speed[g] .* v2(cos(heading[g]), sin(heading[g])) for g in 1:G]

    function record!(t)
        global_flow[t, :] .= flow
        for g in 1:G
            group_centers[t, g, :] .= centers[g]
        end
        for i in 1:N
            g = group_of[i]
            p = centers[g] + offset[i]
            v = gvel[g] + relvel[i]
            positions[t, i, :] .= p
            velocities[t, i, :] .= v
        end
    end
    record!(1)

    for t in 2:T
        # --- global flow (shared by all groups)
        if D.global_flow_std > 0
            flow = a_f * flow + (b_f * D.global_flow_std) * randn2(rng)
        end

        # --- group motion: each group is an independent draw from GlobalDynamics
        for g in 1:G
            heading[g] += D.heading_diffusion * sqrt(dt) * randn(rng)
            speed[g] = D.speed_mean + a_s * (speed[g] - D.speed_mean) + b_s * D.speed_std * randn(rng)
            speed[g] = max(speed[g], 0.0)
            own = speed[g] .* v2(cos(heading[g]), sin(heading[g]))
            v = flow + own
            newc, newv = reflect(centers[g] + dt * v, v, halfw, halfh, margin)
            if newv != v
                # a bounce flips the *total* velocity; push the change into the
                # group's own heading so it persists (flow is shared and untouched)
                own = newv - flow
                speed[g] = vnorm(own)
                heading[g] = atan(own[2], own[1])
            end
            centers[g] = newc
            gvel[g] = newv
        end

        # --- object motion relative to its group (damped spring + OU noise)
        for i in 1:N
            pull = (-O.spring_k * dt) * (offset[i] - rest[i])
            relvel[i] = a_r * relvel[i] + pull + (b_r * O.rel_speed_std) * randn2(rng)
            offset[i] = offset[i] + dt * relvel[i]
        end

        record!(t)
    end

    # --- observations: noisy, possibly missed, then shuffled
    observations = Vector{Matrix{Float64}}(undef, T)
    obs_source = Vector{Vector{Int}}(undef, T)
    for t in 1:T
        pts = Vec2[]; src = Int[]
        for i in 1:N
            rand(rng) < Obs.p_detect || continue
            push!(pts, v2(positions[t, i, 1], positions[t, i, 2]) + Obs.noise_std .* randn2(rng))
            push!(src, i)
        end
        perm = randperm(rng, length(pts))
        M = zeros(length(pts), 2)
        for (j, k) in enumerate(perm)
            M[j, 1], M[j, 2] = pts[k]
        end
        observations[t] = M
        obs_source[t] = src[perm]
    end

    times = collect(0:T-1) .* dt
    return Stimulus(cfg, Int(seed), times, group_of, sizes, positions, velocities,
                    group_centers, global_flow, observations, obs_source)
end

# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

struct2dict(x) = Dict(string(f) => (v = getfield(x, f);
    v isa Union{GlobalDynamics,ObjectDynamics,ObservationModel} ? struct2dict(v) : v)
    for f in fieldnames(typeof(x)))

# (T, K, 2) array -> T × K × [x, y] nested vectors (row-major, JSON friendly)
nested(A::Array{Float64,3}) = [[[A[t, k, 1], A[t, k, 2]] for k in axes(A, 2)] for t in axes(A, 1)]
nested(M::Matrix{Float64}) = [[M[j, 1], M[j, 2]] for j in axes(M, 1)]

"""
    save_stimulus(path, stim)

Write a stimulus to JSON. Positions are indexed `[frame][object] = [x, y]`
(1-based in Julia, 0-based once loaded elsewhere). `observations[frame]` is an
unlabeled list of `[x, y]`; `obs_source` gives the hidden object id.
"""
function save_stimulus(path::AbstractString, s::Stimulus)
    mkpath(dirname(abspath(path)))
    d = Dict(
        "config" => struct2dict(s.config),
        "seed" => s.seed,
        "times" => s.times,
        "group_of" => s.group_of,
        "group_sizes" => s.group_sizes,
        "positions" => nested(s.positions),
        "velocities" => nested(s.velocities),
        "group_centers" => nested(s.group_centers),
        "global_flow" => [[s.global_flow[t, 1], s.global_flow[t, 2]] for t in axes(s.global_flow, 1)],
        "observations" => [nested(M) for M in s.observations],
        "obs_source" => s.obs_source,
    )
    open(path, "w") do io
        JSON3.write(io, d)
    end
    return path
end

"Load a stimulus JSON written by `save_stimulus` back into plain arrays."
function load_stimulus(path::AbstractString)
    d = JSON3.read(read(path, String))
    to3(x) = (T = length(x); K = length(x[1]);
              A = zeros(T, K, 2);
              for t in 1:T, k in 1:K; A[t, k, 1] = x[t][k][1]; A[t, k, 2] = x[t][k][2]; end; A)
    toM(x) = (M = zeros(length(x), 2); for j in eachindex(x); M[j, 1] = x[j][1]; M[j, 2] = x[j][2]; end; M)
    return (
        width = Float64(d.config.width),
        height = Float64(d.config.height),
        fps = Int(d.config.fps),
        times = Float64.(d.times),
        group_of = Int.(d.group_of),
        group_sizes = Int.(d.group_sizes),
        positions = to3(d.positions),
        group_centers = to3(d.group_centers),
        observations = [toM(o) for o in d.observations],
        obs_source = [Int.(s) for s in d.obs_source],
    )
end
