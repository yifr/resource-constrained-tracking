#=
Generate hierarchical-cluster MOT stimuli and save them as JSON.

Usage:
    julia --project=. scripts/generate_stimuli.jl [--key=value ...]

Options (all optional):
    --n_trials=1            number of trials (seeds seed, seed+1, ...)
    --seed=1
    --n_groups=4
    --group_sizes=3,1,5,2   fixed sizes (overrides random 1..5 sizes)
    --duration=10           seconds
    --fps=30
    --width=20 --height=20
    --speed=2.5             mean group speed
    --heading_diffusion=0.8
    --global_flow=0.0       std of shared flow velocity (0 = off)
    --rel_speed=0.35        std of objects' velocity relative to their group
    --noise=0.03            observation noise std
    --p_detect=1.0
    --out=output            output directory
=#

include(joinpath(@__DIR__, "..", "src", "stimuli.jl"))

function parse_args(args)
    opts = Dict{String,String}()
    for a in args
        m = match(r"^--([A-Za-z_]+)=(.*)$", a)
        m === nothing && error("Bad argument '$a' (expected --key=value)")
        opts[m[1]] = m[2]
    end
    return opts
end

function main(args = ARGS)
    o = parse_args(args)
    get_f(k, d) = parse(Float64, get(o, k, string(d)))
    get_i(k, d) = parse(Int, get(o, k, string(d)))

    sizes = haskey(o, "group_sizes") ? parse.(Int, split(o["group_sizes"], ",")) : nothing
    n_groups = sizes === nothing ? get_i("n_groups", 4) : length(sizes)

    cfg = StimulusConfig(
        width = get_f("width", 20.0),
        height = get_f("height", 20.0),
        n_groups = n_groups,
        group_sizes = sizes,
        duration = get_f("duration", 10.0),
        fps = get_i("fps", 30),
        dynamics = GlobalDynamics(
            speed_mean = get_f("speed", 2.5),
            heading_diffusion = get_f("heading_diffusion", 0.8),
            global_flow_std = get_f("global_flow", 0.0),
        ),
        object_dynamics = ObjectDynamics(rel_speed_std = get_f("rel_speed", 0.35)),
        observation = ObservationModel(
            noise_std = get_f("noise", 0.03),
            p_detect = get_f("p_detect", 1.0),
        ),
    )

    outdir = get(o, "out", "output")
    seed0 = get_i("seed", 1)
    for k in 0:get_i("n_trials", 1)-1
        seed = seed0 + k
        stim = generate_stimulus(cfg; seed = seed)
        path = save_stimulus(joinpath(outdir, "stimulus_seed$(seed).json"), stim)
        println("wrote $path  (groups: $(stim.group_sizes), objects: $(length(stim.group_of)), frames: $(length(stim.times)))")
    end
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
