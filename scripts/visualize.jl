#=
Render a stimulus JSON as a side-by-side mp4:
  left  = ground truth, colored by group (with short trails and centroid marker)
  right = uncolored, unlabeled observations (what the tracker actually sees)

Usage:
    julia --project=. scripts/visualize.jl output/stimulus_seed1.json [out.mp4] [--trail=15] [--speed=2] [--res=1]

  --speed  playback speed multiplier (2 = twice real time). Implemented by
           skipping simulation frames, so the video stays at the stimulus fps.
  --res    pixel density (1 = 1200x640, 2 = 2400x1280). Higher is slower to render.
=#

include(joinpath(@__DIR__, "..", "src", "stimuli.jl"))
using CairoMakie

# Okabe–Ito-style palette, distinguishable for color-blind viewers; cycles if >8 groups
const GROUP_COLORS = [
    colorant"#E69F00", colorant"#56B4E9", colorant"#009E73", colorant"#F0E442",
    colorant"#0072B2", colorant"#D55E00", colorant"#CC79A7", colorant"#999999",
]
group_color(g) = GROUP_COLORS[mod1(g, length(GROUP_COLORS))]

function animate_stimulus(json_path, mp4_path; trail = 15, marker_size = 16, speed = 2.0, res = 1)
    s = load_stimulus(json_path)
    T, N = size(s.positions, 1), size(s.positions, 2)
    G = length(s.group_sizes)
    hw, hh = s.width / 2, s.height / 2

    frame = Observable(1)

    fig = Figure(size = (1200, 640), backgroundcolor = :white)
    title_str = @lift "t = $(round(s.times[$frame], digits = 2)) s   (frame $($frame)/$T)"
    Label(fig[0, 1:2], title_str, fontsize = 18)

    ax_kw = (aspect = DataAspect(), limits = (-hw, hw, -hh, hh),
             xticksvisible = false, yticksvisible = false,
             xticklabelsvisible = false, yticklabelsvisible = false,
             xgridvisible = false, ygridvisible = false)
    ax_gt = Axis(fig[1, 1]; title = "Ground truth ($G groups, $N objects)", ax_kw...)
    ax_obs = Axis(fig[1, 2]; title = "Observations", ax_kw...)

    # --- ground truth: trails, objects, group centroids
    obj_colors = [group_color(g) for g in s.group_of]
    for i in 1:N
        pts = @lift begin
            t0 = max(1, $frame - trail)
            Point2f.(s.positions[t0:$frame, i, 1], s.positions[t0:$frame, i, 2])
        end
        lines!(ax_gt, pts, color = (obj_colors[i], 0.35), linewidth = 2)
    end
    gt_pts = @lift Point2f.(s.positions[$frame, :, 1], s.positions[$frame, :, 2])
    scatter!(ax_gt, gt_pts, color = obj_colors, markersize = marker_size,
             strokecolor = :black, strokewidth = 0.8)
    ctr_pts = @lift Point2f.(s.group_centers[$frame, :, 1], s.group_centers[$frame, :, 2])
    scatter!(ax_gt, ctr_pts, color = [group_color(g) for g in 1:G],
             marker = :xcross, markersize = 10)

    # --- observations: all detections identical, no identity, no color
    obs_pts = @lift begin
        M = s.observations[$frame]
        isempty(M) ? Point2f[] : Point2f.(M[:, 1], M[:, 2])
    end
    scatter!(ax_obs, obs_pts, color = :gray25, markersize = marker_size)

    @assert speed > 0 "speed must be positive"
    frames = unique(round.(Int, range(1, T; step = speed)))
    record(fig, mp4_path, frames; framerate = s.fps, px_per_unit = res) do t
        frame[] = t
    end
    return mp4_path
end

function main(args = ARGS)
    pos = filter(a -> !startswith(a, "--"), args)
    isempty(pos) && error("usage: visualize.jl stimulus.json [out.mp4] [--trail=15] [--speed=2] [--res=1]")
    json_path = pos[1]
    mp4_path = length(pos) >= 2 ? pos[2] : replace(json_path, r"\.json$" => ".mp4")
    trail, speed, res = 15, 2.0, 1
    for a in args
        (m = match(r"^--trail=(\d+)$", a)) !== nothing && (trail = parse(Int, m[1]))
        (m = match(r"^--speed=([\d.]+)$", a)) !== nothing && (speed = parse(Float64, m[1]))
        (m = match(r"^--res=([\d.]+)$", a)) !== nothing && (res = parse(Float64, m[1]))
    end
    animate_stimulus(json_path, mp4_path; trail = trail, speed = speed, res = res)
    println("wrote $mp4_path")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
