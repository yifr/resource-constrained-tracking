#=
Proof of concept: grouping as a single-frame facility-location MILP.

Each frame is solved independently (no temporal linking yet):

    minimize    sum_ij c_ij x_ij  +  lambda * sum_j y_j
    subject to  sum_j x_ij = 1        every object tracked through one representative
                x_ij <= y_j           only through a representative that is switched on
                x, y binary

    c_ij = |p_i - p_j|^2 / sigma_p^2  +  |v_i - v_j|^2 / sigma_v^2

Candidate representatives are the objects themselves. Positions and velocities
are the TRUE ones from the generator (velocities are not observable from
unlabeled detections without a matching step; that is future work).

Usage:
    julia --project=. scripts/poc_grouping.jl output/stimulus_seed1.json [--lambda=6] [--sigp=1.2] [--sigv=0.5] [--speed=2]

Writes <basename>_poc_grouping.mp4 and <basename>_poc_frontier.png next to the JSON.
=#

include(joinpath(@__DIR__, "..", "src", "stimuli.jl"))
using JuMP, Gurobi, CairoMakie

const GRB_ENV = Ref{Gurobi.Env}()

function load_velocities(json_path)
    d = JSON3.read(read(json_path, String))
    x = d.velocities
    T, N = length(x), length(x[1])
    V = zeros(T, N, 2)
    for t in 1:T, i in 1:N
        V[t, i, 1] = x[t][i][1]; V[t, i, 2] = x[t][i][2]
    end
    return V
end

"Cost matrix: c[i, j] = cost of tracking object i through representative j."
function cost_matrix(P, V; sigp, sigv)
    N = size(P, 1)
    return [sum(abs2, P[i, :] .- P[j, :]) / sigp^2 + sum(abs2, V[i, :] .- V[j, :]) / sigv^2
            for i in 1:N, j in 1:N]
end

"""
Solve the grouping MILP for one frame. Give `lambda` (price per group) or
`K` (hard limit on number of groups), or both.
Returns (rep, mismatch): rep[i] = representative of object i, mismatch = sum c_ij x_ij.
"""
function solve_grouping(c; lambda = 0.0, K = nothing)
    N = size(c, 1)
    m = direct_model(Gurobi.Optimizer(GRB_ENV[]))
    set_silent(m)
    @variable(m, x[1:N, 1:N], Bin)
    @variable(m, y[1:N], Bin)
    @constraint(m, [i in 1:N], sum(x[i, :]) == 1)
    @constraint(m, [i in 1:N, j in 1:N], x[i, j] <= y[j])
    K === nothing || @constraint(m, sum(y) <= K)
    @objective(m, Min, sum(c .* x) + lambda * sum(y))
    optimize!(m)
    @assert termination_status(m) == MOI.OPTIMAL
    xv = value.(x)
    rep = [argmax(xv[i, :]) for i in 1:N]
    return rep, sum(c[i, rep[i]] for i in 1:N)
end

"Relabel groups by their smallest member so colors stay stable across frames."
function canonical(rep)
    lab = Dict{Int,Int}()
    for i in eachindex(rep)
        lab[rep[i]] = min(get(lab, rep[i], i), i)
    end
    return [lab[r] for r in rep]
end

same_partition(a, b) = all((a[i] == a[j]) == (b[i] == b[j]) for i in eachindex(a), j in eachindex(a))

function main(args = ARGS)
    pos = filter(a -> !startswith(a, "--"), args)
    json_path = isempty(pos) ? "output/stimulus_seed1.json" : pos[1]
    opt = Dict("lambda" => 6.0, "sigp" => 1.2, "sigv" => 0.5, "speed" => 2.0)
    for a in args
        m = match(r"^--(\w+)=([\d.]+)$", a)
        m !== nothing && (opt[m[1]] = parse(Float64, m[2]))
    end
    lambda, sigp, sigv = opt["lambda"], opt["sigp"], opt["sigv"]

    GRB_ENV[] = Gurobi.Env()
    s = load_stimulus(json_path)
    V = load_velocities(json_path)
    P = s.positions
    T, N = size(P, 1), size(P, 2)
    G = length(s.group_sizes)
    frames = unique(round.(Int, range(1, T; step = opt["speed"])))
    costs = Dict(t => cost_matrix(P[t, :, :], V[t, :, :]; sigp, sigv) for t in frames)

    # ---- 1. sweep the price lambda: how often is the true grouping recovered?
    println("\nlambda sweep over $(length(frames)) frames (sigp=$sigp, sigv=$sigv), truth has $G groups")
    println("  lambda   mean #groups   frames matching truth")
    for lam in (0.5, 1, 2, 4, 6, 8, 12, 20, 40, 80)
        ng, hit = 0, 0
        for t in frames
            rep, _ = solve_grouping(costs[t]; lambda = lam)
            ng += length(unique(rep)); hit += same_partition(rep, s.group_of)
        end
        println("  ", rpad(lam, 8), rpad(round(ng / length(frames), digits = 2), 15),
                round(100 * hit / length(frames), digits = 1), "%")
    end

    # ---- 2. hard-K frontier: mismatch vs number of slots, averaged over frames
    Ks = 1:N
    frontier = [sum(solve_grouping(costs[t]; K = K)[2] for t in frames) / length(frames) for K in Ks]
    println("\nK frontier (mean mismatch cost per frame):")
    for K in Ks
        println("  K=", rpad(K, 4), round(frontier[K], digits = 2))
    end
    base = replace(json_path, r"\.json$" => "")
    fig = Figure(size = (640, 420))
    ax = Axis(fig[1, 1]; xlabel = "number of slots K", ylabel = "mean mismatch cost per frame",
              title = "Memory vs. tracking error ($N objects, $G true groups)", xticks = 1:N)
    scatterlines!(ax, collect(Ks), frontier, color = :black)
    vlines!(ax, [G], color = :red, linestyle = :dash, label = "true number of groups")
    axislegend(ax)
    save(base * "_poc_frontier.png", fig)
    println("wrote $(base)_poc_frontier.png")

    # ---- 3. animation at the chosen lambda: ground truth vs MILP grouping
    sol = Dict(t => solve_grouping(costs[t]; lambda = lambda)[1] for t in frames)
    palette = Makie.wong_colors()
    extra = cgrad(:tab20, 20, categorical = true)
    colr(k) = k <= length(palette) ? palette[k] : extra[mod1(k, 20)]
    hw, hh = s.width / 2, s.height / 2
    frame = Observable(frames[1])
    fig = Figure(size = (1200, 660))
    Label(fig[0, 1:2], @lift("t = $(round(s.times[$frame], digits = 2)) s    lambda = $lambda    " *
        "MILP groups: $(length(unique(sol[$frame])))    matches truth: $(same_partition(sol[$frame], s.group_of))"),
        fontsize = 18)
    ax_kw = (aspect = DataAspect(), limits = (-hw, hw, -hh, hh),
             xticklabelsvisible = false, yticklabelsvisible = false,
             xticksvisible = false, yticksvisible = false, xgridvisible = false, ygridvisible = false)
    ax1 = Axis(fig[1, 1]; title = "Ground truth ($G groups)", ax_kw...)
    ax2 = Axis(fig[1, 2]; title = "MILP grouping (ring = representative, lines = assignment)", ax_kw...)
    pts = @lift Point2f.(P[$frame, :, 1], P[$frame, :, 2])
    scatter!(ax1, pts, color = [colr(g) for g in s.group_of], markersize = 16,
             strokecolor = :black, strokewidth = 0.8)
    segs = @lift begin
        r = sol[$frame]
        out = Point2f[]
        for i in 1:N
            push!(out, Point2f(P[$frame, i, 1], P[$frame, i, 2]))
            push!(out, Point2f(P[$frame, r[i], 1], P[$frame, r[i], 2]))
        end
        out
    end
    linesegments!(ax2, segs, color = (:black, 0.4), linewidth = 1.5)
    scatter!(ax2, pts, color = @lift([colr(k) for k in canonical(sol[$frame])]), markersize = 16,
             strokecolor = :black, strokewidth = 0.8)
    reps = @lift begin
        r = unique(sol[$frame])
        Point2f.(P[$frame, r, 1], P[$frame, r, 2])
    end
    scatter!(ax2, reps, color = :transparent, markersize = 28, strokecolor = :black, strokewidth = 2)
    mp4 = base * "_poc_grouping.mp4"
    record(fig, mp4, frames; framerate = s.fps) do t
        frame[] = t
    end
    println("wrote $mp4")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
