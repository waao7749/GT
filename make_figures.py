"""
one command produces every figure in the report, with fixed random seeds, written to figures/.
"""
import experiments
import sys
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")


# ------------------------------------------
# E1 fig
# ------------------------------------------
def fig_E1_matrix(data, ax, cfg):
    # score matrix A, rows and columns sorted by rank: who exploits whom
    order = data["order"]
    names = [data["keys"][i] for i in order]
    M = [[data["A"][i][j] for j in order] for i in order]
    payoffs = cfg["game"]["payoffs"]
    im = ax.imshow(M, cmap="viridis", vmin=payoffs["S"], vmax=payoffs["T"])
    for r in range(len(order)):
        for c in range(len(order)):
            ax.text(c, r, f"{M[r][c]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if M[r][c] < (payoffs["S"] + payoffs["T"]) / 2 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("opponent j")
    ax.set_ylabel("strategy i (best rank at top)")
    ax.set_title("E1: score per round A[i][j]")
    ax.figure.colorbar(im, ax=ax, label="score per round of i")


# ------------------------------------------
# E2 fig
# ------------------------------------------
def fig_E2(data, ax, cfg):
    # tournament score of every strategy as a function of N
    colors = experiments.strategy_colors(data["keys"], cfg)
    Ns = [row["N"] for row in data["per_n"]]
    for i, key in enumerate(data["keys"]):
        ax.plot(Ns, [row["scores"][i] for row in data["per_n"]], marker="o", markersize=3,
                label=key, color=colors[key], linewidth=2.4 if key == "TFT" else 1.2)
    ax.axvline(data["threshold"], color="grey", linestyle="--", linewidth=0.8)
    ax.text(data["threshold"], ax.get_ylim()[1], " (T-P)/(R-P)", fontsize=8,
            color="grey", va="top")
    ax.set_xscale("log")
    ax.set_xlabel("rounds per match N (log scale)")
    ax.set_ylabel("tournament score S_i (per round)")
    ax.set_title("E2: effect of the number of encounters")
    ax.legend(ncol=2, fontsize=7, frameon=False)

def fig_E2_population(data, ax, cfg):
    # what the population settles on, as a function of N
    Ns = [row["N"] for row in data["per_n"]]
    ax.plot(Ns, [row["population"]["cooperation"] for row in data["per_n"]],
            marker="o", color="tab:blue", label="mutual cooperation in stable population")
    ax.plot(Ns, [row["tau_vs_default"] for row in data["per_n"]],
            marker="s", color="tab:orange",
            label=f"Kendall tau of ranking vs N = {data['default_rounds']}")
    if data["crossover"] is not None:
        ax.axvline(data["crossover"], color="grey", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("rounds per match N (log scale)")
    ax.set_title("E2: stable population and ranking stability")
    ax.legend(fontsize=8, frameon=False)


# ------------------------------------------
# E3 fig
# ------------------------------------------
def fig_E3(data, ax, cfg):
    # stable mutual cooperation over the (T, S) plane
    Ts, Ss = data["Ts"], data["Ss"]
    dT, dS = (Ts[1] - Ts[0]) / 2, (Ss[1] - Ss[0]) / 2
    im = ax.imshow(data["cooperation"], origin="lower", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[Ts[0] - dT, Ts[-1] + dT, Ss[0] - dS, Ss[-1] + dS], aspect="auto")
    ax.axvline(1, color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    for x, y, label in [(0.5, 0.5, "harmony"), (1.5, 0.5, "snowdrift"),
                        (0.5, -0.5, "stag hunt"), (1.4, -0.75, "prisoner's\ndilemma")]:
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
    ax.plot([2.0], [-0.5], marker="*", color="black", markersize=10)
    ax.annotate("Axelrod (5,3,1,0)", (2.0, -0.5), textcoords="offset points",
                xytext=(-100, 10), fontsize=8)
    ax.set_xlabel("T (temptation), with R = 1, P = 0")
    ax.set_ylabel("S (sucker's payoff)")
    ax.set_title("E3: mutual cooperation in the stable population")
    ax.figure.colorbar(im, ax=ax, label="share of mutual cooperation")

def fig_E3_named(data, ax, cfg):
    # slope chart: rank of every strategy under the three concrete matrices
    colors = experiments.strategy_colors(data["keys"], cfg)
    names = list(data["named"])
    for key in data["keys"]:
        ranks = []
        for name in names:
            for rank, k, _ in data["named"][name]["ranking"]:
                if k == key:
                    ranks.append(rank)
        ax.plot(range(len(names)), ranks, marker="o", color=colors[key],
                linewidth=2.4 if key == "TFT" else 1.2)
        ax.text(len(names) - 1 + 0.08, ranks[-1], key, fontsize=8, va="center")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\ntau={data['named'][n]['tau_vs_reference']:.2f}" for n in names])
    ax.set_xlim(-0.3, len(names) - 0.4)
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("E3b: ranking under three payoff matrices")


# ------------------------------------------
# E5 fig
# ------------------------------------------
def fig_E5(data, ax, cfg):
    # shares over the generations from the uniform start
    colors = experiments.strategy_colors(data["keys"], cfg)
    history = data["uniform"]["history"]
    for i, key in enumerate(data["keys"]):
        ax.plot([x[i] for x in history], label=key, color=colors[key],
                linewidth=2.2 if key == "TFT" else 1.4)
    ax.set_xlabel("generation")
    ax.set_ylabel("population share")
    ax.set_title("E5: replicator dynamics from a uniform start")
    ax.legend(ncol=2, fontsize=8, frameon=False)


# ------------------------------------------
# E6 fig
# ------------------------------------------
def fig_E6(data, ax, cfg):
    # one panel: survival frequency (robustness) against growth when rare (viability)
    colors = experiments.strategy_colors(data["keys"], cfg)
    for key in data["keys"]:
        x = data["viability_field"][key]["growth"]
        y = data["robustness"][key]["survived"] / data["n_random_starts"]
        ax.scatter(x, y, color=colors[key], s=40)
        ax.annotate(key, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("growth factor when rare (initial viability)")
    ax.set_ylabel("share of random starts survived (robustness)")
    ax.set_title("E6: viability and robustness")


# ------------------------------------------
# E7 fig
# ------------------------------------------
def fig_E7(data, ax, cfg):
    # best and mean fitness per generation for every run
    for k, run in enumerate(data["runs"]):
        gens = [h["generation"] for h in run["history"]]
        ax.plot(gens, [h["best"] for h in run["history"]], color="tab:blue", alpha=0.6,
                linewidth=1, label="best" if k == 0 else None)
        ax.plot(gens, [h["mean"] for h in run["history"]], color="tab:orange", alpha=0.4,
                linewidth=1, label="mean" if k == 0 else None)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness (score per round vs fixed field)")
    ax.set_title(f"E7: genetic algorithm, {len(data['runs'])} runs")
    ax.legend(frameon=False)


# ------------------------------------------
# Command line
# ------------------------------------------
def save_figure(draws, data, cfg, name):
    # one figure per experiment: one panel per draw function, stacked top to bottom
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(__file__).with_name(cfg["figures"]["out_dir"])
    out_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(draws), 1, figsize=(7, 3.2 * len(draws)), squeeze=False)
    for k, (draw, ax) in enumerate(zip(draws, axes[:, 0])):
        draw(data, ax, cfg)
        if len(draws) > 1:
            # panel letter (a), (b), ... so the report can refer to Fig. 1a, 1b
            ax.text(-0.12, 1.06, f"({'abcdefgh'[k]})", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="bottom")
    fig.tight_layout()
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=cfg["figures"]["dpi"])
    plt.close(fig)
    return path

def report_E1(data):
    self_play = "with" if data["include_self"] else "no"
    print(f"E1 ranking (score per round, {self_play} self-play):")
    for rank, name, score in data["ranking"]:
        print(f"   {rank:2d}. {name:4s} {score:.3f}")

def report_E2(data):
    print(f"E2 theory threshold (T-P)/(R-P) = {data['threshold']:.2f}; "
          f"crossover N* (stable cooperation > 0.5) = {data['crossover']}")
    print("    N     best  TFT-rank  coop   survivors  tau")

    for row in data["per_n"]:
        best = row["ranking"][0][1]
        tft = [r for r, name, _ in row["ranking"] if name == "TFT"]
        pop = row["population"]
        print(f"   {row['N']:5d}  {best:4s}  {tft[0] if tft else '-':>4}      "
              f"{pop['cooperation']:.3f}  {pop['survivors']:2d}         {row['tau_vs_default']:.2f}")

def report_E3(data):
    print("E3a mean stable cooperation per game type:")
    for game, s in data["type_summary"].items():
        print(f"   {game:20s} {s['cells']:3d} cells  mean {s['mean_cooperation']:.3f}  "
              f"min {s['min_cooperation']:.3f}")
    print(f"E3b rankings (tau compared with {data['reference']}):")

    for name, d in data["named"].items():
        top = ", ".join(n for _, n, _ in d["ranking"][:3])
        print(f"   {name:13s} PD={d['is_pd']!s:5s} T+S>2R={d['alternating_beats_mutual_C']!s:5s} "
              f"top 3: {top:16s} tau={d['tau_vs_reference']:.2f}  "
              f"stable coop={d['population']['cooperation']:.3f}")

def report_E5(data):
    u = data["uniform"]
    print(f"E5 uniform start: converged = {u['converged']} after {u['generations']} generations")
    for key, share in zip(data["keys"], u["history"][-1]):
        print(f"   {key:4s} {share:.3f}")
    print(f"   population mutual cooperation: {u['cooperation'][-1]:.3f}")
    not_conv = sum(not r["converged"] for r in data["random_runs"])
    print(f"   random starts: {len(data['random_runs'])} runs, {not_conv} did not converge")

def report_E6(data):
    print("E6  key  growth(field) growth(ALLD) vsRandom survived rank   resisted/neutral/invaded")
    for key in data["keys"]:
        v, a = data["viability_field"][key], data["viability_alld"][key]
        r, s = data["robustness"][key], data["stability"][key]
        print(f"    {key:4s} {v['growth']:.3f}         {a['growth']:.3f}        "
              f"{r['vs_random_tables']:.3f}    {r['survived']:2d}/{data['n_random_starts']}  "
              f"{r['rank_best']}-{r['rank_worst']}  "
              f"{len(s['resisted'])}/{len(s['neutral'])}/{len(s['invaded_by'])}")

def report_E7(data):
    print(f"E7 best table {data['best_table']}  fitness {data['best_fitness']:.3f}  "
          f"(same as a field strategy: {data['equals_field_strategy']})")
    for rank, name, score in data["ranking"][:5]:
        print(f"   {rank:2d}. {name:4s} {score:.3f}")
    print(f"   final share of GA strategy: {data['final_shares']['GA']:.3f}")
    if "memory1_check" in data:
        m1 = data["memory1_check"]
        print(f"   memory-1 check passed = {m1['passed']}  exhaustive best {m1['exhaustive_best']}")


if __name__ == "__main__":
    cfg = experiments.load_config()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    # name: (run function, list of panel functions, report function); one figure file per experiment
    jobs = {"E1": (experiments.run_E1, [fig_E1_matrix], report_E1),
            "E2": (experiments.run_E2, [fig_E2, fig_E2_population], report_E2),
            "E3": (experiments.run_E3, [fig_E3, fig_E3_named], report_E3),
            "E5": (experiments.run_E5, [fig_E5], report_E5),
            "E6": (experiments.run_E6, [fig_E6], report_E6),
            "E7": (experiments.run_E7, [fig_E7], report_E7)}
    for name in (jobs if which == "all" else [which]):
        run, panels, report = jobs[name]
        data = run(cfg)
        report(data)
        print("   figure:", save_figure(panels, data, cfg, name))