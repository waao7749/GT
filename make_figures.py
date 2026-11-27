"""
one command produces every figure in the report, with fixed random seeds, written to figures/.
"""
import experiments
import sys
import Strategies
import model
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
# E4 fig
# ------------------------------------------
# Shared colours
NICE_COLOR, OTHER_COLOR = "#257de0", "#f47230"
TEXT, MUTED = "#191919", "#666666"


def is_nice(key):
    return "nice" in Strategies.get(key).traits


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)


def spread_labels(values, min_gap):
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)

    for a, b in zip(order, order[1:]):
        placed[b] = max(placed[b], placed[a] + min_gap)

    return placed


def fig_E4(data, ax, cfg):
    history = data["uniform"]["history"]
    keys = data["keys"]
    gens = range(len(history))
    final = history[-1]

    # Plot strategy populations
    for i, key in enumerate(keys):
        nice = is_nice(key)
        color = "black" if key == "TFT" else NICE_COLOR if nice else OTHER_COLOR

        ax.plot(gens, [h[i] for h in history], color=color,
                linewidth=2.0 if key == "TFT" else 1.4,
                linestyle="-" if nice else "--")

    # Label surviving and extinct strategies
    alive = [i for i, v in enumerate(final) if v > 0.01]

    for i, y in zip(alive, spread_labels([final[i] for i in alive], 0.012)):
        ax.text(len(history) + 1, y, f"{keys[i]} {final[i]:.3f}",
                fontsize=7, color=TEXT, va="center")

    gone = ", ".join(k for k, v in zip(keys, final) if v <= 0.01)
    ax.text(len(history) + 1, 0, f"extinct: {gone}",
            fontsize=7, color=MUTED, va="bottom")

    # Mark convergence
    g = data["uniform"]["generations"]
    ax.axvline(g, color=MUTED, linestyle=":", linewidth=0.8)
    ax.text(g, 0.03, f"converged\n(gen {g})", fontsize=7,
            color=MUTED, ha="right", va="bottom")

    # Legend
    ax.plot([], [], color=NICE_COLOR, label="nice strategies")
    ax.plot([], [], color="black", linewidth=2, label="TFT (benchmark)")
    ax.plot([], [], color=OTHER_COLOR, linestyle="--", label="non-nice strategies")
    ax.legend(fontsize=7, frameon=False, loc="center", bbox_to_anchor=(0.45, 0.25))

    ax.set(xlim=(0, len(history) * 1.25),
           xlabel="generation", ylabel="population share",
           title="E4: replicator dynamics from a uniform start")

    clean_axes(ax)


def fig_E4_steady_states(data, ax, cfg):
    from matplotlib.patches import Patch

    keys = data["keys"]
    runs = [("U", data["uniform"]["history"][-1])]
    runs += [(str(n), r["final"]) for n, r in enumerate(data["random_runs"], 1)]

    # Group nice-only states first
    runs.sort(key=lambda r: any(v > 0.01 for k, v in zip(keys, r[1]) if not is_nice(k)))
    # Plot steady states
    for x, (_, final) in enumerate(runs):
        bottom = 0

        for k, v in sorted(zip(keys, final), key=lambda r: -r[1]):
            if v < 0.005:
                continue

            ax.bar(x, v, bottom=bottom, width=0.8,
                   color=NICE_COLOR if is_nice(k) else OTHER_COLOR,
                   edgecolor="white", linewidth=1)

            if v >= 0.12:
                ax.text(x, bottom + v / 2, k, fontsize=5.5,
                        color="white", ha="center", va="center", rotation=90)

            bottom += v

    ax.set(xticks=range(len(runs)),
           xticklabels=[name for name, _ in runs],
           ylim=(0, 1))
    ax.set_xlabel("start (U = uniform, 1-20 = random)", fontsize=8, color=MUTED)
    ax.set_ylabel("share in steady state", fontsize=8, color=MUTED)
    ax.set_title("E4: steady state reached from each start",
                 fontsize=9, color=TEXT, loc="left")

    ax.legend(handles=[
        Patch(color=NICE_COLOR, label="nice strategy"),
        Patch(color=OTHER_COLOR, label="non-nice strategy")
    ], fontsize=7, frameon=False, loc="lower right",
       bbox_to_anchor=(1, 1.0), ncol=2)

    clean_axes(ax)


# ------------------------------------------
# E5 fig
# ------------------------------------------
def fig_E5(data, ax, cfg):
    # one panel: survival frequency (robustness) against growth when rare (viability)
    colors = experiments.strategy_colors(data["keys"], cfg)
    for key in data["keys"]:
        x = data["viability_field"][key]["growth"]
        y = data["robustness"][key]["survived"] / data["n_random_starts"]
        ax.scatter(x, y, color=colors[key], s=40)
        ax.annotate(key, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xlabel("growth factor when rare")
    ax.set_ylabel("share of random starts survived")
    ax.set_title("E5: viability and robustness")


# ------------------------------------------
# E6 fig
# ------------------------------------------

def fig_E6(data, ax, cfg):
    import GA

    histories = [r["history"] for r in data["runs"]]
    length = min(max(len(h) for h in histories), 101)

    best = []
    mean = []

    for i in range(length):
        best.append(sorted(h[min(i, len(h)-1)]["best"] for h in histories))
        mean.append(sorted(h[min(i, len(h)-1)]["mean"] for h in histories))

    gens = range(length)
    mid = len(histories) // 2

    ax.fill_between(gens, [x[0] for x in best], [x[-1] for x in best], color=NICE_COLOR, alpha=0.15)
    ax.plot(gens, [x[mid] for x in best], color=NICE_COLOR,
            label=f"best in population (median, {len(histories)} runs)")
    ax.plot(gens, [x[mid] for x in mean], color=OTHER_COLOR,
            label="population mean (median)")

    field = [s.table for s in Strategies.fixed_field]

    for key in ("PAV", "TFT"):
        fitness = GA.candidate_fitness(
            Strategies.get(key).table, field,
            cfg["game"], cfg["ranking"]["include_self"]
        )
        ax.hlines(fitness, 0, length-1, color=MUTED, linestyle=":")
        ax.text(length-1, fitness, f" {key} ({fitness:.3f})",
                fontsize=7, color=MUTED, va="center")

    ax.text(length-1, data["best_fitness"],
            f" evolved strategy ({data['best_fitness']:.3f})",
            fontsize=7, color=TEXT, va="center")

    ax.set(
        xlim=(0, length * 1.5),
        xlabel="generation",
        ylabel="fitness (score per round)",
        title="E6: genetic algorithm"
    )

    ax.legend(fontsize=7, frameon=False, loc="lower right")
    clean_axes(ax)


def fig_E6_advantage(data, ax, cfg):
    game = cfg["game"]
    rows = []

    for opp in Strategies.fixed_field + [Strategies.ALLD]:
        ga = model.play_match(data["best_table"], opp.table,game["rounds"], game["payoffs"])["score_a"]
        pav = model.play_match(
            Strategies.PAVLOV2.table, opp.table, game["rounds"], game["payoffs"])["score_a"]
        rows.append((opp.key, ga - pav))

    alld = next(r for r in rows if r[0] == "ALLD")
    rows = [alld] + sorted([r for r in rows if r[0] != "ALLD"],key=lambda r: r[1])

    for y, (key, diff) in enumerate(rows):
        color = OTHER_COLOR if key == "ALLD" else NICE_COLOR
        ax.barh(y, diff, height=0.6, color=color)

        if abs(diff) < 1e-9:
            ax.text(0, y, " same score", fontsize=7, color=TEXT, va="center")
        elif diff > 0:
            ax.text(diff, y, f" {diff:+.3f}", fontsize=7, color=TEXT, va="center")
        else:
            # negative bar: label on the left end of the bar
            ax.text(diff, y, f"{diff:+.3f} ", fontsize=7, color=TEXT, va="center", ha="right")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([k + (" (not in field)" if k == "ALLD" else "") for k, _ in rows])
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set(
        xlabel="score per round: evolved strategy minus PAV",
        title="E6: where the evolved strategy beats PAV"
    )
    # room for positive and negative bars plus their labels
    lo = min(0, min(d for _, d in rows))
    hi = max(0, max(d for _, d in rows))
    pad = 0.3 * (hi - lo)
    ax.set_xlim(lo - pad if lo < 0 else 0, hi + pad)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.tick_params(colors=MUTED, labelsize=7)

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

def report_E4(data):
    u = data["uniform"]
    keys = data["keys"]
    runs = data["random_runs"]

    def present(shares, digits):
        # strategies still present (share > 1%)
        return ", ".join(f"{k} {v:.{digits}f}" for k, v in zip(keys, shares) if v > 0.01)

    # uniform start
    status = "converged" if u["converged"] else "NOT converged"
    print(f"E4 uniform start: {status} after {u['generations']} generations, "
          f"mutual cooperation {u['cooperation'][-1]:.3f}")
    print("   steady state:", present(u["history"][-1], 3))

    # random starts
    print(f"   random starts: {len(runs)} runs, {sum(not r['converged'] for r in runs)} did not converge")
    print("   survived (share > 1%):",
          ", ".join(f"{k} {sum(r['final'][i] > 0.01 for r in runs)}" for i, k in enumerate(keys)),
          f"(out of {len(runs)})")
    for r in runs:
        if any(r["final"][i] > 0.01 for i, k in enumerate(keys) if "nice" not in Strategies.get(k).traits):
            print("   other steady state:", present(r["final"], 2))

def print_table(headers, rows, indent="    "):
    table = [headers] + [[str(x) for x in row] for row in rows]
    widths = [max(len(row[c]) for row in table) for c in range(len(headers))]
    for row in table:
        print(indent + "  ".join(cell.rjust(w) for cell, w in zip(row, widths)))

def report_E5(data):
    print("E5 initial viability, robustness and stability")
    headers = ["key", "Growth(field)", "Growth(ALLD)", "vsRandomOpp",
               "Survived", "Rank_range", "Resisted/neutral/invaded"]
    rows = []
    for key in data["keys"]:
        v, a = data["viability_field"][key], data["viability_alld"][key]
        r, s = data["robustness"][key], data["stability"][key]
        rows.append([key,
                     f"{v['growth']:.3f}",
                     f"{a['growth']:.3f}",
                     f"{r['vs_random_tables']:.3f}",
                     f"{r['survived']}/{data['n_random_starts']}",
                     f"{r['rank_best']}-{r['rank_worst']}",
                     f"{len(s['resisted'])}/{len(s['neutral'])}/{len(s['invaded_by'])}"])
    print_table(headers, rows)
            
def used_rows(table, opponents, rounds):
    # rule-table rows that are actually looked up when `table` plays each opponent
    used = set()
    for opp in opponents:
        own, other = [], []
        for _ in range(rounds):
            used.add(Strategies.row_index(own, other, 2))
            a = Strategies.next_move(table, own, other)
            b = Strategies.next_move(opp, other, own)
            own.append(a)
            other.append(b)
    return used

def report_E6(data):
    cfg = experiments.load_config()
    game = cfg["game"]
    runs = data["runs"]
    field = [s.table for s in Strategies.fixed_field]

    # 1. the GA runs and the best strategy
    fits = [r["best_fitness"] for r in runs]
    mean = sum(fits) / len(fits)
    sd = (sum((f - mean) ** 2 for f in fits) / len(fits)) ** 0.5
    reached = sum(abs(f - data["best_fitness"]) < 1e-9 for f in fits)
    print(f"E6 {len(runs)} GA runs: best fitness mean {mean:.3f}, sd {sd:.3f}, "
          f"{reached} runs reached {data['best_fitness']:.3f}, "
          f"{sum(r['converged'] for r in runs)} converged")
    print(f"   best table {data['best_table']}  (same as a field strategy: {data['equals_field_strategy']})")
    for rank, name, score in data["ranking"][:5]:
        print(f"   {rank:2d}. {name:4s} {score:.3f}")
    print(f"   final share of GA strategy: {data['final_shares']['GA']:.3f}")

    # 2. where the GA strategy gains: score per round against every opponent, next to PAV
    rows = []
    for opp in Strategies.fixed_field + [Strategies.ALLD]:
        ga = model.play_match(data["best_table"], opp.table, game["rounds"], game["payoffs"])["score_a"]
        pav = model.play_match(Strategies.PAVLOV2.table, opp.table, game["rounds"], game["payoffs"])["score_a"]
        rows.append([opp.key, f"{ga:.3f}", f"{pav:.3f}", f"{ga - pav:+.3f}"])
    print("   score per round against each opponent (ALLD is not in the field):")
    print_table(["opponent", "GA", "PAV", "GA - PAV"], rows, indent="      ")

    # 3. different tables with the same best fitness: they differ only in rows never used
    #    against the field, which still change their score against an outside opponent (ALLD)
    tops = sorted({r["best_table"] for r in runs if abs(r["best_fitness"] - data["best_fitness"]) < 1e-9})
    unused = set(range(len(data["best_table"])))
    for t in tops:
        unused -= used_rows(t, field + [t], game["rounds"])
    differ = [i for i in range(len(tops[0])) if len({t[i] for t in tops}) > 1]
    print(f"   {len(tops)} different tables reach the best fitness; they differ in rows {differ}")
    print(f"   rows never used against the field: {sorted(unused)}")
    vs_alld = {}
    for t in tops:
        score = round(model.play_match(t, Strategies.ALLD.table, game["rounds"], game["payoffs"])["score_a"], 3)
        vs_alld[score] = vs_alld.get(score, 0) + 1
    print("   their score against ALLD:", ", ".join(f"{sc:.3f} ({n} tables)" for sc, n in sorted(vs_alld.items())))

    # 4. validation with memory 1
    if "memory1_check" in data:
        m1 = data["memory1_check"]
        print(f"   memory-1 check passed = {m1['passed']}")
        print("   best memory-1 tables (exhaustive):",
              ", ".join(f"{t} {f:.3f}" for f, t in m1["exhaustive_top3"]))


if __name__ == "__main__":
    cfg = experiments.load_config()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    # name: (run function, list of panel functions, report function); one figure file per experiment
    jobs = {"E1": (experiments.run_E1, [fig_E1_matrix], report_E1),
            "E2": (experiments.run_E2, [fig_E2, fig_E2_population], report_E2),
            "E3": (experiments.run_E3, [fig_E3, fig_E3_named], report_E3),
            "E4": (experiments.run_E4, [fig_E4, fig_E4_steady_states], report_E4),
            "E5": (experiments.run_E5, [fig_E5], report_E5),
            "E6": (experiments.run_E6, [fig_E6, fig_E6_advantage], report_E6)}
    for name in (jobs if which == "all" else [which]):
        run, panels, report = jobs[name]
        data = run(cfg)
        report(data)
        print("   figure:", save_figure(panels, data, cfg, name))