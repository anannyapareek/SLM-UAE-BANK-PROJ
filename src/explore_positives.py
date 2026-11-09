"""When do positives happen, and how would candidate four-way splits look?

Label-side analysis: reads labels.parquet and malicious_events.parquet, which feature code
never reads. It writes figures and prints tables only. It does not build a split.
Run with: python run.py explore_positives
"""
import os
import pathlib

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

P = pathlib.Path(os.environ.get("PROCESSED_DIR", "data/processed"))
FIGURES = pathlib.Path("results/figures/explore")
GLOBAL_CONFIG = pathlib.Path("configs/global.yaml")
SPLITS = ["train", "validation", "calibration", "test"]
# Each candidate is (validation start, calibration start, test start), whole days.
CANDIDATES = {
    "A": ("2010-09-01", "2010-11-01", "2011-01-01"),
    "B": ("2010-10-01", "2010-12-01", "2011-02-01"),
    "C": ("2010-11-01", "2011-01-01", "2011-03-01"),
}
HOLDOUT_FRACTION = 0.175
MIN_POSITIVE_SESSIONS = 10
# Categorical slots 1-3 of the reference palette, validated all-pairs in light mode.
SCENARIO_COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
SCENARIO_MARKERS = {1: "o", 2: "s", 3: "^"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(con):
    sessions = con.execute(f"""
        SELECT l.session_id, any_value(l."user") AS "user", any_value(s.start) AS start,
               count(*) AS rows, sum(l.y) AS positive_rows,
               max(l.in_positive_session) AS positive, any_value(l.scenario) AS scenario,
               any_value(l.incident_id) AS incident_id, any_value(l.actor_user) AS actor_user
        FROM '{P}/labels.parquet' l JOIN '{P}/sessions.parquet' s USING (session_id)
        GROUP BY l.session_id
    """).df()
    sessions["day"] = sessions["start"].dt.normalize()
    involved = con.execute(f"""
        SELECT DISTINCT incident_id, "user" FROM '{P}/malicious_events.parquet'
    """).df()
    return sessions, involved


def table(header, rows):
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = lambda r: " | ".join(str(x).ljust(w) for x, w in zip(r, widths))
    return "\n".join([line(header), "-+-".join("-" * w for w in widths)] + [line(r) for r in rows])


def print_overview(sessions):
    print(f"scoring sessions: {sessions['start'].min():%Y-%m-%d} to "
          f"{sessions['start'].max():%Y-%m-%d} (session start)")
    month = sessions["start"].dt.to_period("M")
    by_month = sessions.groupby(month).agg(
        sessions=("session_id", "size"), positive_sessions=("positive", "sum"),
        positive_rows=("positive_rows", "sum"),
        **{f"scenario_{s}": ("scenario", lambda x, s=s: int((x == s).sum())) for s in SCENARIO_COLOURS})
    print("\nsessions and positives per month (by session start)")
    print(table(["month", *by_month.columns],
                [[str(m), *(f"{int(v):,}" for v in r)] for m, r in by_month.iterrows()]))
    return by_month


def plot_months(by_month):
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(by_month))
    bottom = np.zeros(len(by_month))
    for s, colour in SCENARIO_COLOURS.items():
        values = by_month[f"scenario_{s}"].to_numpy()
        ax.bar(x, values, 0.7, bottom=bottom, color=colour, edgecolor=SURFACE, linewidth=2,
               label=f"Scenario {s}")
        bottom += values
    for xi, total in zip(x, bottom):
        if total:
            ax.text(xi, total + 3, f"{int(total)}", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(x, [str(m) for m in by_month.index], rotation=45, ha="right", fontsize=8,
                  color=MUTED)
    ax.set_ylabel("Positive sessions", color=MUTED)
    ax.set_title("Positive sessions per month, by scenario", loc="left", color=INK, fontsize=11)
    style(ax, grid="y")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "positive_sessions_per_month.png")


def plot_timeline(sessions):
    positive = sessions[sessions["positive"] == 1]
    spans = (positive.groupby("incident_id")
             .agg(first=("start", "min"), last=("start", "max"), scenario=("scenario", "first"))
             .sort_values(["first", "last"]))
    fig, ax = plt.subplots(figsize=(9, 9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (incident, r) in enumerate(spans.iterrows()):
        colour = SCENARIO_COLOURS[r["scenario"]]
        ax.plot([r["first"], r["last"]], [y, y], color=colour, linewidth=2, solid_capstyle="round")
        ax.scatter([r["first"], r["last"]], [y, y], s=28, color=colour,
                   marker=SCENARIO_MARKERS[r["scenario"]], edgecolor=SURFACE, linewidth=1, zorder=3)
    ax.set_yticks(range(len(spans)), [str(i) for i in spans.index], fontsize=6, color=MUTED)
    ax.set_ylim(len(spans), -1)
    ax.set_ylabel("Incident id (one per insider), ordered by first malicious session", color=MUTED)
    ax.set_title("First and last malicious session per insider", loc="left", color=INK, fontsize=11)
    for s, colour in SCENARIO_COLOURS.items():
        ax.scatter([], [], color=colour, marker=SCENARIO_MARKERS[s], label=f"Scenario {s}")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    style(ax, grid="x")
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    save(fig, "insider_timeline.png")


def style(ax, grid):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d0")
    ax.grid(axis=grid, color="#e8e7e3", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {FIGURES / name}")


def held_out_users(sessions, involved, seed, active_from=None):
    """Same stratified hold-out as the split itself (src/splits.py)."""
    from src.splits import held_out_users as split_held_out
    return split_held_out(sessions, involved, seed, HOLDOUT_FRACTION, active_from)


def assign(sessions, bounds, held):
    edges = [pd.Timestamp(b) for b in bounds]
    period = np.select([sessions["day"] < edges[0], sessions["day"] < edges[1],
                        sessions["day"] < edges[2]], SPLITS[:3], SPLITS[3])
    keep = (~sessions["user"].isin(held)) | (period == "test")
    return pd.Series(np.where(keep, period, "dropped"), index=sessions.index)


def summarise(name, bounds, sessions, held, held_incidents):
    split = assign(sessions, bounds, held)
    rows, scenario_rows, flags = [], [], []
    for s in SPLITS + ["dropped"]:
        part = sessions[split == s]
        positive = part[part["positive"] == 1]
        if part.empty:
            rows.append([s, "-", 0, 0, 0, 0, 0, 0, 0, "-"])
            continue
        scenarios = ",".join(str(int(x)) for x in sorted(positive["scenario"].dropna().unique()))
        rows.append([s, f"{part['day'].min():%Y-%m-%d} to {part['day'].max():%Y-%m-%d}",
                     part["user"].nunique(), f"{len(part):,}", f"{part['rows'].sum():,}",
                     len(positive), f"{int(positive['positive_rows'].sum()):,}",
                     positive["incident_id"].nunique(), positive["actor_user"].nunique(),
                     scenarios or "-"])
        scenario_rows.append([s] + [
            f"{int((positive['scenario'] == sc).sum())} / "
            f"{int(positive.loc[positive['scenario'] == sc, 'positive_rows'].sum()):,} / "
            f"{positive.loc[positive['scenario'] == sc, 'incident_id'].nunique()}"
            for sc in SCENARIO_COLOURS])
        if s in SPLITS and len(positive) == 0:
            flags.append(f"{s} has no positives")
        if s in ("validation", "calibration") and len(positive) < MIN_POSITIVE_SESSIONS:
            flags.append(f"{s} has only {len(positive)} positive sessions")
        if s in SPLITS:
            for sc in SCENARIO_COLOURS:
                if not (positive["scenario"] == sc).any():
                    flags.append(f"{s} has no scenario {sc}")

    print(f"\n=== Candidate {name}: validation from {bounds[0]}, calibration from {bounds[1]}, "
          f"test from {bounds[2]}")
    print(table(["split", "date range", "users", "sessions", "rows", "positive sessions",
                 "positive rows", "incidents", "insiders", "scenarios present"], rows))
    print("\npositives by scenario (positive sessions / positive rows / incidents)")
    print(table(["split", *(f"scenario {s}" for s in SCENARIO_COLOURS)], scenario_rows))

    positive = sessions[sessions["positive"] == 1]
    spread = positive.assign(split=split[positive.index]).query("split != 'dropped'")
    crossing = (spread.groupby("incident_id")["split"].nunique() > 1).sum()
    held_in_test = positive[positive["incident_id"].isin(held_incidents)
                            & (split[positive.index] == "test")]
    print(f"\nincidents with positive sessions in more than one split: {crossing}")
    print(f"held-out incidents: {len(held_incidents)}; with positives in test: "
          f"{held_in_test['incident_id'].nunique()}; their positive sessions in test: "
          f"{len(held_in_test)}")
    print("flags: " + ("; ".join(flags) if flags else "none"))


def run():
    # Single-threaded: multi-threaded writes corrupted Parquet output (see decisions_log.md).
    con = duckdb.connect(config={"threads": 1})
    sessions, involved = load(con)
    by_month = print_overview(sessions)
    plot_months(by_month)
    plot_timeline(sessions)

    seed = yaml.safe_load(GLOBAL_CONFIG.read_text())["seed"]
    held, held_incidents = held_out_users(sessions, involved, seed)
    held_positive = sessions[(sessions["positive"] == 1) & sessions["incident_id"].isin(held_incidents)]
    print(f"\nheld-out users (seed {seed}, {HOLDOUT_FRACTION:.1%} per stratum): {len(held)} of "
          f"{sessions['user'].nunique()}; held-out incidents by scenario: "
          f"{held_positive.groupby('scenario')['incident_id'].nunique().to_dict()}")
    for name, bounds in CANDIDATES.items():
        summarise(name, bounds, sessions, held, held_incidents)

    # Variant: hold out insiders whose incidents are still active in the test period, so
    # held-out positives land in test instead of being dropped.
    name, bounds = "A", CANDIDATES["A"]
    held, held_incidents = held_out_users(sessions, involved, seed, active_from=bounds[2])
    print(f"\nVARIANT: candidate {name} with insiders held out only from incidents active from "
          f"{bounds[2]}: {len(held)} users held out")
    summarise(f"{name}-active", bounds, sessions, held, held_incidents)
