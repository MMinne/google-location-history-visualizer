"""
Google Location History – Temporal Patterns
Thesis: "Inferring Sensitive Personal Information from Google Location History
         Using Data Visualization Techniques"
Author: Matthijs Hagedoorn

=== HOW TO USE ===

1. Export your Google Location History as a single JSON file.
   For instructions on how to do this, see:
   https://support.google.com/maps/answer/6258979?hl=en&co=GENIE.Platform%3DAndroid

2. Place the exported JSON file in the same folder as this script
   and set JSON_FILE in CONFIGURATION below to match its filename.

3. Adjust START_DATE and END_DATE to your desired date range.

4. Run the script:
       python GH_temporal_patterns.py

5. Find the output image in:
       thesis_visualizations/temporal_patterns.png
   (The folder is created automatically if it does not exist.)

=== DEPENDENCIES ===
    pip install pandas numpy matplotlib
"""

import json
import re
import os
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0.  CONFIGURATION — only this section needs editing
# ─────────────────────────────────────────────

JSON_FILE  = "Timeline.json"  # Place your JSON export in the same folder as this script and rename it to Timeline.json
OUTPUT_DIR = "thesis_visualizations"  # Output folder — created automatically if it doesn't exist
START_DATE = datetime(2023, 2, 25, tzinfo=timezone.utc)   # Adjust to your data range
END_DATE   = datetime(2026, 2, 25, tzinfo=timezone.utc)   # Adjust to your data range

LOCAL_UTC_OFFSET = 1    # Adjust for your timezone: Netherlands = UTC+1 (winter) / UTC+2 (summer)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1.  DATA LOADING & PARSING
# ─────────────────────────────────────────────

def parse_latlng(s: str) -> tuple[float, float]:
    nums = re.findall(r"[-+]?\d+\.\d+", s)
    return float(nums[0]), float(nums[1])


def parse_time(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except ValueError:
        s2 = re.sub(r"([+-]\d{2}):(\d{2})$", r"\1\2", s)
        return datetime.strptime(s2, "%Y-%m-%dT%H:%M:%S.%f%z").astimezone(timezone.utc)


def load_segments(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find data at: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("semanticSegments", data) if isinstance(data, dict) else data
    return segments


def segments_to_dataframes(segments: list[dict]):
    path_rows, visit_rows, act_rows = [], [], []

    for seg in segments:
        try:
            t_start = parse_time(seg["startTime"])
            t_end   = parse_time(seg["endTime"])
        except Exception:
            continue

        if t_end < START_DATE or t_start > END_DATE:
            continue

        if "timelinePath" in seg:
            for pt in seg["timelinePath"]:
                try:
                    lat, lon = parse_latlng(pt["point"])
                    t = parse_time(pt["time"])
                    path_rows.append({"time": t, "lat": lat, "lon": lon})
                except Exception:
                    continue

        elif "visit" in seg:
            v  = seg["visit"]
            tc = v.get("topCandidate", {})
            loc = tc.get("placeLocation", {}).get("latLng", "")
            if not loc:
                continue
            try:
                lat, lon = parse_latlng(loc)
            except Exception:
                continue
            duration_min = (t_end - t_start).total_seconds() / 60
            sem_type = tc.get("semanticType", "UNKNOWN").upper()
            visit_rows.append({
                "start": t_start, "end": t_end,
                "duration_min": duration_min,
                "lat": lat, "lon": lon,
                "semantic_type": sem_type,
                "place_id": tc.get("placeId", ""),
                "probability": tc.get("probability", 0),
            })

        elif "activity" in seg:
            a = seg["activity"]
            try:
                slat, slon = parse_latlng(a["start"]["latLng"])
                elat, elon = parse_latlng(a["end"]["latLng"])
            except Exception:
                continue
            duration_min = (t_end - t_start).total_seconds() / 60
            act_rows.append({
                "start": t_start, "end": t_end,
                "duration_min": duration_min,
                "start_lat": slat, "start_lon": slon,
                "end_lat": elat, "end_lon": elon,
                "distance_m": a.get("distanceMeters", 0),
                "mode": a.get("topCandidate", {}).get("type", "UNKNOWN"),
            })

    path_df  = pd.DataFrame(path_rows)
    visit_df = pd.DataFrame(visit_rows)
    act_df   = pd.DataFrame(act_rows)

    for df in [path_df, visit_df, act_df]:
        if df.empty:
            continue
        time_col = "time" if "time" in df.columns else "start"
        df["hour"]    = df[time_col].dt.hour
        df["weekday"] = df[time_col].dt.day_name()
        if time_col == "start":
            df["month"] = df["start"].dt.to_period("M")
            df["year"]  = df["start"].dt.year
            df["date"]  = df["start"].dt.date

    return path_df, visit_df, act_df


# ─────────────────────────────────────────────
# 2.  VIZ 2 – TEMPORAL ACTIVITY PLOTS
# ─────────────────────────────────────────────

def make_temporal_plots(path_df: pd.DataFrame, visit_df: pd.DataFrame, act_df: pd.DataFrame):
    """
    Three-panel temporal overview:
      A – Activity by hour of day (raw GPS datapoints from path_df, i.e. 'activity')
      B – Visit frequency heatmap: weekday x hour (from visit_df)
      C – Monthly visit count bar chart, full Feb 2023 → Feb 2026 range
    """
    print("  Building temporal activity plots ...")

    WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    BG   = "#0f0e17"
    FG   = "#fffffe"
    ACC1 = "#ff8906"   # raw activity points (orange)
    ACC3 = "#ef4565"   # monthly bars (red)

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        "Temporal Mobility Patterns  |  Feb 2023 - Feb 2026",
        color=FG, fontsize=16, fontweight="bold", y=0.97
    )
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.07, right=0.96, top=0.92, bottom=0.08)

    def style_ax(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=FG, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        ax.set_title(title, color=FG, fontsize=11, pad=8, fontweight="bold")
        ax.yaxis.label.set_color(FG)
        ax.xaxis.label.set_color(FG)

    # ── A) Hour-of-day – raw GPS datapoints (path_df = "activity" in thesis) ──
    ax_a = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, 25)
    hours_p = path_df["hour"].dropna().values if not path_df.empty else np.array([])
    counts_p, _ = np.histogram(hours_p, bins=bins)
    x = bins[:-1]
    ax_a.bar(x, counts_p, color=ACC1, alpha=0.85, label="Activity (raw datapoints)",
             width=0.9, edgecolor="none")
    ax_a.set_xlabel("Hour of Day")
    ax_a.set_ylabel("Count")
    ax_a.set_xticks(range(0, 24, 2))
    ax_a.legend(framealpha=0, labelcolor=FG, fontsize=9)
    style_ax(ax_a, "A . Activity by Hour of Day")

    # ── B) Weekday x Hour heatmap (visit_df, unchanged) ──
    ax_b = fig.add_subplot(gs[0, 1])
    if not visit_df.empty:
        pivot = visit_df.groupby(["weekday", "hour"]).size().reset_index(name="count")
        grid  = pd.DataFrame(0, index=WEEKDAY_ORDER, columns=range(24))
        for _, r in pivot.iterrows():
            if r["weekday"] in grid.index:
                grid.loc[r["weekday"], r["hour"]] = r["count"]
        im = ax_b.imshow(
            grid.values, aspect="auto", cmap="YlOrRd",
            extent=[-0.5, 23.5, -0.5, 6.5], origin="upper"
        )
        cbar = fig.colorbar(im, ax=ax_b, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=FG, labelsize=8)
        cbar.set_label("Visit Count", color=FG, fontsize=9)
        ax_b.set_yticks(range(7))
        ax_b.set_yticklabels(WEEKDAY_ORDER[::-1], fontsize=8)
        ax_b.set_xticks(range(0, 24, 2))
        ax_b.set_xlabel("Hour of Day")
    style_ax(ax_b, "B . Visit Frequency: Weekday x Hour")

    # ── C) Monthly bar chart – full date range (visit_df, unchanged) ──
    ax_c = fig.add_subplot(gs[1, :])
    if not visit_df.empty:
        full_months = pd.period_range(
            START_DATE.strftime("%Y-%m"),
            END_DATE.strftime("%Y-%m"), freq="M"
        )
        monthly_counts = (
            visit_df.groupby(visit_df["start"].dt.to_period("M"))
            .size()
            .reindex(full_months, fill_value=0)
            .rename("visits")
            .reset_index()
        )
        monthly_counts.columns = ["month", "visits"]
        monthly_counts["label"] = monthly_counts["month"].astype(str)
        x = np.arange(len(monthly_counts))

        ax_c.bar(x, monthly_counts["visits"], color=ACC3, alpha=0.85, width=0.8)

        for i, row in monthly_counts.iterrows():
            if i % 3 == 0:
                ax_c.text(i, row["visits"] + 1, row["label"][:7],
                          color=FG, fontsize=7, ha="center", va="bottom", rotation=45)

        ax_c.set_xticks(x[::3])
        ax_c.set_xticklabels(monthly_counts["label"][::3], rotation=45, ha="right", fontsize=8)
        ax_c.set_xlim(-0.5, len(monthly_counts) - 0.5)
        ax_c.set_ylabel("Number of Visits")
        ax_c.set_xlabel("Month")

        first_nonzero_rows = monthly_counts[monthly_counts["visits"] > 0]
        if not first_nonzero_rows.empty:
            first_nonzero = first_nonzero_rows["month"].iloc[0]
            if str(first_nonzero) > "2023-02":
                ax_c.annotate(
                    f"No data before {first_nonzero}",
                    xy=(0, 0), xytext=(1, monthly_counts["visits"].max() * 0.85),
                    color="#adb5bd", fontsize=8, fontstyle="italic"
                )
    style_ax(ax_c, "C . Monthly Visit Count  (Feb 2023 - Feb 2026)")

    out = os.path.join(OUTPUT_DIR, "temporal_patterns.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"    Saved -> {out}")


# ─────────────────────────────────────────────
# 3.  MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Google Location History – Temporal Patterns (Viz 2)")
    print("=" * 55)

    print(f"\n[1/3] Loading segments from: {JSON_FILE}")
    segments = load_segments(JSON_FILE)
    print(f"      Loaded {len(segments):,} raw segments")

    print(f"\n[2/3] Parsing  ({START_DATE.date()} -> {END_DATE.date()})")
    path_df, visit_df, act_df = segments_to_dataframes(segments)
    print(f"      GPS points (activity) : {len(path_df):,}")
    print(f"      Visits                : {len(visit_df):,}")
    print(f"      Movement segments     : {len(act_df):,}")

    if visit_df.empty and path_df.empty:
        print("\n  No data after filtering. Check date range or file path.")
        return

    print("\n[3/3] Viz 2 – Temporal Patterns")
    make_temporal_plots(path_df, visit_df, act_df)

    print(f"\n  Output saved to: ./{OUTPUT_DIR}/temporal_patterns.png")
    print("=" * 55)


if __name__ == "__main__":
    main()
