"""generate docs/PARITY_REPORT.md from data/parity_test_results.csv.

usage:
    python scripts/parity_report.py           # regenerate markdown from existing CSV
    python scripts/parity_report.py --rebuild  # re-run full comparison against DB (slow)

the csv must already exist unless --rebuild is passed.
"""

import argparse
import os
import sys
from pathlib import Path

CSV_PATH = Path("data/parity_test_results.csv")
OUT_PATH = Path("docs/PARITY_REPORT.md")
REF_PARQUET = "data/features.parquet"
SINCE = "2026-07-26"
UNTIL = "2026-08-31"


def _category(feature: str) -> str:
    if feature in ("lag_24h", "lag_48h", "lag_168h"):
        return "lag"
    if feature in ("lag_24h_ratio", "avail_std", "reb_per_week"):
        return "station-stat"
    if feature == "hours_since_reb":
        return "sparse"
    if feature.startswith("lag"):
        return "lag"
    return "exact"


def _tolerance_badge(row: dict) -> str:
    cat = _category(row["feature"])
    corr = float(row["corr"]) if row["corr"] else None
    mean_d = float(row["mean_absdiff"])

    if cat == "exact":
        ok = (corr is None or corr >= 0.9999) and mean_d < 0.001
    elif cat == "lag":
        ok = corr is not None and corr >= 0.998 and mean_d < 0.1
    elif cat == "station-stat":
        ok = corr is not None and corr >= 0.70
    else:  # sparse
        ok = corr is None or corr >= 0.99

    return "PASS" if ok else "FAIL"


def _build_csv(since: str, until: str, ref: str) -> None:
    """run the full parity comparison and write data/parity_test_results.csv."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import numpy as np
    import pandas as pd
    from src.features import build_features

    print(f"[parity] building features {since} to {until} ...")
    built = build_features(since=since, until=until, geo_parquet=ref)

    print(f"[parity] loading reference from {ref}")
    ref_df = pd.read_parquet(ref)
    ref_df["hour_local"] = pd.to_datetime(ref_df.get("hour_local", ref_df.get("hour", None)))
    built["hour_local"] = pd.to_datetime(built["hour_local"])

    merge_key = ["station_uid", "hour_local"]
    merged = ref_df.merge(built, on=merge_key, suffixes=("_ref", "_built"), how="inner")

    feature_cols = [c for c in built.columns if c not in merge_key + ["avg_available"]]
    rows = []
    for feat in feature_cols:
        col_ref  = f"{feat}_ref"  if f"{feat}_ref"  in merged.columns else feat
        col_built = f"{feat}_built" if f"{feat}_built" in merged.columns else feat
        if col_ref not in merged.columns or col_built not in merged.columns:
            continue
        a = pd.to_numeric(merged[col_ref],   errors="coerce")
        b = pd.to_numeric(merged[col_built], errors="coerce")
        both = a.notna() & b.notna()
        n = int(both.sum())
        corr = float(a[both].corr(b[both])) if n > 10 else None
        mean_d = float((a[both] - b[both]).abs().mean()) if n > 0 else np.nan
        max_d  = float((a[both] - b[both]).abs().max())  if n > 0 else np.nan
        rows.append({
            "feature": feat,
            "ref_nan":    round(a.isna().mean(), 4),
            "built_nan":  round(b.isna().mean(), 4),
            "overlap_n":  n,
            "corr":       round(corr, 4) if corr is not None else "",
            "mean_absdiff": round(mean_d, 4),
            "max_absdiff":  round(max_d, 4),
            "note": "",
        })

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
    print(f"[parity] wrote {len(rows)} rows to {CSV_PATH}")


def _write_markdown() -> None:
    import csv

    with open(CSV_PATH, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    lines = [
        "# feature parity report",
        "",
        "comparison between `src/features.py` output and `data/features.parquet` (reference).",
        "",
        "**groups**:",
        "- **exact** — geo, time, weather: corr ≥ 0.9999 and mean diff < 0.001",
        "- **lag** — lag_24h/48h/168h: corr ≥ 0.998 and mean diff < 0.1",
        "- **station-stat** — avail_std, reb_per_week, lag_24h_ratio: corr ≥ 0.70 (improve after training-window stats refactor)",
        "- **sparse** — hours_since_reb (97% NaN): corr ≥ 0.99 on non-null rows",
        "",
        "| feature | group | status | overlap_n | corr | mean_diff | max_diff | ref_nan | built_nan |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        badge = _tolerance_badge(row)
        cat   = _category(row["feature"])
        corr  = row["corr"] if row["corr"] else "—"
        lines.append(
            f"| {row['feature']} | {cat} | {badge} "
            f"| {row['overlap_n']} | {corr} "
            f"| {row['mean_absdiff']} | {row['max_absdiff']} "
            f"| {row['ref_nan']} | {row['built_nan']} |"
        )

    n_fail = sum(1 for r in rows if _tolerance_badge(r) == "FAIL")
    n_pass = len(rows) - n_fail
    lines += [
        "",
        f"**{n_pass} features pass, {n_fail} features fail** (see tolerance thresholds above).",
        "",
        "_generated by `scripts/parity_report.py`_",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[parity] wrote {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild", action="store_true",
        help="re-run full feature comparison (requires DB)",
    )
    parser.add_argument("--since", default=SINCE)
    parser.add_argument("--until", default=UNTIL)
    parser.add_argument("--ref",   default=REF_PARQUET)
    args = parser.parse_args()

    if args.rebuild:
        _build_csv(args.since, args.until, args.ref)
    elif not CSV_PATH.exists():
        print(f"error: {CSV_PATH} not found — run with --rebuild to generate it", file=sys.stderr)
        sys.exit(1)

    _write_markdown()
