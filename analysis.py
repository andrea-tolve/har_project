"""Focused exploratory data analysis for the HAR experiments.

The script reads both the original experiments and the reduced-body-part
experiments stored under ``results/<context>/``. It deliberately produces a
small, presentation-oriented set of tables and plots instead of exporting
one figure for every possible grouping.

Supported result sources:

* ``results/<context>/summary.csv``: standard seven-body-part experiment;
* ``results/<context>/<experiment>/summary.csv``: reduced-body-part result;
* ``results/<context>/fold_results.csv``: fallback for the standard experiment;
* ``results/<context>/<experiment>/folderN/<model>/metrics.csv``: fallback
  when an experiment has no summary file.

Output layout:

* ``analysis/tables``: CSV tables;
* ``analysis/plots``: presentation-ready figures;
* ``analysis/summary``: short textual and numeric summaries.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "analysis"
TABLES_DIR = OUTPUT_DIR / "tables"
PLOTS_DIR = OUTPUT_DIR / "plots"
SUMMARY_DIR = OUTPUT_DIR / "summary"
for directory in (TABLES_DIR, PLOTS_DIR, SUMMARY_DIR):
    directory.mkdir(parents=True, exist_ok=True)

CONTEXTS = ["baseline", "gps", "light", "mic", "all_context"]
MODELS = ["random_forest", "svm", "xgboost"]
ACTIVITIES = ["lying", "running", "sitting", "standing", "walking"]
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
FOLD_RE = re.compile(r"(?:folder|fold)[_-]?(\d+)$", re.IGNORECASE)

# Labels used in plots and tables.
CONTEXT_LABELS = {
    "baseline": "Baseline",
    "gps": "Baseline + GPS",
    "light": "Baseline + light",
    "mic": "Baseline + microphone",
    "all_context": "Baseline + all context",
}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "svm": "SVM",
    "xgboost": "XGBoost",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def canonical_model(value):
    value = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "randomforest": "random_forest",
        "random_forest": "random_forest",
        "rf": "random_forest",
        "svm": "svm",
        "support_vector_machine": "svm",
        "support_vector_classifier": "svm",
        "xgboost": "xgboost",
        "xg_boost": "xgboost",
    }.get(value, value)


def save_csv(dataframe, filename):
    path = TABLES_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"[OK] Saved {path}")
    return dataframe


def save_figure(filename):
    path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {path}")


def save_summary(text, filename="eda_summary.txt"):
    path = SUMMARY_DIR / filename
    path.write_text(text, encoding="utf-8")
    print(f"[OK] Saved {path}")


def fold_number(path):
    match = FOLD_RE.search(path.name)
    return int(match.group(1)) if match else np.nan


def display_model(value):
    return MODEL_LABELS.get(value, str(value))


def parse_experiment_name(name, context):
    """Return metadata describing an experiment directory.

    The returned ``body_part_count`` is the number of inertial body parts.
    ``only_*`` means one selected body part; ``first_N_*`` means a progressive
    combination. The standard directory is represented by seven body parts.
    """
    name = str(name)
    if name == "standard":
        return {
            "experiment": "standard",
            "body_part_count": 7,
            "body_part_mode": "all_7",
            "body_part": "all_7",
            "context": context,
        }

    only_match = re.match(r"only_(.+?)(?:_plus_(?:gps|light|mic|all))?$", name)
    if only_match:
        body_part = only_match.group(1)
        return {
            "experiment": name,
            "body_part_count": 1,
            "body_part_mode": "single",
            "body_part": body_part,
            "context": context,
        }

    first_match = re.match(r"first_(\d+)_(?:body_parts|parts)(?:_plus_(?:gps|light|mic|all))?$", name)
    if first_match:
        count = int(first_match.group(1))
        return {
            "experiment": name,
            "body_part_count": count,
            "body_part_mode": "progressive",
            "body_part": f"first_{count}",
            "context": context,
        }

    return {
        "experiment": name,
        "body_part_count": np.nan,
        "body_part_mode": "other",
        "body_part": name,
        "context": context,
    }


def summary_from_metrics(experiment_dir, metadata):
    """Reconstruct a summary from detailed fold metrics when needed."""
    rows = []
    for metrics_path in experiment_dir.glob("folder*/**/metrics.csv"):
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception as exc:
            warnings.warn(f"Could not read {metrics_path}: {exc}")
            continue
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        row.update(metadata)
        row["model"] = canonical_model(row.get("model", metrics_path.parent.name))
        row["fold"] = row.get("fold", fold_number(metrics_path.parent.parent))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    detail = pd.DataFrame(rows)
    group_columns = ["context", "experiment", "body_part_count", "body_part_mode", "body_part", "model"]
    aggregation = {metric: ["mean", "std", "median"] for metric in METRICS if metric in detail.columns}
    if not aggregation:
        return pd.DataFrame()
    summary = detail.groupby(group_columns, dropna=False).agg(aggregation).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if str(part) not in ("", "None"))
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    return summary


# ============================================================
# LOADING RESULTS
# ============================================================

def load_eda_results():
    """Load standard and body-part experiment summaries into one long table."""
    frames = []
    detail_frames = []

    for context in CONTEXTS:
        context_dir = RESULTS_DIR / context
        if not context_dir.exists():
            warnings.warn(f"Missing context directory: {context_dir}")
            continue

        # The root summary is the seven-body-part reference experiment.
        root_summary = context_dir / "summary.csv"
        root_fold_results = context_dir / "fold_results.csv"
        metadata = parse_experiment_name("standard", context)
        if root_summary.exists():
            summary = pd.read_csv(root_summary)
            summary["context"] = context
            summary["experiment"] = "standard"
            summary["body_part_count"] = 7
            summary["body_part_mode"] = "all_7"
            summary["body_part"] = "all_7"
            summary["model"] = summary["model"].map(canonical_model)
            frames.append(summary)
        elif root_fold_results.exists():
            fold_results = pd.read_csv(root_fold_results)
            fold_results["model"] = fold_results["model"].map(canonical_model)
            aggregation = {metric: ["mean", "std", "median"]
                           for metric in METRICS if metric in fold_results.columns}
            summary = fold_results.groupby("model").agg(aggregation).reset_index()
            summary.columns = [
                "_".join(str(part) for part in column if str(part) not in ("", "None"))
                if isinstance(column, tuple) else column
                for column in summary.columns
            ]
            summary["context"] = context
            summary["experiment"] = "standard"
            summary["body_part_count"] = 7
            summary["body_part_mode"] = "all_7"
            summary["body_part"] = "all_7"
            frames.append(summary)

        # Named body-part experiments.
        for experiment_dir in sorted(context_dir.iterdir()):
            if not experiment_dir.is_dir() or experiment_dir.name.startswith("folder"):
                continue
            metadata = parse_experiment_name(experiment_dir.name, context)
            summary_path = experiment_dir / "summary.csv"
            if summary_path.exists():
                summary = pd.read_csv(summary_path)
                summary["context"] = context
                for key in ("experiment", "body_part_count", "body_part_mode", "body_part"):
                    summary[key] = metadata[key]
                summary["model"] = summary["model"].map(canonical_model)
                frames.append(summary)
            else:
                reconstructed = summary_from_metrics(experiment_dir, metadata)
                if not reconstructed.empty:
                    frames.append(reconstructed)

            # Keep fold metrics for variability and paired comparisons.
            for metrics_path in experiment_dir.glob("folder*/**/metrics.csv"):
                try:
                    metrics = pd.read_csv(metrics_path)
                except Exception:
                    continue
                if metrics.empty:
                    continue
                metrics["context"] = context
                for key in ("experiment", "body_part_count", "body_part_mode", "body_part"):
                    metrics[key] = metadata[key]
                metrics["model"] = metrics["model"].map(canonical_model)
                metrics["fold"] = metrics.get("fold", fold_number(metrics_path.parent.parent))
                detail_frames.append(metrics)

    if not frames:
        raise FileNotFoundError("No experiment summaries were found in results/.")

    results = pd.concat(frames, ignore_index=True)
    results["model"] = results["model"].map(canonical_model)
    results["context"] = results["context"].astype(str)
    results["body_part_count"] = pd.to_numeric(results["body_part_count"], errors="coerce")
    results = results.drop_duplicates(
        subset=["context", "experiment", "model"], keep="last"
    ).reset_index(drop=True)

    save_csv(results.sort_values(["context", "body_part_count", "model"]),
             "eda_experiment_summary.csv")

    if detail_frames:
        details = pd.concat(detail_frames, ignore_index=True)
        save_csv(details.sort_values(["context", "experiment", "model", "fold"]),
                 "eda_fold_metrics.csv")
    else:
        details = pd.DataFrame()
    return results, details


# ============================================================
# TABLES FOR THE PRESENTATION
# ============================================================

def select_metric_column(results, metric="f1_macro"):
    candidates = [f"{metric}_mean", metric]
    for candidate in candidates:
        if candidate in results.columns:
            return candidate
    raise KeyError(f"No summary column found for metric {metric!r}.")


def create_global_ranking(results):
    metric = select_metric_column(results)
    table = results[["context", "experiment", "body_part_count", "body_part_mode",
                     "body_part", "model", metric]].copy()
    table = table.rename(columns={metric: "macro_f1_mean"})
    table["context_label"] = table["context"].map(lambda value: CONTEXT_LABELS.get(value, value))
    table["model_label"] = table["model"].map(display_model)
    table = table.sort_values("macro_f1_mean", ascending=False)
    table["rank"] = range(1, len(table) + 1)
    return save_csv(table, "ranking_all_experiments.csv")


def create_body_part_single_ranking(results):
    metric = select_metric_column(results)
    table = results[(results["body_part_mode"] == "single") &
                    (results["context"] == "baseline")].copy()
    if table.empty:
        return table
    table = table[["body_part", "model", metric]].rename(columns={metric: "macro_f1_mean"})
    table["model_label"] = table["model"].map(display_model)
    table = table.sort_values(["model", "macro_f1_mean"], ascending=[True, False])
    table["rank_within_model"] = table.groupby("model")["macro_f1_mean"].rank(
        ascending=False, method="min"
    ).astype(int)
    return save_csv(table, "ranking_single_body_parts_baseline.csv")


def create_body_part_curve(results):
    metric = select_metric_column(results)
    table = results[results["body_part_mode"].isin(["progressive", "all_7"])].copy()
    if table.empty:
        return table
    table = table[["context", "experiment", "body_part_count", "body_part_mode",
                   "model", metric]].rename(columns={metric: "macro_f1_mean"})
    table["context_label"] = table["context"].map(lambda value: CONTEXT_LABELS.get(value, value))
    table["model_label"] = table["model"].map(display_model)
    return save_csv(table.sort_values(["context", "model", "body_part_count"]),
                    "body_part_count_curve.csv")


def create_context_gain_table(results):
    """Compare baseline and context at equal body-part counts.

    The comparison is valid only where the same body-part experiment exists in
    both contexts. For each context, the table reports context F1 minus baseline
    F1 for each model and body-part setup.
    """
    metric = select_metric_column(results)
    keys = ["experiment", "body_part_count", "body_part_mode", "body_part", "model"]
    baseline = results[results["context"] == "baseline"][keys + [metric]].rename(
        columns={metric: "baseline_macro_f1"}
    )
    rows = []
    for context in ["gps", "light", "mic", "all_context"]:
        current = results[results["context"] == context][keys + [metric]].rename(
            columns={metric: "context_macro_f1"}
        )
        merged = baseline.merge(current, on=keys, how="inner")
        if merged.empty:
            continue
        merged["context"] = context
        merged["delta_macro_f1"] = merged["context_macro_f1"] - merged["baseline_macro_f1"]
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    table["context_label"] = table["context"].map(lambda value: CONTEXT_LABELS.get(value, value))
    table["model_label"] = table["model"].map(display_model)
    return save_csv(table.sort_values(["context", "model", "body_part_count"]),
                    "context_gain_at_equal_body_parts.csv")


def create_activity_table():
    """Read activity-level summary files when available.

    Activity tables are optional in the result export. This function searches
    all experiment directories for ``activity_performance.csv`` and combines
    them without failing if a given experiment has no such file.
    """
    rows = []
    for context in CONTEXTS:
        context_dir = RESULTS_DIR / context
        candidates = [context_dir / "activity_performance.csv"]
        candidates += list(context_dir.glob("*/activity_performance.csv"))
        for path in candidates:
            try:
                table = pd.read_csv(path)
            except Exception:
                continue
            experiment = "standard" if path.parent == context_dir else path.parent.name
            metadata = parse_experiment_name(experiment, context)
            table["context"] = context
            table["experiment"] = experiment
            for key in ("body_part_count", "body_part_mode", "body_part"):
                table[key] = metadata[key]
            rows.append(table)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    return save_csv(combined, "activity_performance_all_experiments.csv")


def create_model_summary(results):
    metric = select_metric_column(results)
    table = (results.groupby(["context", "model"], as_index=False)[metric]
             .agg(mean="mean", std="std", best="max"))
    table = table.rename(columns={metric: "macro_f1"})
    table["context_label"] = table["context"].map(lambda value: CONTEXT_LABELS.get(value, value))
    table["model_label"] = table["model"].map(display_model)
    return save_csv(table.sort_values("mean", ascending=False), "model_summary.csv")

def plot_body_part_curve(results):
    metric = select_metric_column(results)
    table = results[results["body_part_mode"].isin(["progressive", "all_7"])].copy()
    if table.empty:
        return
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 4.8), sharey=True)
    if len(MODELS) == 1:
        axes = [axes]
    for ax, model in zip(axes, MODELS):
        subset = table[table["model"] == model]
        for context in CONTEXTS:
            curve = subset[subset["context"] == context].sort_values("body_part_count")
            if curve.empty:
                continue
            ax.plot(curve["body_part_count"], curve[metric], marker="o",
                    linewidth=2, label=CONTEXT_LABELS[context])
        ax.set_title(display_model(model))
        ax.set_xlabel("Number of inertial body parts")
        ax.set_xticks(range(1, 8))
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean macro F1")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Effect of the number of body parts and context")
    save_figure("01_body_part_count_vs_context.png")


def plot_context_gain(results):
    metric = select_metric_column(results)

    keys = ["body_part_count", "body_part_mode", "body_part", "model"]

    # Baseline
    baseline = (
        results[results["context"] == "baseline"][keys + [metric]]
        .rename(columns={metric: "baseline_f1"})
    )

    fig, axes = plt.subplots(
        1, len(MODELS),
        figsize=(15, 4.8),
        sharey=True
    )

    if len(MODELS) == 1:
        axes = [axes]

    plotted = False

    for ax, model in zip(axes, MODELS):

        model_base = baseline[baseline["model"] == model]

        for context in ["gps", "light", "mic", "all_context"]:

            current = (
                results[
                    (results["context"] == context) &
                    (results["model"] == model)
                ][keys + [metric]]
                .rename(columns={metric: "context_f1"})
            )

            # Match baseline and context for the SAME body-part configuration
            merged = model_base.merge(
                current,
                on=keys,
                how="inner"
            )

            if merged.empty:
                continue

            # ---------------------------------------------------------
            # 1. Compute the marginal contribution for each configuration
            # ---------------------------------------------------------
            merged["delta"] = (
                merged["context_f1"] -
                merged["baseline_f1"]
            )

            # ---------------------------------------------------------
            # 2. Aggregate configurations having the same number
            #    of inertial body parts
            # ---------------------------------------------------------
            aggregated = (
                merged
                .groupby("body_part_count", as_index=False)["delta"]
                .mean()
                .sort_values("body_part_count")
            )

            # ---------------------------------------------------------
            # 3. Plot one point per number of body parts
            # ---------------------------------------------------------
            ax.plot(
                aggregated["body_part_count"],
                aggregated["delta"],
                marker="o",
                linewidth=2,
                label=CONTEXT_LABELS[context]
            )

            plotted = True

        # Zero reference line
        ax.axhline(0, color="black", linewidth=1)

        ax.set_title(display_model(model))
        ax.set_xlabel("Number of inertial body parts")
        ax.set_xticks(range(1, 8))
        ax.grid(alpha=0.25)

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_ylabel("Δ macro F1 vs baseline")

    handles, labels = axes[-1].get_legend_handles_labels()

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            bbox_to_anchor=(0.5, -0.04)
        )

    fig.suptitle(
        "Marginal value of context at equal inertial information"
    )

    save_figure("02_context_gain_at_equal_body_parts.png")

def plot_single_body_part_ranking(results):
    metric = select_metric_column(results)
    table = results[(results["body_part_mode"] == "single") &
                    (results["context"] == "baseline")].copy()
    if table.empty:
        return
    pivot = table.pivot(index="body_part", columns="model", values=metric)
    pivot = pivot.reindex(columns=[m for m in MODELS if m in pivot.columns])
    ax = pivot.sort_values(list(pivot.columns)[-1]).plot(kind="barh", figsize=(10, 6))
    ax.set_xlabel("Mean macro F1")
    ax.set_ylabel("Single body part")
    ax.set_title("Single-body-part baseline ranking")
    ax.grid(axis="x", alpha=0.25)
    ax.legend([display_model(c) for c in pivot.columns], title="Model")
    save_figure("03_single_body_part_ranking.png")


def plot_global_configuration_comparison(results):
    metric = select_metric_column(results)
    table = results[results["body_part_mode"] == "all_7"].copy()
    if table.empty:
        return
    pivot = table.pivot(index="context", columns="model", values=metric).reindex(CONTEXTS)
    pivot = pivot.rename(index=CONTEXT_LABELS)
    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_ylabel("Mean macro F1")
    ax.set_xlabel("")
    ax.set_title("Seven-body-part reference: context comparison")
    ax.set_xticklabels([CONTEXT_LABELS[c] for c in CONTEXTS], rotation=20, ha="right")
    ax.legend([display_model(c) for c in pivot.columns], title="Model")
    ax.grid(axis="y", alpha=0.25)
    save_figure("04_seven_body_parts_context_comparison.png")


def plot_model_ranking(results):
    metric = select_metric_column(results)
    table = results[results["body_part_mode"] == "all_7"].copy()
    if table.empty:
        return
    pivot = table.pivot(index="model", columns="context", values=metric)
    pivot = pivot.reindex(index=MODELS, columns=CONTEXTS)
    pivot = pivot.rename(index=MODEL_LABELS, columns=CONTEXT_LABELS)
    ax = pivot.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Mean macro F1")
    ax.set_xlabel("")
    ax.set_title("Model comparison using all seven body parts")
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.legend(title="Dataset configuration")
    ax.grid(axis="y", alpha=0.25)
    save_figure("05_model_comparison_all_body_parts.png")


def plot_activity_summary(activity_table):
    if activity_table.empty:
        print("[INFO] No activity-level summary CSVs found; skipping activity plot.")
        return
    # Accept either one row per activity or wide f1_<activity> columns.
    f1_columns = [column for column in activity_table.columns if column.startswith("f1_")]
    if not f1_columns:
        return
    table = activity_table[activity_table["body_part_mode"] == "all_7"].copy()
    if table.empty:
        return
    model = "svm" if "svm" in table["model"].unique() else table["model"].iloc[0]
    table = table[table["model"] == model]
    means = table.groupby("context")[f1_columns].mean().reindex(CONTEXTS)
    means.columns = [column.removeprefix("f1_") for column in means.columns]
    ax = means.plot(kind="bar", figsize=(11, 6))
    ax.set_ylabel("Mean F1")
    ax.set_xlabel("")
    ax.set_title(f"Activity-level comparison using {display_model(model)}")
    ax.set_xticklabels([CONTEXT_LABELS[c] for c in CONTEXTS], rotation=20, ha="right")
    ax.legend(title="Activity")
    ax.grid(axis="y", alpha=0.25)
    save_figure("06_activity_comparison_if_available.png")


# ============================================================
# SYNTHETIC SUMMARY FOR PRESENTATION
# ============================================================

def create_presentation_summary(results, context_gain):
    metric = select_metric_column(results)
    reference = results[results["body_part_mode"] == "all_7"].copy()
    lines = [
        "Exploratory Data Analysis summary",
        "=================================",
        "",
        "The primary metric is mean macro F1, which gives equal weight to all activities.",
        "The seven-body-part experiments are the reference; reduced-body-part experiments",
        "are compared at equal body-part counts whenever the same experiment exists.",
        "",
    ]
    if not reference.empty:
        best = reference.loc[reference[metric].idxmax()]
        lines += [
            f"Best seven-body-part result: {display_model(best['model'])} / "
            f"{CONTEXT_LABELS.get(best['context'], best['context'])} "
            f"(macro F1={best[metric]:.3f}).",
        ]
        for model in MODELS:
            model_ref = reference[reference["model"] == model]
            if model_ref.empty:
                continue
            baseline = model_ref[model_ref["context"] == "baseline"][metric]
            context = model_ref[model_ref["context"] == "all_context"][metric]
            if not baseline.empty and not context.empty:
                lines.append(
                    f"{display_model(model)}: all context minus baseline = "
                    f"{context.iloc[0] - baseline.iloc[0]:+.3f} macro F1."
                )
    if not context_gain.empty:
        lines += ["", "Context gains at equal body-part counts:"]
        grouped = context_gain.groupby("context")["delta_macro_f1"].agg(["mean", "median"])
        for context, row in grouped.iterrows():
            lines.append(
                f"  {CONTEXT_LABELS.get(context, context)}: "
                f"mean {row['mean']:+.3f}, median {row['median']:+.3f}."
            )
    save_summary("\n".join(lines))


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("FOCUSED HAR EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    results, details = load_eda_results()
    print(f"Loaded {len(results)} experiment/model summaries.")
    if not details.empty:
        print(f"Loaded {len(details)} fold-level metric rows.")
    """
    create_global_ranking(results)
    create_model_summary(results)
    create_body_part_single_ranking(results)
    create_body_part_curve(results)
    context_gain = create_context_gain_table(results)
    activity_table = create_activity_table() """

    #plot_body_part_curve(results)
    plot_context_gain(results)
    """
    plot_single_body_part_ranking(results)
    plot_global_configuration_comparison(results)
    plot_model_ranking(results)
    plot_activity_summary(activity_table)

    create_presentation_summary(results, context_gain)
    """

    print("\n" + "=" * 70)
    print("EDA COMPLETED")
    print(f"Tables:  {TABLES_DIR}")
    print(f"Plots:   {PLOTS_DIR}")
    print(f"Summary: {SUMMARY_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
