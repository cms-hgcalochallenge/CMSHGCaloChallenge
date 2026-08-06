#!/usr/bin/env python3
"""Pareto plots: shower quality vs generation time, parameters, or FLOPs per model."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import mplhep as hep
from pathlib import Path

plt.style.use(hep.style.CMS)
mpl.rcParams.update({
    "axes.labelpad": 5,
    "legend.frameon": False,
    "legend.handletextpad": 0.8,
    "yaxis.labellocation": "center",
})

MODEL_COLORS = {
    "HGCaloDiffusion": "#5790fc",
    "HGCaloDream":     "#f89c20",
    "HGCaloTrilogy":   "#e42536",
    "GraphCNF":        "#964a8b",
    "AllShowers":      "#9c9ca1",
    "CaloDiT-2":       "#7a21dd",
}


def _parse_name(raw: str) -> tuple[str, str]:
    """Return (model_display_name, particle) from a timing entry name."""
    particle = "Photon" if "photon" in raw.lower() else "Pion"
    model = raw.replace("Photon", "").replace("Pion", "")
    return model, particle


def collect_timing(data: list, batch_size: int, remove_first_batch: bool = False) -> dict:
    """Return {(model, particle, energy): (x_mean, x_err)} for a given batch size."""
    out = {}
    for pt in data:
        if pt["batch"] != batch_size:
            continue
        if remove_first_batch:
            times = np.array([v for k, v in pt["timing"].items()
                              if "sample_batch" in k and k != "sample_batch_0"])
        else:
            times = np.array([v for k, v in pt["timing"].items() if "sample_batch" in k])

        model, particle = _parse_name(pt["name"])
        key = (model, particle, pt["energy"])
        # Each sample_batch_N call produces `batch_size` showers, not
        # pt["samples"] (the trial total across all calls) — divide by the
        # former to get a genuine per-shower time.
        out[key] = (float(np.mean(times / batch_size)),
                    float(np.std(times  / batch_size)))
    return out


def collect_model_stats(stat_data: list, particle: str, energy: int) -> dict:
    """Return {model: (params, flops_mean, flops_xerr_lo, flops_xerr_hi)}.

    Energy-dependent entries (AllShowers) are filtered to the requested energy.
    Models without an energy field match any energy.
    Variant entries like 'HGCaloTrilogy(N step)' are skipped.
    """
    result = {}
    for entry in stat_data:
        name = entry.get("model", "")
        if "(" in name:
            continue
        if entry.get("particle") != particle:
            continue
        if "energy" in entry and entry["energy"] != energy:
            continue
        flops = entry["flops"]
        mean  = float(np.mean(flops))
        lo    = mean - float(min(flops))
        hi    = float(max(flops)) - mean
        result[name] = (entry["parameters"], mean, lo, hi)
    return result


def _extend_axes_for_legend(ax, fig, xs, ys):
    """Extend the x-axis (log scale) rightward so the upper-right legend
    does not overlap any plotted (x, y) points."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend = ax.get_legend()
    if legend is None:
        return

    leg_disp = legend.get_window_extent(renderer)
    ax_disp  = ax.get_window_extent(renderer)
    leg_x0_frac = (leg_disp.x0 - ax_disp.x0) / ax_disp.width
    leg_y0_frac = (leg_disp.y0 - ax_disp.y0) / ax_disp.height

    log_xmin, log_xmax = np.log10(ax.get_xlim()[0]), np.log10(ax.get_xlim()[1])
    ymin, ymax = ax.get_ylim()

    def frac_to_logx(f):
        return 10 ** (log_xmin + f * (log_xmax - log_xmin))

    if ax.get_yscale() == "log":
        log_ymin, log_ymax = np.log10(ymin), np.log10(ymax)
        def frac_to_y(f):
            return 10 ** (log_ymin + f * (log_ymax - log_ymin))
    else:
        def frac_to_y(f):
            return ymin + f * (ymax - ymin)

    leg_x0_data = frac_to_logx(leg_x0_frac)
    leg_y0_data = frac_to_y(leg_y0_frac)

    conflict_xs = [x for x, y in zip(xs, ys) if x >= leg_x0_data and y >= leg_y0_data]
    if not conflict_xs:
        return

    max_cx = max(conflict_xs)
    target_frac = leg_x0_frac * 0.95
    new_log_range = (np.log10(max_cx) - log_xmin) / target_frac
    ax.set_xlim(10 ** log_xmin, 10 ** (log_xmin + new_log_range))


def plot_pareto(x_vals: dict, x_errs: dict, x_label: str,
                metrics: dict, particle: str, energy: int,
                metric: str, out_path: Path) -> None:
    """
    Scatter plot of shower quality vs an x-axis quantity for all models.

    Args:
        x_vals: {model: x_value}
        x_errs: {model: (xerr_lo, xerr_hi)} or None for no error bars
        x_label: x-axis label string
        metrics: {display_name -> {dataset -> {metric -> value}}}
        metric:  "AUC" | "FPD" | "Sep_all"
        out_path: output file path
    """
    dataset_key = f"{particle}_E{energy}"

    fig, ax = plt.subplots(figsize=(10, 8))

    plotted = []
    xs, ys = [], []
    for model in MODEL_COLORS:
        if model not in x_vals:
            continue
        model_metrics = metrics.get(model, {}).get(dataset_key, {})

        if metric == "AUC":
            y_val = model_metrics.get("AUC")
            if y_val is None:
                continue
            y_err = None
        elif metric == "FPD":
            y_val = model_metrics.get("FPD")
            y_err = model_metrics.get("FPD_err")
            if y_val is None:
                continue
        else:  # Sep_all
            raw = model_metrics.get("Sep_all")
            if raw is None:
                continue
            y_val = raw * 1e3
            y_err = None

        x_val = x_vals[model]
        xe = None
        if x_errs and model in x_errs:
            lo, hi = x_errs[model]
            if lo > 0 or hi > 0:
                xe = [[lo], [hi]]

        color = MODEL_COLORS[model]
        ax.errorbar(
            x_val, y_val,
            xerr=xe,
            yerr=[[y_err], [y_err]] if y_err is not None else None,
            color=color,
            marker="o", alpha=0.8,
            ms=1, elinewidth=1, capsize=3,
        )
        ax.scatter(
            x_val, y_val,
            color=color, marker="o",
            alpha=0.9, s=100,
            edgecolors="black", linewidths=0.5,
        )
        plotted.append(model)
        xs.append(x_val)
        ys.append(y_val)

    if not plotted:
        plt.close()
        print(f"No data for {particle} {energy} GeV {metric} [{x_label}], skipping.")
        return

    hep.cms.label("Preliminary", ax=ax, data=False,
                  rlabel=f"{particle} {energy} GeV", loc=0)

    ax.set_xscale("log")
    ax.set_xlabel(x_label, fontsize=24)

    if metric == "AUC":
        ax.set_ylabel("AUC", fontsize=24)
    elif metric == "FPD":
        ax.set_yscale("log")
        ax.set_ylabel(r"FPD ($\times 10^{-3}$)", fontsize=24)
    else:  # Sep_all
        ax.set_yscale("log")
        ax.set_ylabel(r"Avg. Sep. Power ($\times 10^{-3}$)", fontsize=24)

    legend_handles = [
        mpatches.Patch(color=color, label=label, ec="black")
        for label, color in MODEL_COLORS.items()
        if label in plotted
    ]
    ax.legend(handles=legend_handles, fontsize=16, loc="upper right")

    plt.tight_layout()
    _extend_axes_for_legend(ax, fig, xs, ys)

    ax.annotate(
        "Better",
        xy=(0.08, 0.07), xycoords="axes fraction",
        xytext=(0.22, 0.19), textcoords="axes fraction",
        fontsize=17, fontweight="bold", color="dimgray",
        ha="center", va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color="dimgray",
            lw=3,
            mutation_scale=28,
        ),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing_input",   default="timing_inputs/combined_timing_profiles.json")
    parser.add_argument("--stats_input",    default="timing_inputs/model_parameters_flops.json")
    parser.add_argument("--metrics_input",  default="metrics_summary.json")
    parser.add_argument("--output_dir",     default="profiling_plots")
    parser.add_argument("--remove_first_batch", action="store_true",
                        help="Exclude first (warmup) batch from timing averages")
    args = parser.parse_args()

    timing_raw = json.loads(Path(args.timing_input).read_text())
    stat_raw   = json.loads(Path(args.stats_input).read_text())
    metrics    = json.loads(Path(args.metrics_input).read_text())
    output_dir = Path(args.output_dir)

    particles = ["Photon", "Pion"]
    energies  = [5, 50, 500]
    quality_metrics = ["AUC", "FPD", "Sep_all"]

    # ── Timing x-axis (batch 1 and batch 100) ──────────────────────────────
    for batch_size in [1, 100]:
        timing_map = collect_timing(timing_raw, batch_size,
                                    remove_first_batch=args.remove_first_batch)
        for particle in particles:
            for energy in energies:
                x_vals = {model: t for (model, p, e), (t, _) in timing_map.items()
                          if p == particle and e == energy}
                x_errs = {model: (0, s) for (model, p, e), (_, s) in timing_map.items()
                          if p == particle and e == energy}
                x_label = f"Time/Shower (s) [batch size = {batch_size}]"

                for metric in quality_metrics:
                    tag = f"{metric.lower()}_{particle.lower()}_{energy}GeV_batch{batch_size}"
                    plot_pareto(x_vals, x_errs, x_label, metrics,
                                particle, energy, metric,
                                output_dir / f"pareto_{tag}.pdf")

    # ── Parameters x-axis ──────────────────────────────────────────────────
    for particle in particles:
        for energy in energies:
            stats = collect_model_stats(stat_raw, particle, energy)
            x_vals = {m: v[0] for m, v in stats.items()}
            x_errs = None  # parameters have no range

            for metric in quality_metrics:
                tag = f"{metric.lower()}_{particle.lower()}_{energy}GeV_params"
                plot_pareto(x_vals, x_errs, "Model Parameters", metrics,
                            particle, energy, metric,
                            output_dir / f"pareto_{tag}.pdf")

    # ── FLOPs x-axis ───────────────────────────────────────────────────────
    for particle in particles:
        for energy in energies:
            stats = collect_model_stats(stat_raw, particle, energy)
            x_vals = {m: v[1] for m, v in stats.items()}
            x_errs = {m: (v[2], v[3]) for m, v in stats.items()}

            for metric in quality_metrics:
                tag = f"{metric.lower()}_{particle.lower()}_{energy}GeV_flops"
                plot_pareto(x_vals, x_errs, "Model FLOPs", metrics,
                            particle, energy, metric,
                            output_dir / f"pareto_{tag}.pdf")
