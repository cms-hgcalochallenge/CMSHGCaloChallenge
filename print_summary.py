#!/usr/bin/env python3
"""Print a summary table of all completed HGCal evaluations."""

import json
import re
import statistics
from pathlib import Path

REPO        = Path(__file__).parent
RESULTS_ALL = REPO / "eval_results_all"

# Maps display name -> directory name(s) on disk, tried in order; first one
# with a metrics.txt for the requested dataset wins. This lets a model's
# directory be upgraded (e.g. "_v2") without losing datasets that haven't
# been re-run under the new directory yet.
MODELS = {
    "HGCaloDiffusion": ["HGCaloDiffusion"],
    "HGCaloDream":     ["HGCaloDream"],
    "HGCaloTrilogy":   ["HGCaloTrilogy"],
    "GraphCNF":        ["GraphCNF"],
    "AllShowers":      ["AllShowers"],
    "CaloDiT-2":       ["CaloDiT-2"],
    "Geant4":          ["Geant4"],
}
# Reference models are shown in tables but excluded from best-value bolding
REFERENCE_MODELS = {"Geant4"}
DATASETS = ["Photon_E5", "Photon_E50", "Photon_E500", "Photon_LogUniform",
            "Pion_E5",   "Pion_E50",   "Pion_E500",   "Pion_LogUniform"]


def resolve_dirname(dirnames: list, dataset: str) -> str:
    """Return the first dirname among dirnames with a metrics.txt for this dataset."""
    for dirname in dirnames:
        if (RESULTS_ALL / dirname / dataset / "metrics.txt").exists():
            return dirname
    return dirnames[0]


def metrics_path(dirnames: list, dataset: str) -> Path:
    """Return the first existing metrics.txt among dirnames for this dataset."""
    return RESULTS_ALL / resolve_dirname(dirnames, dataset) / dataset / "metrics.txt"


def collect_seed_metrics(dirnames: list, dataset: str) -> list:
    """Parse metrics.txt from each classifier-reseed run under cls_seeds/seed_*/
    (see condor/submit_cls_seed_grid.sub), used to estimate AUC/OOD-fraction
    run-to-run variation from the classifier's random initialization."""
    seed_dir = RESULTS_ALL / resolve_dirname(dirnames, dataset) / dataset / "cls_seeds"
    samples = []
    if seed_dir.is_dir():
        for seed_path in sorted(seed_dir.glob("seed_*/metrics.txt")):
            sm = parse_metrics(seed_path)
            if sm:
                samples.append(sm)
    return samples


def add_seed_error_bars(m: dict, dirnames: list, dataset: str) -> None:
    """Pool the base (seed=123) run with any classifier-reseed runs and attach
    <metric>_err (sample std-dev across seeds) / <metric>_n (pool size) to m."""
    samples = [m] + collect_seed_metrics(dirnames, dataset)
    for key in ("AUC", "OOD_frac"):
        vals = [s[key] for s in samples if key in s]
        if len(vals) >= 2:
            m[f"{key}_err"] = statistics.stdev(vals)
            m[f"{key}_n"] = len(vals)


def parse_metrics(path: Path) -> dict:
    """Parse a metrics.txt file, returning a dict of metric name -> value."""
    result = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        # "Avg separation power / KS of <Category> features: X / Y"
        m = re.match(r"Avg separation power / KS of (\w+) features:\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)", line)
        if m:
            cat, sep, ks = m.group(1), float(m.group(2)), float(m.group(3))
            result[f"KS_{cat}"] = ks
            result[f"Sep_{cat}"] = sep
            continue
        # "Num generated showers: N"
        m = re.match(r"Num generated showers:\s*(\d+)", line)
        if m:
            result["n_showers"] = int(m.group(1))
            continue
        # "Result of classifier test (AUC): X"
        m = re.match(r"Result of classifier test \(AUC\):\s*([\d.e+\-]+)", line)
        if m:
            result["AUC"] = float(m.group(1))
            continue
        # "FPD: X ± Y" or "FPD [NLL-trim 1%]: X ± Y"
        m = re.match(r"FPD(?:\s*\[[^\]]*\])?:\s*([\d.e+\-]+)\s*±\s*([\d.e+\-]+)", line)
        if m:
            result["FPD"] = float(m.group(1))
            result["FPD_err"] = float(m.group(2))
            continue
        # "KPD: X ± Y"
        m = re.match(r"KPD:\s*([\d.e+\-]+)\s*±\s*([\d.e+\-]+)", line)
        if m:
            result["KPD"] = float(m.group(1))
            result["KPD_err"] = float(m.group(2))
            continue
        # "OOD fraction (combined / gen / geant): X / Y / Z"
        m = re.match(r"OOD fraction \(combined / gen / geant\):\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)\s*/\s*([\d.e+\-]+)", line)
        if m:
            result["OOD_frac"] = float(m.group(1))
            result["OOD_frac_gen"] = float(m.group(2))
            result["OOD_frac_geant"] = float(m.group(3))
            continue
        # "OOD test set size (combined / gen / geant): N / N / N"
        m = re.match(r"OOD test set size \(combined / gen / geant\):\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", line)
        if m:
            result["OOD_n"] = int(m.group(1))
            result["OOD_n_gen"] = int(m.group(2))
            result["OOD_n_geant"] = int(m.group(3))
            continue
        # "Reweighting efficiency (n_eff/n, gen->geant): X" (X may be "nan"
        # when the OOD-fraction guard kicks in, i.e. near-total separation)
        m = re.match(r"Reweighting efficiency \(n_eff/n, gen->geant\):\s*(nan|[\d.e+\-]+)", line)
        if m and m.group(1) != "nan":
            result["ReweightEff"] = float(m.group(1))
    return result


def _bf(s):
    return f"\\textbf{{{s}}}"


def _fmt_sep(v):
    """Format sep/KS value in units of 1e-3, 2 decimal places."""
    return f"{v * 1e3:.2f}"


def _fmt_ood(frac, n):
    """Format an OOD fraction as a plain 0-1 number, e.g. "0.22".

    If zero OOD events were actually observed, a classifier test set of
    finite size n has no power to distinguish "truly zero" from some small
    nonzero rate -- a single outlier could exist and still not show up.
    Report a flat "<1e-4" sensitivity floor instead of asserting 0 (test
    sets here run ~1e4-2e5 events, so 1e-4 is the right order of magnitude;
    exact precision isn't the point of this note).
    """
    if frac is None:
        return "--"
    if n and round(frac * n) == 0:
        return "<1e-4"
    return f"{frac:.2f}"


def _fmt_fpd(v, e):
    if v >= 100:
        return f"{v:.0f} $\\pm$ {e:.0f}"
    return f"{v:.1f} $\\pm$ {e:.1f}"


def _fmt_kpd(v, e):
    return f"{v:.3f} $\\pm$ {e:.3f}"


def print_latex_tables(rows):
    """Print one LaTeX table per dataset, with Sep/KS, AUC, FPD, KPD columns."""
    lat_cats = ["Energy", "Transverse", "Center", "Width", "Occupancy"]
    lat_cat_labels = {"Energy": "Longitudinal"}  # display label override for the latex header
    ncats = len(lat_cats)

    # Group by dataset
    by_dataset = {}
    for model, dataset, m in rows:
        by_dataset.setdefault(dataset, {})[model] = m

    print("%" + "=" * 70)
    print("% LaTeX Tables")
    print("%" + "=" * 70)

    for dataset in DATASETS:
        if dataset not in by_dataset:
            continue
        model_data = by_dataset[dataset]

        # Find best values for bolding — exclude reference models from competition
        competing = {mdl: m for mdl, m in model_data.items() if mdl not in REFERENCE_MODELS}
        best_sep, best_ks = {}, {}
        for cat in lat_cats:
            seps = [m[f"Sep_{cat}"] for m in competing.values() if m.get(f"Sep_{cat}") is not None]
            kss  = [m[f"KS_{cat}"]  for m in competing.values() if m.get(f"KS_{cat}")  is not None]
            best_sep[cat] = min(seps) if seps else None
            best_ks[cat]  = min(kss)  if kss  else None

        auc_vals = {mdl: m["AUC"] for mdl, m in competing.items() if m.get("AUC") is not None}
        best_auc_dist = min(abs(v - 0.5) for v in auc_vals.values()) if auc_vals else None

        fpds = [m["FPD"] for m in competing.values() if m.get("FPD") is not None]
        best_fpd = min(fpds) if fpds else None

        kpds = [m["KPD"] for m in competing.values() if m.get("KPD") is not None]
        best_kpd = min(kpds) if kpds else None

        col_spec = "l" + "c" * ncats + "ccc"
        print(f"\n% Dataset: {dataset}")
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        cat_heads = " & ".join(lat_cat_labels.get(cat, cat) for cat in lat_cats)
        print(f"   Model & \\multicolumn{{{ncats}}}{{c}}{{Sep. Power / KS $(\\times10^{{-3}})$}} & AUC (OOD) & FPD & KPD \\\\")
        print(f"         & {cat_heads} & & $(\\times10^{{-3}})$ & $(\\times10^{{-3}})$ \\\\")
        print("   \\hline")

        ordered_models = [m for m in MODELS if m in REFERENCE_MODELS] + \
                         [m for m in MODELS if m not in REFERENCE_MODELS]
        printed_ref = False
        for model in ordered_models:
            if model not in model_data:
                continue
            if model in REFERENCE_MODELS:
                pass  # reference models come first
            elif not printed_ref:
                print("   \\hline")
                printed_ref = True
            m = model_data[model]
            cells = [f"   {model}"]

            for cat in lat_cats:
                sep = m.get(f"Sep_{cat}")
                ks  = m.get(f"KS_{cat}")
                if sep is None and ks is None:
                    cells.append("--")
                    continue
                sep_s = _fmt_sep(sep) if sep is not None else "--"
                ks_s  = _fmt_sep(ks)  if ks  is not None else "--"
                is_ref = model in REFERENCE_MODELS
                if not is_ref and sep is not None and best_sep[cat] is not None and sep <= best_sep[cat] * (1 + 1e-6):
                    sep_s = _bf(sep_s)
                if not is_ref and ks is not None and best_ks[cat] is not None and ks <= best_ks[cat] * (1 + 1e-6):
                    ks_s = _bf(ks_s)
                cells.append(f"{sep_s} / {ks_s}")

            is_ref = model in REFERENCE_MODELS

            # AUC (with OOD fraction sharing the same cell)
            auc = m.get("AUC")
            auc_err = m.get("AUC_err")
            ood = m.get("OOD_frac")
            ood_err = m.get("OOD_frac_err")
            if auc is not None:
                auc_s = f"{auc:.3f}"
                if auc_err is not None:
                    auc_s += f" $\\pm$ {auc_err:.3f}"
                if not is_ref and best_auc_dist is not None and abs(abs(auc - 0.5) - best_auc_dist) < 1e-6:
                    auc_s = _bf(auc_s)
                if ood is not None:
                    ood_n = m.get("OOD_n")
                    if ood_n and round(ood * ood_n) == 0:
                        ood_s = "$<10^{-4}$"
                    else:
                        ood_s = f"{ood * 100:.0f}\\%"
                        if ood_err is not None:
                            ood_s += f" $\\pm$ {ood_err * 100:.0f}\\%"
                    auc_s += f" ({ood_s})"
            else:
                auc_s = "--"
            cells.append(auc_s)

            # FPD
            fpd, fpd_e = m.get("FPD"), m.get("FPD_err")
            if fpd is not None:
                fpd_s = _fmt_fpd(fpd, fpd_e)
                if not is_ref and best_fpd is not None and fpd <= best_fpd * (1 + 1e-6):
                    fpd_s = _bf(fpd_s)
            else:
                fpd_s = "--"
            cells.append(fpd_s)

            # KPD
            kpd, kpd_e = m.get("KPD"), m.get("KPD_err")
            if kpd is not None:
                kpd_s = _fmt_kpd(kpd, kpd_e)
                if not is_ref and best_kpd is not None and kpd <= best_kpd * (1 + 1e-6):
                    kpd_s = _bf(kpd_s)
            else:
                kpd_s = "--"
            cells.append(kpd_s)

            print(" & ".join(cells) + " \\\\")

        print("\\end{tabular}")


def write_json(rows, out_path: Path):
    """Write {display_name: {dataset: {metric: value}}} to a JSON file."""
    data = {}
    for display_name, dataset, m in rows:
        data.setdefault(display_name, {})[dataset] = m
    out_path.write_text(json.dumps(data, indent=2))
    print(f"Wrote metrics JSON -> {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", metavar="FILE", default=None,
                        help="Write all metrics to a JSON file (default: off)")
    args = parser.parse_args()

    # Collect all results from eval_results_all only
    rows = []
    for display_name, dirnames in MODELS.items():
        for dataset in DATASETS:
            m = parse_metrics(metrics_path(dirnames, dataset))
            if m:
                add_seed_error_bars(m, dirnames, dataset)
                rows.append((display_name, dataset, m))

    if not rows:
        print("No completed evaluations found.")
        return

    if args.json:
        write_json(rows, Path(args.json))

    cats   = ["Energy", "Transverse", "Center", "Width", "Occupancy", "all"]
    col_w  = 10
    lbl_w  = 5   # width of the "Sep:" / "KS: " label
    id_w   = 16 + 1 + 22  # model + space + dataset

    # --- Table 1: Sep and KS scores for 1D feature categories ---
    total_w = id_w + lbl_w + len(cats) * col_w
    print("=" * total_w)
    print("  Separation Power / KS Scores  (lower = better)")
    print("=" * total_w)
    header = " " * (id_w + lbl_w) + "".join(f"{c:>{col_w}}" for c in cats)
    print(header)
    print("-" * total_w)

    prev_model = None
    for model, dataset, m in rows:
        sep_vals = [m.get(f"Sep_{c}") for c in cats]
        ks_vals  = [m.get(f"KS_{c}")  for c in cats]
        if not any(v is not None for v in ks_vals):
            continue
        if model != prev_model and prev_model is not None:
            print()
        prev_model = model

        prefix = f"{model:<16} {dataset:<22}"
        sep_str = "".join(f"{v:>{col_w}.4f}" if v is not None else f"{'--':>{col_w}}" for v in sep_vals)
        ks_str  = "".join(f"{v:>{col_w}.4f}" if v is not None else f"{'--':>{col_w}}" for v in ks_vals)
        print(f"{prefix}{'Sep: '}{sep_str}")
        print(f"{' ' * id_w}{'KS:  '}{ks_str}")

    # --- Table 2: Full metrics (AUC, FPD, KPD) ---
    if rows:
        print()
        print("=" * 90)
        print("  Full Metrics  (from eval_results_all)")
        print("=" * 90)
        header2 = (f"{'Model':<16} {'Dataset':<22} {'N_showers':>10} {'KS_all':>10} {'AUC':>16} "
                   f"{'FPD (1e-3)':>14} {'KPD (1e-3)':>14} {'OOD(all/gen/g4)':>28} {'ReweightEff':>12}")
        print(header2)
        print("-" * len(header2))
        prev_model = None
        for model, dataset, m in rows:
            if model != prev_model and prev_model is not None:
                print()
            prev_model = model
            n_sh  = m.get("n_showers")
            ks_all = m.get("KS_all")
            auc   = m.get("AUC")
            fpd   = m.get("FPD");  fpd_e = m.get("FPD_err")
            kpd   = m.get("KPD");  kpd_e = m.get("KPD_err")
            ood, ood_gen, ood_g4 = m.get("OOD_frac"), m.get("OOD_frac_gen"), m.get("OOD_frac_geant")
            auc_err = m.get("AUC_err")
            ood_err = m.get("OOD_frac_err")
            rw_eff = m.get("ReweightEff")

            n_sh_str = f"{n_sh:>10d}"          if n_sh  is not None else f"{'--':>10}"
            ks_str   = f"{ks_all:>10.4f}"       if ks_all is not None else f"{'--':>10}"
            if auc is not None:
                auc_str = f"{auc:>7.3f}±{auc_err:<6.3f}" if auc_err is not None else f"{auc:>7.3f}      "
                auc_str = f"{auc_str:>16}"
            else:
                auc_str = f"{'--':>16}"
            fpd_str  = f"{fpd:>7.1f}±{fpd_e:<5.1f}"     if fpd is not None else f"{'--':>14}"
            kpd_str  = f"{kpd:>7.4f}±{kpd_e:<5.4f}"     if kpd is not None else f"{'--':>14}"
            if ood is not None:
                ood_disp = _fmt_ood(ood, m.get("OOD_n"))
                if ood_err is not None and not ood_disp.startswith("<"):
                    ood_disp += f"±{ood_err:.2f}"
                ood_str = f"{ood_disp} | {_fmt_ood(ood_gen, m.get('OOD_n_gen'))} | {_fmt_ood(ood_g4, m.get('OOD_n_geant'))}"
            else:
                ood_str = "--"
            rw_str   = f"{rw_eff:.4f}"          if rw_eff is not None else "--"
            print(f"{model:<16} {dataset:<22}{n_sh_str}{ks_str}{auc_str} {fpd_str} {kpd_str} {ood_str:>28} {rw_str:>12}")

    # --- LaTeX tables (one per dataset) ---
    print()
    print_latex_tables(rows)

    # --- Completion status ---
    print()
    print("=" * 60)
    print("  Completion Status  (eval_results_all)")
    print("=" * 60)
    total    = (len(MODELS) - len(REFERENCE_MODELS)) * len(DATASETS)
    done_full    = 0
    done_partial = 0
    missing  = []
    partial  = []
    for display_name, dirnames in MODELS.items():
        if display_name in REFERENCE_MODELS:
            continue
        for dataset in DATASETS:
            m = parse_metrics(metrics_path(dirnames, dataset))
            if not m:
                missing.append(f"  {display_name}/{dataset}")
            elif any(k in m for k in ("AUC", "FPD", "KPD")):
                done_full += 1
            else:
                done_partial += 1
                partial.append(f"  {display_name}/{dataset}")

    print(f"Full (KS+AUC+FPD+KPD): {done_full}/{total}")
    print(f"Partial (KS only):      {done_partial}/{total}")
    if partial:
        print("\nPartial (missing AUC/FPD/KPD):")
        for entry in partial:
            print(entry)
    if missing:
        print("\nMissing entirely:")
        for entry in missing:
            print(entry)


if __name__ == "__main__":
    main()
