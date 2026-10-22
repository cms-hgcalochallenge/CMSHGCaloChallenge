#!/usr/bin/env python3
"""Correct the "all" separation power produced before commit a9196fb.

Before that commit, hgcal_metrics.py added every per-feature separation power
(Energy Ratio, X/Y Center, X/Y Width, Occupancy) to sep_power_sums['all'] twice,
while sep_power_counts['all'] and ks_sums['all'] were incremented once. Only the
"all" separation power is wrong; KS values and per-category averages are not.

Two sources can be corrected:

  metrics.txt (default, positional roots are searched recursively)
      If a sibling sep_power.txt exists, the "all" average is recomputed exactly
      from the per-feature values in it (the file was never affected by the bug).
      Otherwise it is rebuilt as the count-weighted mean of the category averages:

        Sep_all = [(L+1) Sep_Energy + R Sep_Transverse + 2L Sep_Center
                   + 2L Sep_Width + L Sep_Occupancy] / (6L + 1 + R)

      with L = MAX_LAYER_EVAL (or the number of layers) and R = number of rings.
      This assumes no feature was skipped for being NaN.

  metrics_summary.json (--summary_json)
      Sep_all is rebuilt with the count-weighted formula above. The buggy value
      is kept under Sep_all_buggy, and entries that already have it are skipped.

Nothing is written unless --write is given.

Example:
    python3 fix_sep_power_all.py eval_results_all eval_scaling --write
    python3 fix_sep_power_all.py --summary_json metrics_summary.json --write
"""

import argparse
import json
import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).parent
CONFIG_DIR = REPO / "configs"
RESULTS_ALL = REPO / "eval_results_all"

logger = logging.getLogger("fix_sep_power_all")

DATASET_CONFIGS: dict[str, str] = {
    "Photon_E5": "config_HGCal_photons_E5.json",
    "Photon_E50": "config_HGCal_photons_E50.json",
    "Photon_E500": "config_HGCal_photons_E500.json",
    "Photon_LogUniform": "config_HGCal_photons.json",
    "Pion_E5": "config_HGCal_pions_E5.json",
    "Pion_E50": "config_HGCal_pions_E50.json",
    "Pion_E500": "config_HGCal_pions_E500.json",
    "Pion_LogUniform": "config_HGCal_pions.json",
}

SEPARATION_CATEGORIES: tuple[str, ...] = ("Energy", "Transverse", "Center", "Width", "Occupancy")

METRICS_LINE_PATTERN = re.compile(
    r"^Avg separation power / KS of (\w+) features:\s*([\d.e+\-]+|nan)\s*/\s*([\d.e+\-]+|nan)"
)
FEATURE_LINE_PATTERN = re.compile(r"^(\d+) (.+): (\S+) / (\S+)\s*$")
PROFILE_LINE_PATTERN = re.compile(r"^(.+) (layer|ring) (\d+): (\S+) / (\S+)\s*$")
RING_HIST_PATTERN = re.compile(r"^Energy(?:fraction)?ring\d+$")

BACKUP_SUFFIX = ".sep_all_buggy"
CONSISTENCY_TOLERANCE = 1e-2


@dataclass
class CategoryCounts:
    energy: int
    transverse: int
    center: int
    width: int
    occupancy: int

    def as_dict(self) -> dict[str, int]:
        return {
            "Energy": self.energy,
            "Transverse": self.transverse,
            "Center": self.center,
            "Width": self.width,
            "Occupancy": self.occupancy,
        }

    def total(self) -> int:
        return sum(self.as_dict().values())


@dataclass
class SepPowerRecomputation:
    category_sums: dict[str, float]
    category_counts: dict[str, int]
    first_loop_sum: float

    def corrected_all(self) -> float:
        return sum(self.category_sums.values()) / sum(self.category_counts.values())

    def buggy_all(self) -> float:
        return (sum(self.category_sums.values()) + self.first_loop_sum) / sum(self.category_counts.values())


def infer_dataset_name(path: Path) -> str | None:
    """Return the dataset name from a path component such as Photon_E5 or Photon_E5_n1000."""
    for path_component in reversed(path.parts):
        for dataset_name in DATASET_CONFIGS:
            if re.fullmatch(rf"{re.escape(dataset_name)}(_n\d+)?", path_component):
                return dataset_name
    return None


def load_dataset_config(dataset_name: str, config_override: Path | None) -> dict:
    config_path = config_override if config_override is not None else CONFIG_DIR / DATASET_CONFIGS[dataset_name]
    logger.debug("Loading dataset config %s for %s", config_path, dataset_name)
    with open(config_path) as config_file:
        return json.load(config_file)


def evaluated_layer_count(dataset_config: dict) -> int:
    """Number of layers that enter the averages (MAX_LAYER_EVAL if set, else all layers)."""
    total_layer_count = int(dataset_config["SHAPE_ORIG"][1])
    max_layer_eval = dataset_config.get("MAX_LAYER_EVAL", None)
    if max_layer_eval is None:
        return total_layer_count
    return min(int(max_layer_eval), total_layer_count)


def count_rings_from_hist_dir(hist_dir: Path) -> int | None:
    if not hist_dir.is_dir():
        return None
    ring_hist_count = sum(1 for hist_path in hist_dir.glob("*.npz") if RING_HIST_PATTERN.match(hist_path.stem))
    logger.debug("Found %i ring histograms in %s", ring_hist_count, hist_dir)
    return ring_hist_count if ring_hist_count > 0 else None


def count_rings_for_dataset(model_name: str, dataset_name: str) -> int | None:
    """Ring count from this model's hist dir, falling back to any model's (the geometry is shared per dataset)."""
    ring_count = count_rings_from_hist_dir(RESULTS_ALL / model_name / dataset_name / "hist")
    if ring_count is not None:
        return ring_count

    for hist_dir in sorted(RESULTS_ALL.glob(f"*/{dataset_name}/hist")):
        ring_count = count_rings_from_hist_dir(hist_dir)
        if ring_count is not None:
            logger.debug("%s %s: using ring count %i from %s", model_name, dataset_name, ring_count, hist_dir)
            return ring_count
    return None


def expected_category_counts(evaluated_layers: int, ring_count: int) -> CategoryCounts:
    return CategoryCounts(
        energy=evaluated_layers + 1,
        transverse=ring_count,
        center=2 * evaluated_layers,
        width=2 * evaluated_layers,
        occupancy=evaluated_layers,
    )


def count_weighted_all(category_averages: dict[str, float], category_counts: CategoryCounts) -> float:
    counts_by_category = category_counts.as_dict()
    weighted_sum = sum(counts_by_category[category] * category_averages[category] for category in SEPARATION_CATEGORIES)
    return weighted_sum / category_counts.total()


def feature_category(feature_name: str) -> str | None:
    """Mirror the category assignment of the per-feature loop in hgcal_metrics.compute_metrics."""
    if "Incident" in feature_name:
        return None
    if "Energy" in feature_name:
        return "Energy"
    if "Center" in feature_name:
        return "Center"
    if "Width" in feature_name:
        return "Width"
    if "Occupancy" in feature_name:
        return "Occupancy"
    return None


def recompute_from_sep_power_file(sep_power_path: Path, max_layer_eval: int | None) -> SepPowerRecomputation:
    """Re-accumulate sep_power.txt exactly as hgcal_metrics.compute_metrics does (without the bug)."""
    category_sums: dict[str, float] = {category: 0.0 for category in SEPARATION_CATEGORIES}
    category_counts: dict[str, int] = {category: 0 for category in SEPARATION_CATEGORIES}
    first_loop_sum = 0.0

    for line in sep_power_path.read_text().splitlines():
        feature_match = FEATURE_LINE_PATTERN.match(line)
        if feature_match:
            feature_name = feature_match.group(2)
            separation_power = float(feature_match.group(3))
            ks_statistic = float(feature_match.group(4))

            category = feature_category(feature_name)
            if category is None:
                logger.debug("Skipping feature %r (not averaged)", feature_name)
                continue
            if math.isnan(separation_power) or math.isnan(ks_statistic):
                logger.debug("Skipping NaN feature %r", feature_name)
                continue
            if max_layer_eval is not None and " Layer " in feature_name:
                layer_index = int(feature_name.split(" Layer ")[-1])
                if layer_index >= max_layer_eval:
                    continue

            category_sums[category] += separation_power
            category_counts[category] += 1
            first_loop_sum += separation_power
            continue

        profile_match = PROFILE_LINE_PATTERN.match(line)
        if profile_match:
            profile_kind = profile_match.group(2)
            profile_index = int(profile_match.group(3))
            separation_power = float(profile_match.group(4))
            ks_statistic = float(profile_match.group(5))

            if math.isnan(separation_power) or math.isnan(ks_statistic):
                logger.debug("Skipping NaN profile %s %i", profile_kind, profile_index)
                continue

            if profile_kind == "layer":
                if max_layer_eval is not None and profile_index >= max_layer_eval:
                    continue
                category_sums["Energy"] += separation_power
                category_counts["Energy"] += 1
            else:
                category_sums["Transverse"] += separation_power
                category_counts["Transverse"] += 1
            continue

        if line.strip():
            logger.debug("Unparsed sep_power.txt line: %r", line)

    logger.debug("Recomputed counts from %s: %s", sep_power_path, category_counts)
    return SepPowerRecomputation(category_sums=category_sums, category_counts=category_counts, first_loop_sum=first_loop_sum)


def parse_metrics_separation_lines(metrics_path: Path) -> dict[str, tuple[float, float]]:
    """Return {category: (separation power, KS)} from metrics.txt."""
    separation_by_category: dict[str, tuple[float, float]] = {}
    for line in metrics_path.read_text().splitlines():
        line_match = METRICS_LINE_PATTERN.match(line)
        if line_match:
            separation_by_category[line_match.group(1)] = (float(line_match.group(2)), float(line_match.group(3)))
    return separation_by_category


def relative_difference(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def corrected_all_for_metrics_file(
    metrics_path: Path,
    dataset_config: dict,
    ring_count_override: int | None,
) -> float | None:
    """Return the corrected "all" separation power for one metrics.txt, or None if it cannot be determined."""
    separation_by_category = parse_metrics_separation_lines(metrics_path)
    if "all" not in separation_by_category:
        logger.debug("No separation power lines in %s, skipping", metrics_path)
        return None
    stored_all_separation = separation_by_category["all"][0]

    sep_power_path = metrics_path.parent / "sep_power.txt"
    max_layer_eval = dataset_config.get("MAX_LAYER_EVAL", None)

    if sep_power_path.exists():
        recomputation = recompute_from_sep_power_file(sep_power_path, max_layer_eval)

        for category in SEPARATION_CATEGORIES:
            if category not in separation_by_category or recomputation.category_counts[category] == 0:
                continue
            recomputed_average = recomputation.category_sums[category] / recomputation.category_counts[category]
            stored_average = separation_by_category[category][0]
            if relative_difference(recomputed_average, stored_average) > CONSISTENCY_TOLERANCE:
                logger.warning(
                    "%s: recomputed %s average %.3e disagrees with metrics.txt %.3e; sep_power.txt may be from another run",
                    metrics_path, category, recomputed_average, stored_average,
                )

        corrected_all_separation = recomputation.corrected_all()
        buggy_all_separation = recomputation.buggy_all()
        logger.debug(
            "%s: stored %.3e, recomputed buggy %.3e, recomputed corrected %.3e",
            metrics_path, stored_all_separation, buggy_all_separation, corrected_all_separation,
        )
        if relative_difference(stored_all_separation, corrected_all_separation) <= CONSISTENCY_TOLERANCE:
            logger.info("%s: already correct (%.3e)", metrics_path, stored_all_separation)
        elif relative_difference(stored_all_separation, buggy_all_separation) > CONSISTENCY_TOLERANCE:
            logger.warning(
                "%s: stored %.3e matches neither buggy (%.3e) nor corrected (%.3e) recomputation",
                metrics_path, stored_all_separation, buggy_all_separation, corrected_all_separation,
            )
        return corrected_all_separation

    if metrics_path.with_name(metrics_path.name + BACKUP_SUFFIX).exists():
        logger.info("%s: backup exists, already corrected, skipping", metrics_path)
        return None

    ring_count = ring_count_override if ring_count_override is not None else count_rings_from_hist_dir(metrics_path.parent / "hist")
    if ring_count is None:
        logger.error("%s: no sep_power.txt and ring count unknown; pass --n_rings", metrics_path)
        return None
    missing_categories = [category for category in SEPARATION_CATEGORIES if category not in separation_by_category]
    if missing_categories:
        logger.error("%s: missing categories %s, cannot apply count-weighted formula", metrics_path, missing_categories)
        return None

    category_averages = {category: separation_by_category[category][0] for category in SEPARATION_CATEGORIES}
    category_counts = expected_category_counts(evaluated_layer_count(dataset_config), ring_count)
    logger.debug("%s: no sep_power.txt, using counts %s (assumes no NaN skips)", metrics_path, category_counts.as_dict())
    return count_weighted_all(category_averages, category_counts)


def rewrite_metrics_all_line(metrics_path: Path, corrected_all_separation: float) -> None:
    backup_path = metrics_path.with_name(metrics_path.name + BACKUP_SUFFIX)
    if not backup_path.exists():
        shutil.copy2(metrics_path, backup_path)
        logger.debug("Backed up %s -> %s", metrics_path, backup_path)

    rewritten_lines: list[str] = []
    for line in metrics_path.read_text().splitlines(keepends=True):
        line_match = METRICS_LINE_PATTERN.match(line)
        if line_match and line_match.group(1) == "all":
            ks_statistic = float(line_match.group(3))
            line = "Avg separation power / KS of all features: %.3e / %.3e \n" % (corrected_all_separation, ks_statistic)
        rewritten_lines.append(line)
    metrics_path.write_text("".join(rewritten_lines))


def correct_metrics_files(
    search_roots: list[Path],
    config_override: Path | None,
    ring_count_override: int | None,
    write: bool,
) -> None:
    metrics_paths = sorted(metrics_path for search_root in search_roots for metrics_path in search_root.rglob("metrics.txt"))
    logger.info("Found %i metrics.txt files", len(metrics_paths))

    for metrics_path in metrics_paths:
        dataset_name = infer_dataset_name(metrics_path.parent)
        if dataset_name is None and config_override is None:
            logger.error("%s: cannot infer dataset from path; pass --config", metrics_path)
            continue
        dataset_config = load_dataset_config(dataset_name, config_override)

        stored_all_separation = parse_metrics_separation_lines(metrics_path).get("all", (math.nan, math.nan))[0]
        corrected_all_separation = corrected_all_for_metrics_file(metrics_path, dataset_config, ring_count_override)
        if corrected_all_separation is None:
            continue

        print("%-70s %.3e -> %.3e" % (metrics_path, stored_all_separation, corrected_all_separation))
        if write:
            rewrite_metrics_all_line(metrics_path, corrected_all_separation)


def correct_summary_json(
    summary_path: Path,
    ring_count_override: int | None,
    write: bool,
) -> None:
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)

    corrected_entry_count = 0
    for model_name, datasets in summary.items():
        for dataset_name, metrics in datasets.items():
            if "Sep_all" not in metrics:
                continue
            if "Sep_all_buggy" in metrics:
                logger.info("%s %s: already corrected, skipping", model_name, dataset_name)
                continue
            if dataset_name not in DATASET_CONFIGS:
                logger.error("%s %s: unknown dataset, skipping", model_name, dataset_name)
                continue

            ring_count = ring_count_override if ring_count_override is not None else count_rings_for_dataset(
                model_name, dataset_name
            )
            if ring_count is None:
                logger.error("%s %s: ring count unknown; pass --n_rings", model_name, dataset_name)
                continue

            dataset_config = load_dataset_config(dataset_name, None)
            category_counts = expected_category_counts(evaluated_layer_count(dataset_config), ring_count)
            category_averages = {category: metrics[f"Sep_{category}"] for category in SEPARATION_CATEGORIES}
            corrected_all_separation = count_weighted_all(category_averages, category_counts)

            feature_loop_sum = (
                category_counts.center * category_averages["Center"]
                + category_counts.width * category_averages["Width"]
                + category_counts.occupancy * category_averages["Occupancy"]
            )
            implied_energy_ratio_separation = category_counts.total() * (metrics["Sep_all"] - corrected_all_separation) - feature_loop_sum
            logger.debug(
                "%s %s: counts %s, implied Energy Ratio sep %.3e",
                model_name, dataset_name, category_counts.as_dict(), implied_energy_ratio_separation,
            )
            if implied_energy_ratio_separation < 0 or implied_energy_ratio_separation > category_counts.energy * category_averages["Energy"]:
                logger.warning(
                    "%s %s: stored Sep_all is inconsistent with the assumed counts (NaN-skipped features?); check this entry",
                    model_name, dataset_name,
                )

            print("%-16s %-18s %.4e -> %.4e" % (model_name, dataset_name, metrics["Sep_all"], corrected_all_separation))
            corrected_entry_count += 1
            if write:
                metrics["Sep_all_buggy"] = metrics["Sep_all"]
                metrics["Sep_all"] = float("%.4g" % corrected_all_separation)

    logger.info("%i entries to correct in %s", corrected_entry_count, summary_path)
    if write and corrected_entry_count > 0:
        backup_path = summary_path.with_name(summary_path.name + BACKUP_SUFFIX)
        if not backup_path.exists():
            shutil.copy2(summary_path, backup_path)
            logger.debug("Backed up %s -> %s", summary_path, backup_path)
        summary_path.write_text(json.dumps(summary, indent=2))
        logger.info("Wrote %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("search_roots", nargs="*", type=Path, help="Directories searched recursively for metrics.txt")
    parser.add_argument("--summary_json", type=Path, default=None, help="Correct Sep_all in a metrics_summary.json")
    parser.add_argument("--config", type=Path, default=None, help="Dataset config to use instead of inferring from the path")
    parser.add_argument("--n_rings", type=int, default=None, help="Number of transverse rings, if it cannot be inferred")
    parser.add_argument("--write", action="store_true", help="Apply corrections (default is a dry run)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    arguments = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if arguments.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not arguments.search_roots and arguments.summary_json is None:
        parser.error("give metrics.txt search roots and/or --summary_json")
    if not arguments.write:
        logger.info("Dry run, nothing will be written (use --write)")

    if arguments.search_roots:
        correct_metrics_files(arguments.search_roots, arguments.config, arguments.n_rings, arguments.write)
    if arguments.summary_json is not None:
        correct_summary_json(arguments.summary_json, arguments.n_rings, arguments.write)


if __name__ == "__main__":
    main()
