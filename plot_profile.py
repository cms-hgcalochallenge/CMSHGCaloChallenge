import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplhep as hep
from pathlib import Path
from matplotlib.lines import Line2D
import os
import matplotlib.patches as mpatches


# Set CMS style
plt.style.use(hep.style.CMS)
mpl.rcParams.update({
    "axes.labelpad": 5,
    "legend.frameon": False,
    "legend.handletextpad": 0.8,
    "yaxis.labellocation": "center",
})

# Model names to colors
MODEL_COLORS = {
    "HGCaloDiffusion": "#5790fc",  # blue
    "HGCaloDream": "#f89c20",  # orange
    "HGCaloDREAM": "#f89c20",  # orange
    "CaloTrilogy": "#e42536",    # red
    "HGCaloTrilogy": "#e42536",    # red
    "GraphCNF": "#964a8b",         # purple
    "AllShowers": "#9c9ca1",       # grey
    "CaloDiT-2": "#7a21dd",        # violet
}
# The exact captialization and spellings
MODEL_NAMES = [
    "HGCaloDiffusion", 
    "HGCaloDream",
    "HGCaloTrilogy",
    "GraphCNF", 
    "AllShowers", 
    "CaloDiT-2"
]
class PlotProfiling:
    def __init__(self, results_path: str = "calodif_test", throughput=False, remove_first_batch=False):
        self.throughput = throughput
        self.remove_first_batch = remove_first_batch
        with open(results_path, "r") as f:
            self.results = json.load(f)
        # CMS CVD-friendly color palette (Petroff, adopted by CMS 2024)
        # Source: https://arxiv.org/pdf/2107.02270
        self.CMS_COLORS =[ "#9c9ca1", "#964a8b", "#5790fc", "#f89c20", "#e42536", "#5790fc"]

        self.energies = [5, 50, 500]

        self.executables = set()
        self.batch_sizes = set ()

        results = []
        for trial in self.results:
            if isinstance(trial, list):
                for result in trial:
                    self.executables.add(result["name"])
                    self.batch_sizes.add(result['batch'])
                    results.append(result)
            else:
                self.executables.add(trial['name'])
                self.batch_sizes.add(trial['batch'])
                results.append(trial)

        self.results = results
        self.label_color_map = [
            (name, color) for name, color 
            in zip(["Load Model", "Load Data", "Preprocess", "Sample", "Post-process", "Write result"], self.CMS_COLORS)]

        self.model_names = []


    def build_data(self) -> dict: 
        """
        result looks like: 

        {"Photon": {
            "5GeV": {
                "model1": {
                    10: [{step0, step1, step2}],
                    100: ...,
                    1000: ...,
                }
            ...
            }

        }, 
        "Pion": {...}

        }

        Where each step is averaged across the trials
        """
        result = {}

        photon_executables = [exe for exe in self.executables if "Photon" in exe]
        pion_executables = [exe for exe in self.executables if "Pion" in exe]
        
        for part_type, executables in [("Photon", photon_executables), ("Pion", pion_executables)]:
            if not executables:
                print(f"No executables found for {part_type} skipping plot.")
                continue
            result[part_type] = {}
            for energy in self.energies: 
                result[part_type][f"{energy}GeV"] = {
                    exe.replace("Photon", "").replace("Pion", ""): {str(batch): [] for batch in self.batch_sizes} for exe in executables
                }
                for exe in executables:
                    for batch in self.batch_sizes: 
                        n_samples = 2000 if batch != 1000 else 6000
                        trials = [r for r in self.results if (r['name'] == exe) and (r['energy'] == energy) and (r['batch']==batch)]

                        averaged_results = np.zeros((6,))
                        
                        # load_data, load_model, write_result are constant-time operations (NOT divided by n_samples)
                        averaged_results[0] = 0 #np.mean([r['timing'].get('load_data', 0) for r in trials])
                        averaged_results[1] = np.mean([r['timing'].get('load_model', 0) for r in trials])
                        averaged_results[5] = np.mean([r['timing'].get('write_result', 0) for r in trials])
                        
                        # post_process: some models store a scalar, others store per-batch post_process_N keys
                        post_times = []
                        for r in trials:
                            if 'post_process' in r['timing']:
                                step = r['timing']['post_process'] / n_samples
                                if not self.throughput: 
                                    post_times.append(step)
                                else: 
                                    post_times.append(1/step)
                            else:
                                total_pp = sum(v for k, v in r['timing'].items() if k.startswith('post_process_'))/n_samples
                                if not self.throughput: 
                                    post_times.append(total_pp)
                                else: 
                                    post_times.append(1/total_pp)
                            averaged_results[4] = np.mean(post_times) if post_times else 0
                            if "graphcnf" in exe.lower():
                                #recorded incorrectly
                                averaged_results[4] = (averaged_results[4]/len([v for k, v in r['timing'].items() if k.startswith('sample_batch_')]))/10000000


                        # sample_batch_* times: sum all batch times per trial, then divide by n_samples
                        sample_times = []
                        for r in trials:
                            if self.remove_first_batch: 
                                total = sum(v for k, v in r['timing'].items() if 'sample_batch_' in k and k != 'sample_batch_0')
                            else: 
                                total = sum(v for k, v in r['timing'].items() if 'sample_batch_' in k)
                            sample_times.append(total)

                        if not self.throughput: 
                            averaged_results[3] = np.mean(sample_times)/n_samples if sample_times else 0
                        else: 
                            averaged_results[3] =  n_samples/np.mean(sample_times) if sample_times else 0
            
                        result[part_type][f"{energy}GeV"][exe.replace("Photon", "").replace("Pion", "")][str(batch)] = averaged_results
        
        return result


    def plot_timing(self, output_dir: str = "profiling_plots"):
        data = self.build_data()
        for part_type, part_experiments in data.items():
            
            calodit_index = MODEL_NAMES.index("CaloDiT-2")
            pion_models = MODEL_NAMES[:calodit_index] + MODEL_NAMES[calodit_index+1:]
            models = MODEL_NAMES if not part_type.lower() == 'pion' else pion_models

            for energy, energy_experiments in part_experiments.items(): 

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12), gridspec_kw={'height_ratios': [2, 1]}, sharex=False)
                hep.cms.label("Preliminary", ax=ax1, data=False, rlabel=f"{part_type} {energy}", loc=0)

                # Get models and batch sizes


                batch_sizes = [1, 10, 100, 1000]
                
                # Set up x-axis positions: models on x-axis, batch sizes as bars within each model group
                spacing = 0.5  # spacing between model groups
                batch_spacing = 0.2  # spacing between batch bars within a model group
                x_positions = np.arange(len(models)) * (spacing + 2 * batch_spacing)
                tick_pos, tick_label = [], []
                # Create a grouped stacked bar chart for sample time only
                # Models on x-axis, batch sizes as side-by-side bars within each model group
                for model_idx, model in enumerate(models):

                    if model.lower() == "calodit-2":
                        try:
                            model_experiment = energy_experiments["CaloDiT-2"]
                        except KeyError:
                            pass
                    else: 
                        try: 
                            model_experiment = energy_experiments[model]
                        except KeyError:
                            print(energy_experiments.keys(), model)
                            assert False

                    for batch_idx, batch_size in enumerate(batch_sizes): 
                        if str(batch_size) not in model_experiment:
                            continue
                            
                        bottom = 0
                        timing = model_experiment[str(batch_size)]
                        # Only plot sample time (index 3) and post_process (index 4)
                        # Skip load_data (0), load_model (1), pre_process (2), write_result (5)
                        x_pos = x_positions[model_idx] + (batch_idx - 1) * batch_spacing
                        tick_pos.append(x_pos)
                        tick_label.append(batch_size)
                        for timing_index in [3, 4]:  # sample and post_process only
                            step = timing[timing_index]
                            # Offset bars within each model group
                            color=self.label_color_map[timing_index][-1]
                            ax1.bar(x_pos, height=step, width=batch_spacing, bottom=bottom, color=color, ec='black')
                            bottom += step

                import matplotlib.patches as mpatches

                # Only include sample and post_process in legend
                sample_handles = [
                    mpatches.Patch(color=color, label=label, ec='black') for label, color
                    in self.label_color_map[3:5]  # sample and post_process
                ]
                ax1.legend(loc='upper right', handles=sample_handles)

                # Set x-axis labels to show models
                ax1.set_xticks(tick_pos)
                ax1.set_xticklabels(tick_label, rotation=25)

                # Calculate center positions for model labels (middle of 4 batch bars)
                model_tick_positions = []
                for i in range(len(models)):
                    # Each model has 4 batch bars (batch 1, 10, 100, 1000)
                    # Find the start and end positions for this model's group
                    start_idx = (i * 4) 
                    end_idx = start_idx + 3
                    center = (tick_pos[start_idx] + tick_pos[end_idx]) / 2
                    model_tick_positions.append(center)
                
                # Plot load_data, load_model, and write_result as stacked bars
                for model_idx, model in enumerate(models):
                    if model.lower() == "calodit-2":
                        model_experiment = energy_experiments["CaloDiT-2"]

                    else: 
                        model_experiment = energy_experiments[model]
                    # Use batch=100 for comparison (typical batch size)
                    batch_key = "100" if "100" in model_experiment else list(model_experiment.keys())[0]
                    timing = model_experiment[batch_key]
                    
                    # Stacked bars for load_data, load_model, and write_result
                    load_model = timing[1]  # load_model
                    write_result = timing[5]  # write_result
                
                    x_pos = model_tick_positions[model_idx]
                    bottom = 0
                    width = .08 * len(models)
                    x_pos = ((model_tick_positions[model_idx]-width) + (model_tick_positions[model_idx]+width))/2

                    ax2.bar(x_pos, load_model, width=width, color=self.label_color_map[1][-1], alpha=0.7, ec='black', label='Load Model' if model_idx == 0 else None, bottom=bottom)
                    bottom += load_model
                    ax2.bar(x_pos, write_result, width=width, color=self.label_color_map[5][-1], alpha=0.7, ec='black', label='Write Result' if model_idx == 0 else None, bottom=bottom)
            


                ax2.set_xticks(model_tick_positions)
                ax2.set_xticklabels(models)
                ax2.legend(loc='upper right')

                # sec.spines['bottom'].set_linewidth(0)
                ax1.set_xlabel("Batch Size", loc='right')

                if not self.throughput: 
                    ax1.set_ylabel("Time/Shower (s)", loc='top')
                else: 
                    ax1.set_ylabel("Shower/Time (1/s)", loc='top')

                
                # Subplot for constant time operations (loading + writing)
                # ax2.set_title("Constant Time Ops", fontsize=12)
                ax2.set_xlabel("Model", loc='right')
                ax1.set_xlim(-.5, len(models)-.5)
                ax2.set_xlim(-.5, len(models)-.5)
                ax2.set_ylabel("Time (s)", loc='top')

                ax1.set_yscale("log")
                plt.subplots_adjust(wspace=0, hspace=.15)
    
                # Save timing plot
                save_path = f"{output_dir}/profiling_timing_{energy}_{part_type.lower()}.pdf"
                #plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Saved: {save_path}")
                plt.close()
                
                
def plot_compare(compare, timing, out, part_type, flops=True, iter_scaled=False, throughput=False):

    # Color map for models
    energies = sorted(set(p['energy'] for p in timing))

    compare_key = "flops_mean" if flops else "parameters"
    # Plot all points for this particle

    for energy in energies:
        fig, ax = plt.subplots(figsize=(10, 8))

        xs, ys = [], []
        for time_profile in timing:
            if time_profile['energy'] != energy:
                continue

            model = time_profile["model"]

            if "AllShower" in model:
                true_name = model.split(" ")[0]
                color = MODEL_COLORS[true_name]
                display_compare = [i[compare_key] for i in compare if ("AllShower" in i['model']) and (i['energy']==energy)]
                if iter_scaled:
                    scale=[i['iterations'] for i in compare if ("AllShower" in i['model']) and (i['energy']==energy)][0]
                    display_compare = [i*scale for i in display_compare]
                display_std = [i["flops_range"] for i in compare if ("AllShower" in i['model']) and (i['energy']==energy)]


            else:
                display_compare = [i[compare_key] for i in compare if i['model']==model]
                display_std = [i["flops_range"] for i in compare if i['model']==model]
                try:
                    if display_std[0] == display_std[1]:
                        display_std = 0
                except IndexError:
                    display_std = 0

                if len(display_compare) == 0:
                    continue

                if iter_scaled:
                    scale = [i['iterations'] for i in compare if i['model']==model][0]
                    display_compare = [i*scale for i in display_compare]

            true_name = model.split(" ")[0]
            color = MODEL_COLORS[true_name]
            if "CaloTrilogy" in model:
                model = "HGCaloTrilogy"

            ax.errorbar(
                display_compare,
                time_profile['time_mean'],
                yerr=time_profile['time_std'],
                xerr=None if not flops else np.array(display_std).T,
                color=color,
                marker='o',
                alpha=0.8,
                ms=1,
                elinewidth=1,
                capsize=3,
            )
            ax.scatter(
                display_compare,
                time_profile['time_mean'],
                color=color,
                marker='o',
                alpha=0.9,
                s=100,
                label=model,
                edgecolors='black',
                linewidths=0.5,
            )
            xs.extend(display_compare)
            ys.append(time_profile['time_mean'])

        # Add CMS Preliminary and particle labels
        hep.cms.label("Preliminary", ax=ax, data=False, rlabel=f"{part_type} {energy} GeV", loc=0)

        ax.set_xscale("log")
        ax.set_yscale("log")

        # Legend
        by_label = [
                    mpatches.Patch(color=color, label=label, ec='black') for label, color
                    in  MODEL_COLORS.items() if label not in ['HGCaloDREAM', 'CaloTrilogy']
                ]
        ax.legend(handles=by_label, fontsize=16, loc="upper right")

        if not throughput:
            ax.set_ylabel("Time/Shower (s), (batch size = 10)", fontsize=24)
        else:
            ax.set_ylabel("Shower/Time (1/s), (batch size = 10)", fontsize=24)

        ax.set_xlabel("FLOPs" if flops else "#Parameters", fontsize=24, labelpad=25)

        plt.tight_layout()
        _extend_axes_for_legend(ax, fig, xs, ys)
        out_path = str(out).replace(".pdf", f"{energy}GeV.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.show()
        plt.close()


def collect_timing(data, throughput=False, remove_first_batch=False) -> list[dict]:
    timing = []
    for point in data:
        if point['batch'] != 10: 
            continue

        if not remove_first_batch: 
            point_timing = np.array([value for key, value in point['timing'].items() if "sample_batch" in key])
        else: 
            point_timing = np.array([value for key, value in point['timing'].items() if "sample_batch" in key and key != "sample_batch_0"])

        # Each sample_batch_N call produces point["batch"] showers, not
        # point["samples"] (the trial total across all calls) — divide by
        # the former to get a genuine per-shower time.
        if throughput:
            timing_mean = np.mean(point["batch"]/point_timing)
            timing_std = np.std(point['batch']/point_timing)
        else:
            timing_mean = np.mean(point_timing/point["batch"])
            timing_std = np.std(point_timing/point['batch'])

        point_timing = [value for key, value in point['timing'].items() if "sample_batch" in key]
        profile = {
            "model": point['name'].replace('Photon', "").replace("Pion", ""),
            "particle": "Photon" if "photon" in point['name'].lower() else "pion", 
            "energy": point['energy'], 
            "time_mean": timing_mean, 
            "time_std": timing_std
        }
        timing.append(profile)
    return timing

def extract_points(data: list[dict]) -> list[dict]:
    """Extract (parameters, flops_mean, flops_range) points from data."""
    points = []
    seen_models = {}  # Track (model, particle) pairs
    for entry in data:
        model = entry["model"]
        particle = entry.get("particle", "unknown")
        
        # For AllShowers, include energy in model name
        energy = entry.get("energy")
        if model == "AllShowers" and energy is not None:
            display_model = f"AllShowers ({energy} GeV)"
        else:
            display_model = model

        key = (display_model, particle)
        
        # Keep only 1-step CaloTrilogy, skip 10-step and 0-step
        if "CaloTrilogy" in display_model and display_model != "HGCaloTrilogy":
            continue
        # Skip duplicates (keep first occurrence)
        if key in seen_models:
            continue
        seen_models[key] = True

        params = entry["parameters"]
        flops = entry["flops"]
        if isinstance(flops, list):
            flops_mean = np.mean(flops)
            flops_range = (flops[0], flops[1])
        else:
            flops_mean = flops
            flops_range = (flops, flops)

        points.append({
            "model": display_model,
            "particle": particle,
            "energy": energy,
            "parameters": params,
            "flops_mean": flops_mean,
            "flops_range": flops_range,
            "iterations": entry.get("iterations", 1),
        })
    return points

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


def load_model_data(json_path: Path) -> list[dict]:
    """Load model data from JSON file."""
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def comparision_main(throughput=False, remove_first_batch=False) -> None: 
    timing_input = Path("combined_timing_profiles.json")
    profiling_input = Path("model_parameters_flops.json")
    output_dir = Path("profiling_plots")
    output_dir.mkdir(exist_ok=True)

    throughput_data = collect_timing(load_model_data(timing_input), throughput=throughput, remove_first_batch=remove_first_batch)
    profiles = extract_points(load_model_data(profiling_input))

    for particle in ["photon", "pion"]: 
        profile_points = [p for p in profiles if p["particle"].lower() == particle]
        throughput_points = [p for p in throughput_data if p['particle'].lower() == particle]
        
        plot_compare(profile_points, throughput_points, output_dir / f"flops_compare_{particle}.pdf", part_type=particle.capitalize(), throughput=throughput)
        plot_compare(profile_points, throughput_points, output_dir / f"param_compare_{particle}.pdf", part_type=particle.capitalize(), flops=False, throughput=throughput)
        plot_compare(profile_points, throughput_points, output_dir / f"flop_scaled_compare_{particle}.pdf", part_type=particle.capitalize(), flops=True, iter_scaled=True, throughput=throughput)

# Additional plots: First 2 batches vs rest throughput comparison
def collect_first_two_batches_vs_rest(data) -> list[dict]:
    """Collect timing data for first 2 batches vs rest of batches."""
    timing = []
    for point in data:
        if point['batch'] == 1:
            continue
        # Get all sample_batch times
        sample_times = {int(k.split('_')[-1]): v for k, v in point['timing'].items() if k.startswith('sample_batch_')}
        
        # First 2 batches (batch 0 and 1)
        first_two_times = [sample_times[i] for i in [0, 1] if i in sample_times]
        # Rest of batches (batch 2 onwards)
        rest_times = [sample_times[i] for i in sample_times if i >= 2]
        
        if first_two_times and rest_times:
            profile = {
                "model": point['name'].replace('Photon', "").replace("Pion", ""),
                "particle": "Photon" if "photon" in point['name'].lower() else "pion", 
                "energy": point['energy'], 
                "batch": point['batch'],
                "first_two_avg": np.mean(first_two_times) / point["samples"],
                "rest_avg": np.mean(rest_times) / point["samples"],
                "first_two_std": np.std(np.array(first_two_times) / point["samples"]),
                "rest_std": np.std(np.array(rest_times) / point["samples"]),
            }
            timing.append(profile)
    return timing

def plot_first_two_vs_rest(throughput, out, part_type):
    """Plot first 2 batches vs rest of batches throughput comparison."""
    models = sorted(set(p["model"] for p in throughput))
    energies = sorted(set(p['energy'] for p in throughput))
    markers = Line2D.filled_markers
    model_shape = dict(zip(energies, markers))

    for energy in energies: 
        fig, ax = plt.subplots(figsize=(12, 9))

        for time_profile in throughput:
            if time_profile['energy'] != energy:
                continue 

            model = time_profile["model"]
            true_name = model.split(" ")[0]
            color = MODEL_COLORS.get(true_name, "#333333")
            
            if "CaloTrilogy" in model: 
                model = "HGCaloTrilogy"

            # Plot first 2 batches vs rest
            midpoint = (time_profile['batch']*2 - time_profile['batch'])/2
            ax.errorbar(
                [time_profile['batch']-midpoint,time_profile['batch']+midpoint],
                [time_profile['first_two_avg'], time_profile['rest_avg']],
                xerr=[time_profile['first_two_std'], time_profile['rest_std']],
                color=color,
                markersize=200,
                markeredgecolor='black',
                markeredgewidth=0.5,
                alpha=0.9,
                elinewidth=1,
                capsize=3,
            )
        # Add CMS Preliminary and particle labels
        hep.cms.label("Preliminary", ax=ax, data=False, rlabel=f"{part_type} {energy} GeV", loc=0)

        ax.set_xscale("log")
        ax.set_yscale("log")

        # Legend
        by_label = [
                    mpatches.Patch(color=color, label=label, ec='black') for label, color
                    in  MODEL_COLORS.items() if label not in ['HGCaloDREAM', 'CaloTrilogy']
                ]
        ax.legend(handles=by_label, fontsize=12)

        # Grid
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        ax.set_ylabel("Throughput (1st 2 batches vs rest)", fontsize=16)
        ax.set_xlabel("Batch Size", fontsize=16, labelpad=15)

        plt.tight_layout()
        out_path = str(out).replace(".pdf", f"{energy}GeV.pdf")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.show()
        plt.close()

def first_two_vs_rest_main() -> None:
    """Main function to generate first 2 batches vs rest plots."""
    throughput_input = Path("combined_timing_profiles.json")
    output_dir = Path("profiling_plots")
    output_dir.mkdir(exist_ok=True)

    throughput_data = collect_first_two_batches_vs_rest(load_model_data(throughput_input))

    for particle in ["photon", "pion"]:
        throughput_points = [p for p in throughput_data if p['particle'].lower() == particle]

        plot_first_two_vs_rest(throughput_points, output_dir / f"first_two_vs_rest_{particle}.pdf", part_type=particle.capitalize())


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Generate HGCal computational-performance plots.")
    parser.add_argument("--timing_input",    default="timing_inputs/combined_timing_profiles.json",
                        help="Path to combined timing profiles JSON")
    parser.add_argument("--profiling_input", default="timing_inputs/model_parameters_flops.json",
                        help="Path to model parameters/FLOPs JSON")
    parser.add_argument("--output_dir",      default="profiling_plots",
                        help="Directory to write plots (default: profiling_plots/)")
    parser.add_argument("--throughput",      action="store_true",
                        help="Plot throughput (showers/s) instead of latency (s/shower)")
    parser.add_argument("--remove_first_batch", action="store_true",
                        help="Exclude the first sample batch from timing averages")
    args = parser.parse_args()

    timing_input    = Path(args.timing_input)
    profiling_input = Path(args.profiling_input)
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Timing bar charts (sample + post-process time per model / energy / batch size)
    pp = PlotProfiling(str(timing_input),
                       throughput=args.throughput,
                       remove_first_batch=args.remove_first_batch)
    pp.plot_timing(str(output_dir))

    # 2. FLOPs / parameter vs time scatter plots
    timing_data  = collect_timing(load_model_data(timing_input),
                                  throughput=args.throughput,
                                  remove_first_batch=args.remove_first_batch)
    profile_data = extract_points(load_model_data(profiling_input))
    for particle in ["photon", "pion"]:
        t_pts = [p for p in timing_data  if p["particle"].lower() == particle]
        p_pts = [p for p in profile_data if p["particle"].lower() == particle]
        plot_compare(p_pts, t_pts, output_dir / f"flops_compare_{particle}.pdf",
                     part_type=particle.capitalize(), flops=True, throughput=args.throughput)
        plot_compare(p_pts, t_pts, output_dir / f"param_compare_{particle}.pdf",
                     part_type=particle.capitalize(), flops=False, throughput=args.throughput)
        plot_compare(p_pts, t_pts, output_dir / f"flop_scaled_compare_{particle}.pdf",
                     part_type=particle.capitalize(), flops=True, iter_scaled=True, throughput=args.throughput)

    # 3. First-two-batches vs rest comparison
    ftv_data = collect_first_two_batches_vs_rest(load_model_data(timing_input))
    for particle in ["photon", "pion"]:
        pts = [p for p in ftv_data if p["particle"].lower() == particle]
        plot_first_two_vs_rest(pts, output_dir / f"first_two_vs_rest_{particle}.pdf",
                               part_type=particle.capitalize())

