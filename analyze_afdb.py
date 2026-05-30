#!/usr/bin/env python3
"""
Analyze AFDB AIUPred results and generate plots.

Replicates the analysis from aiupred.ipynb cells 22-29.

Usage:
    python analyze_afdb.py \
        --data_dir /home/jupyter-chenxi/data/afdb \
        [--output_dir /home/jupyter-chenxi/data/afdb] \
        [--version v6] \
        [--dpi 300]
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset registry matching the notebook's `names` / `pdb_dirs` lists.
# Order matters for plot layout (4 per row).
MODEL_ORGANISM_DATASETS = [
    ("UP000005640_9606_HUMAN",   "Human"),
    ("UP000000625_83333_ECOLI",  "E. coli"),
    ("UP000001940_6239_CAEEL",   "C. elegans"),
    ("UP000006548_3702_ARATH",   "A. thaliana"),
    ("UP000000559_237561_CANAL", "C. albicans"),
    ("UP000000437_7955_DANRE",   "D. rerio"),
    ("UP000002195_44689_DICDI",  "D. discoideum"),
    ("UP000000803_7227_DROME",   "D. melanogaster"),
    ("UP000008827_3847_SOYBN",   "G. max"),
    ("UP000000805_243232_METJA", "M. jannaschii"),
    ("UP000000589_10090_MOUSE",  "M. musculus"),
    ("UP000059680_39947_ORYSJ",  "O. sativa"),
    ("UP000002494_10116_RAT",    "R. norvegicus"),
    ("UP000002311_559292_YEAST", "S. cerevisiae"),
    ("UP000002485_284812_SCHPO", "S. pombe"),
    ("UP000007305_4577_MAIZE",   "Z. mays"),
    ("UP000008854_6183_SCHMA",   "*S. mansoni"),
    ("UP000001450_36329_PLAF7",  "*P. falciparum"),
    ("UP000001631_447093_AJECG", "*A. capsulatus"),
    ("UP000000806_272631_MYCLE", "*M. leprae"),
]

TSV_DATASETS = [
    ("uniprotkb_Nematocida_parisii_2025_04_25", "*N. parisii"),
]


def list_str_to_array(df):
    """Convert stored list-of-float strings back to numpy arrays."""
    df = df.copy()
    df["disorder_score"] = df["disorder_score"].apply(lambda x: np.array(eval(x)))
    df["plddt"] = df["plddt"].apply(lambda x: np.array(eval(x)))
    return df


def load_datasets(data_dir, version):
    """
    Load all result CSVs. Returns (dfs, names, pdb_dirs).
    Missing CSV files are skipped with a warning.
    """
    dfs = []
    names = []
    pdb_dirs = []

    all_datasets = MODEL_ORGANISM_DATASETS + TSV_DATASETS
    for base_name, display_name in all_datasets:
        # TSV datasets: CSV name may or may not have version suffix
        if base_name.startswith("uniprotkb_"):
            csv_candidates = [
                os.path.join(data_dir, f"{base_name}_{version}.csv"),
                os.path.join(data_dir, f"{base_name}.csv"),
            ]
        else:
            csv_candidates = [
                os.path.join(data_dir, f"{base_name}_{version}.csv"),
            ]

        loaded = False
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                try:
                    df = list_str_to_array(pd.read_csv(csv_path))
                    dfs.append(df)
                    names.append(display_name)
                    pdb_dirs.append(base_name)
                    print(f"  Loaded {display_name}: {len(df)} proteins ({os.path.basename(csv_path)})")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  WARNING: Failed to load {csv_path}: {e}")

        if not loaded:
            print(f"  WARNING: No CSV found for {display_name} ({base_name}_{version}.csv)")

    return dfs, names, pdb_dirs


def print_statistics(dfs, names):
    """Print summary statistics matching notebook cell 22."""
    model_organism_dfs = [df for name, df in zip(names, dfs) if not name.startswith("*")]
    print(f"\nNumber of model organism dataframes: {len(model_organism_dfs)}")

    if not model_organism_dfs:
        print("No model organism data found.")
        return

    concat_df = pd.concat(model_organism_dfs)
    total = len(concat_df)
    print(f"Number of proteins in concatenated dataframe: {total}")

    low_plddt = concat_df[concat_df["mean_plddt"] < 0.7]
    print(f"Number of proteins with mean pLDDT < 0.7: "
          f"{len(low_plddt)} ({len(low_plddt)/total*100:.2f}%)")

    low_plddt_ordered = low_plddt[low_plddt["mean_disorder_score"] < 0.5]
    print(f"Number of proteins with mean pLDDT < 0.7 and mean disorder score < 0.5: "
          f"{len(low_plddt_ordered)} ({len(low_plddt_ordered)/total*100:.2f}%)")


def plot_masked_plddt_distribution(dfs, names, output_path, dpi=300):
    """Plot histogram of masked mean pLDDT split by high/low confidence (cell 24)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row
    bin_edges = np.linspace(0, 1, 100)

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("Mean pLDDT Masked by AIUPred IDR Prediction", y=0.915)

    for i in range(len(dfs)):
        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        hi = dfs[i][dfs[i]["masked_mean_plddt"] > 0.7]
        lo = dfs[i][dfs[i]["masked_mean_plddt"] < 0.7]
        mean_len_hi = hi['sequence'].apply(len).mean() if len(hi) > 0 else 0
        mean_len_lo = lo['sequence'].apply(len).mean() if len(lo) > 0 else 0
        ax.hist(hi["masked_mean_plddt"], alpha=0.5, bins=bin_edges, color="blue",
                label=f"High Confidence (n={len(hi)}, Mean Length={mean_len_hi:.2f})")
        ax.hist(lo["masked_mean_plddt"], alpha=0.5, bins=bin_edges, color="red",
                label=f"Low Confidence (n={len(lo)}, Mean Length={mean_len_lo:.2f})")
        ax.axvline(x=0.7, color="green", linestyle="--", label="pLDDT Threshold=0.7")
        ax.set_title(names[i])
        ax.set_xlabel("Mean pLDDT")
        ax.set_ylabel("Number of Proteins")
        ax.legend()

    # Hide unused subplots
    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_disorder_vs_plddt_colored(dfs, names, output_path, dpi=300):
    """Scatter plot: mean disorder vs mean pLDDT coloured by confidence (cell 25)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("Mean Disorder Score vs Mean pLDDT", y=0.915)

    for i in range(len(dfs)):
        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        hi = dfs[i][dfs[i]["masked_mean_plddt"] > 0.7]
        lo = dfs[i][dfs[i]["masked_mean_plddt"] < 0.7]
        ax.scatter(hi["mean_plddt"], hi["disorder_score"].apply(np.mean),
                   s=10, alpha=0.3, color="blue", label="High Confidence")
        ax.scatter(lo["mean_plddt"], lo["disorder_score"].apply(np.mean),
                   s=10, alpha=0.3, color="red", label="Low Confidence")
        ax.set_title(names[i])
        ax.set_xlabel("Mean pLDDT")
        ax.set_ylabel("Mean Disorder Score")
        ax.legend()

    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_disorder_vs_plddt_hex(dfs, names, output_path, dpi=300):
    """Hexbin density plot: mean disorder vs mean pLDDT (cell 26)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("Mean Disorder Score vs Mean pLDDT", y=0.915)

    for i in range(len(dfs)):
        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.hexbin(dfs[i]["mean_plddt"], dfs[i]["disorder_score"].apply(np.mean),
                  extent=[0, 1, 0, 1], gridsize=(50, 50))
        ax.set_title(names[i])
        ax.set_xlabel("Mean pLDDT")
        ax.set_ylabel("Mean Disorder Score")
        rect = patches.Rectangle((0.01, 0.01), 0.69, 0.49,
                                  linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_disorder_vs_plddt_hex_log(dfs, names, output_path, dpi=300):
    """Log-scale hexbin with ordered-low-confidence fraction annotation (cell 27)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row
    disorder_cutoff = 0.5
    plddt_cutoff = 0.7

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))

    for i in range(len(dfs)):
        ordered = dfs[i]["disorder_score"].apply(lambda x: np.mean(x) < disorder_cutoff)
        low_conf = dfs[i]["plddt"].apply(lambda x: np.mean(x) < plddt_cutoff)
        frac = dfs[i][ordered & low_conf].shape[0] / dfs[i].shape[0]

        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.hexbin(dfs[i]["mean_plddt"], dfs[i]["disorder_score"].apply(np.mean),
                  norm="log", gridsize=(30, 30), extent=[0, 1, 0, 1], linewidths=0.5)
        ax.set_title(names[i])
        ax.set_xlabel("Mean pLDDT")
        ax.set_ylabel("Mean Disorder Score")
        rect = patches.Rectangle(
            (0.01, 0.01), 0.69, 0.49, linewidth=1, edgecolor='r', facecolor='none',
            label=f"Predicted ordered low confidence fraction={frac:.2f}"
        )
        ax.add_patch(rect)
        ax.legend()

    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_proportion_disordered_vs_masked_plddt(dfs, names, output_path, dpi=300):
    """Scatter: proportion of disordered residues vs masked mean pLDDT (cell 28)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("Proportion Disordered vs Masked Mean pLDDT", y=0.915)

    for i in range(len(dfs)):
        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        hi = dfs[i][dfs[i]["masked_mean_plddt"] > 0.7]
        lo = dfs[i][dfs[i]["masked_mean_plddt"] < 0.7]
        ax.scatter(hi["masked_mean_plddt"],
                   hi["disorder_score"].apply(lambda x: np.mean(x < 0.5)),
                   s=10, alpha=0.3, color="blue", label="High Confidence")
        ax.scatter(lo["masked_mean_plddt"],
                   lo["disorder_score"].apply(lambda x: np.mean(x < 0.5)),
                   s=10, alpha=0.3, color="red", label="Low Confidence")
        ax.set_title(names[i])
        ax.set_xlabel("Masked Mean pLDDT")
        ax.set_ylabel("Proportion Disordered")
        ax.legend()

    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def get_top_p_ordered_mean_plddt(row, p=0.5):
    """Return mean pLDDT of the top-p most ordered residues."""
    plddt = row["plddt"]
    disorder_score = row["disorder_score"]
    sort_index = np.argsort(disorder_score)
    top_p_index = sort_index[:int(len(sort_index) * p)]
    return np.mean(plddt[top_p_index])


def plot_top_ordered_plddt(dfs, names, output_path, dpi=300):
    """Scatter: top 50% ordered residues mean pLDDT vs overall mean pLDDT (cell 29)."""
    num_plot_per_row = 4
    num_rows = (len(dfs) + num_plot_per_row - 1) // num_plot_per_row

    fig, axs = plt.subplots(num_rows, num_plot_per_row, figsize=(20, 5 * num_rows))
    fig.suptitle("Top 50% Ordered Mean pLDDT vs Mean pLDDT")

    for i in range(len(dfs)):
        hi = dfs[i][dfs[i]["masked_mean_plddt"] > 0.7].copy()
        lo = dfs[i][dfs[i]["masked_mean_plddt"] < 0.7].copy()
        hi["top_05"] = hi.apply(lambda x: get_top_p_ordered_mean_plddt(x, 0.5), axis=1)
        lo["top_05"] = lo.apply(lambda x: get_top_p_ordered_mean_plddt(x, 0.5), axis=1)

        row = i // num_plot_per_row
        column = i % num_plot_per_row
        ax = axs[row, column]
        ax.scatter(hi["mean_plddt"], hi["top_05"], s=10, alpha=0.3,
                   color="blue", label="High Confidence")
        ax.scatter(lo["mean_plddt"], lo["top_05"], s=10, alpha=0.3,
                   color="red", label="Low Confidence")
        ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="y=x")
        ax.set_title(names[i])
        ax.set_xlabel("Mean pLDDT")
        ax.set_ylabel("Top 50% Ordered Mean pLDDT")
        ax.legend()

    for i in range(len(dfs), num_rows * num_plot_per_row):
        axs[i // num_plot_per_row, i % num_plot_per_row].set_visible(False)

    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze AFDB AIUPred results and generate plots')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing result CSV files')
    parser.add_argument('--output_dir', default=None,
                        help='Directory to save plots (default: same as data_dir)')
    parser.add_argument('--version', default='v6',
                        help='Version suffix used in CSV filenames (default: v6)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure DPI (default: 300)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading datasets from {args.data_dir} (version={args.version})...")
    dfs, names, pdb_dirs = load_datasets(args.data_dir, args.version)

    if not dfs:
        print("No datasets loaded. Exiting.")
        sys.exit(1)

    print_statistics(dfs, names)

    print("\nGenerating plots...")
    plot_masked_plddt_distribution(
        dfs, names,
        os.path.join(output_dir, "mean_masked_plddt_distribution.png"), args.dpi)

    plot_disorder_vs_plddt_colored(
        dfs, names,
        os.path.join(output_dir, "mean_plddt_vs_mean_disorder_score_colored.png"), args.dpi)

    plot_disorder_vs_plddt_hex(
        dfs, names,
        os.path.join(output_dir, "mean_plddt_vs_mean_disorder_score_hex.png"), args.dpi)

    plot_disorder_vs_plddt_hex_log(
        dfs, names,
        os.path.join(output_dir, "mean_plddt_vs_mean_disorder_score_hex_no_title.png"), args.dpi)

    plot_proportion_disordered_vs_masked_plddt(
        dfs, names,
        os.path.join(output_dir, "proportion_disordered_vs_masked_mean_plddt.png"), args.dpi)

    plot_top_ordered_plddt(
        dfs, names,
        os.path.join(output_dir, "top_50_ordered_mean_plddt_vs_mean_plddt.png"), args.dpi)

    print("\nDone.")


if __name__ == "__main__":
    main()
