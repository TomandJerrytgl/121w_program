from functions.lab1_step1 import extract_masses
from functions.significance import scan_significance
import numpy as np
import matplotlib.pyplot as plt
import os

DATA_PATH = "data/input/p121_lab1_data/"

samples = {
    200: ("zp_mzp200_electrons.txt", "ee_mll175_electrons.txt"),
    300: ("zp_mzp300_electrons.txt", "ee_mll275_electrons.txt"),
    400: ("zp_mzp400_electrons.txt", "ee_mll375_electrons.txt"),
    500: ("zp_mzp500_electrons.txt", "ee_mll450_electrons.txt"),
    750: ("zp_mzp750_electrons.txt", "ee_mll650_electrons.txt"),
    1000: ("zp_mzp1000_electrons.txt", "ee_mll900_electrons.txt"),
}

weights = {
    200: {"signal": 3.0, "background": 2.0},
    300: {"signal": 2.0, "background": 2.0},
    400: {"signal": 1.5, "background": 2.0},
    500: {"signal": 1.0, "background": 2.0},
    750: {"signal": 0.75, "background": 2.0},
    1000: {"signal": 0.50, "background": 2.0},
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

print("\nTable 2 data:")
print(
    f"{'Mass':<8} "
    f"{'Best δ':<10} "
    f"{'S':<12} "
    f"{'B':<14} "
    f"{'√B':<10} "
    f"{'Significance':<10}"
)

for i, mzp in enumerate(samples):

    signal_file, background_file = samples[mzp]

    signal_path = os.path.join(DATA_PATH, signal_file)
    background_path = os.path.join(DATA_PATH, background_file)

    signal_masses, _, _ = extract_masses(signal_path)
    background_masses, _, _ = extract_masses(background_path)

    signal_weight = weights[mzp]["signal"]
    background_weight = weights[mzp]["background"]

    # ---------- 自动 plot range ----------
    all_masses = signal_masses + background_masses
    rough_edges = np.linspace(min(all_masses), max(all_masses), 100)
    rough_centers = 0.5 * (rough_edges[:-1] + rough_edges[1:])

    s_counts_rough, _ = np.histogram(
        signal_masses,
        bins=rough_edges,
        weights=np.full(len(signal_masses), signal_weight)
    )

    peak = rough_centers[np.argmax(s_counts_rough)]

    half_width = max(50, 0.1 * mzp)
    mass_range = (peak - half_width, peak + half_width)

    # ---------- histogram ----------
    bins = 30
    edges = np.linspace(mass_range[0], mass_range[1], bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    b_counts, _ = np.histogram(
        background_masses,
        bins=edges,
        weights=np.full(len(background_masses), background_weight)
    )

    s_counts, _ = np.histogram(
        signal_masses,
        bins=edges,
        weights=np.full(len(signal_masses), signal_weight)
    )

    sb_counts = b_counts + s_counts

    ax = axes[i]

    ax.bar(
        centers,
        sb_counts,
        width=width,
        color="forestgreen",
        alpha=0.8,
        label="Signal + Bkg"
    )

    ax.bar(
        centers,
        b_counts,
        width=width,
        color="tomato",
        alpha=0.9,
        label="Bkg"
    )

    ax.set_title(f"{mzp} GeV", fontsize=14)
    ax.set_xlim(mass_range)

    ymax = max(sb_counts) * 1.1 if max(sb_counts) > 0 else 1
    ax.set_ylim(0, ymax)

    ax.set_xlabel("Mass [GeV]", fontsize=12)
    ax.set_ylabel("Entries / bin", fontsize=12)
    ax.tick_params(labelsize=11)

    # ---------- significance scan ----------
    deltas, signal_counts, background_counts, significances, best_delta, best_sig = scan_significance(
        signal_masses,
        background_masses,
        mzp,
        signal_weight,
        background_weight
    )

    best_index = deltas.index(best_delta)

    best_S = signal_counts[best_index]
    best_B = background_counts[best_index]
    sqrt_B = np.sqrt(best_B)

    print(
        f"{mzp:<8} "
        f"{best_delta:<10} "
        f"{best_S:<12.2f} "
        f"{best_B:<14.2f} "
        f"{sqrt_B:<10.2f} "
        f"{best_sig:<10.2f}"
    )

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=12)

fig.suptitle("Invariant Mass Distributions for Z′ Mass Hypotheses", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()