import matplotlib.pyplot as plt
import numpy as np


def plot_histograms(signal_masses, background_masses, mzp,
                    signal_weight, background_weight,
                    bins=30, mass_range=None):

    # 自动聚焦
    if mass_range is None:
        half_width = 75
        mass_range = (mzp - half_width, mzp + half_width)

    # 统一bin
    bin_edges = np.linspace(mass_range[0], mass_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    # 加权 histogram
    background_counts, _ = np.histogram(
        background_masses,
        bins=bin_edges,
        weights=np.full(len(background_masses), background_weight)
    )

    signal_counts, _ = np.histogram(
        signal_masses,
        bins=bin_edges,
        weights=np.full(len(signal_masses), signal_weight)
    )

    # 核心：S+B
    signal_plus_background = background_counts + signal_counts

    # ---------- 画图 ----------
    plt.figure(figsize=(8, 6))

    # S+B（绿色）
    plt.bar(
        bin_centers,
        signal_plus_background,
        width=bin_width,
        color="forestgreen",
        alpha=0.8,
        label="Signal + Bkg"
    )

    # Background（红色）
    plt.bar(
        bin_centers,
        background_counts,
        width=bin_width,
        color="tomato",
        alpha=0.9,
        label="Bkg"
    )

    # 轴
    plt.xlabel("Mass [GeV]", fontsize=14)
    plt.ylabel("Entries / bin", fontsize=14)

    plt.title(f"{mzp} GeV", fontsize=16)

    plt.xlim(mass_range)

    # 自动y轴
    ymax = max(signal_plus_background) * 1.1
    plt.ylim(0, ymax)

    plt.tick_params(axis='both', labelsize=12)

    plt.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.show()