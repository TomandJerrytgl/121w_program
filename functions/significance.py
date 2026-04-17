import numpy as np
import matplotlib.pyplot as plt


def count_weighted_events_in_window(masses, center, delta, weight):
    """
    Count weighted events in the symmetric window [center-delta, center+delta].
    """
    lower = center - delta
    upper = center + delta

    count = 0.0
    for m in masses:
        if lower <= m <= upper:
            count += weight

    return count


def compute_significance(signal_count, background_count):
    """
    significance = S / sqrt(B)
    Avoid division by zero.
    """
    if background_count <= 0:
        return 0.0
    return signal_count / np.sqrt(background_count)


def scan_significance(signal_masses, background_masses, mzp,
                      signal_weight, background_weight,
                      delta_min=1, delta_max=150, delta_step=1):
    """
    Scan symmetric mass windows around mzp and find the best delta.

    Returns
    -------
    deltas : list
        scanned delta values
    signal_counts : list
        weighted signal counts in each window
    background_counts : list
        weighted background counts in each window
    significances : list
        S/sqrt(B) for each window
    best_delta : float
        delta giving maximum significance
    best_significance : float
        maximum significance
    """

    deltas = []
    signal_counts = []
    background_counts = []
    significances = []

    best_delta = None
    best_significance = -1.0

    for delta in range(delta_min, delta_max + 1, delta_step):
        s = count_weighted_events_in_window(signal_masses, mzp, delta, signal_weight)
        b = count_weighted_events_in_window(background_masses, mzp, delta, background_weight)
        sig = compute_significance(s, b)

        deltas.append(delta)
        signal_counts.append(s)
        background_counts.append(b)
        significances.append(sig)

        if sig > best_significance:
            best_significance = sig
            best_delta = delta

    return deltas, signal_counts, background_counts, significances, best_delta, best_significance


def plot_significance_scan(deltas, significances, mzp, best_delta, best_significance):
    """
    Plot significance vs delta.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(deltas, significances, linewidth=2)

    plt.xlabel(r"$\delta$ [GeV]", fontsize=14)
    plt.ylabel(r"Significance $= S/\sqrt{B}$", fontsize=14)
    plt.title(f"Significance scan for {mzp} GeV", fontsize=16)

    plt.xlim(min(deltas), max(deltas))
    plt.ylim(0, max(significances) * 1.1)

    # mark best point
    plt.scatter([best_delta], [best_significance], s=50)
    plt.annotate(
        f"best δ = {best_delta}\nmax sig = {best_significance:.3f}",
        xy=(best_delta, best_significance),
        xytext=(best_delta + 5, best_significance * 0.9),
        fontsize=11
    )

    plt.tight_layout()
    plt.show()