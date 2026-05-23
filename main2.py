from functions.read_muons import read_muon_events
from functions.selection import process_events, apply_isolation_cut, get_masses
from functions.fit_model import fit_mass_histogram, fit_function
from functions.optimization import scan_isolation_cut, find_best_isolation

import numpy as np
import matplotlib.pyplot as plt


DATA_PATH = "data/input/mini_muons.txt"


def plot_fit_result(fit_result, title):
    counts = fit_result["counts"]
    edges = fit_result["edges"]
    centers = fit_result["centers"]
    params = fit_result["params"]

    width = edges[1] - edges[0]

    x_fit = np.linspace(edges[0], edges[-1], 1000)
    y_fit = fit_function(x_fit, *params)

    plt.figure(figsize=(9, 6))

    plt.bar(
        centers,
        counts,
        width=width,
        alpha=0.7,
        label="Data"
    )

    plt.plot(
        x_fit,
        y_fit,
        linewidth=2,
        label="Signal + Background Fit"
    )

    plt.xlabel("Invariant Mass [GeV]")
    plt.ylabel("Events / bin")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_isolation_scan(results):
    iso_cuts = [r["iso_cut"] for r in results]
    signal_fractions = [r["signal_fraction"] for r in results]
    uncertainties = [r["uncertainty"] for r in results]

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        iso_cuts,
        signal_fractions,
        yerr=uncertainties,
        fmt="o-",
        capsize=3
    )

    plt.xlabel("Isolation Cut")
    plt.ylabel("Signal Fraction")
    plt.title("Signal Fraction vs Isolation Cut")
    plt.tight_layout()
    plt.show()


def main():
    print("Reading muon data...")
    events = read_muon_events(DATA_PATH)

    print(f"Total events read: {len(events)}")

    processed_events = process_events(events)
    print(f"Events with exactly two muons: {len(processed_events)}")

    # ---------- no isolation cut / loose baseline ----------
    baseline_masses = get_masses(processed_events, opposite_sign_only=True)
    print(f"Opposite-sign events before isolation cut: {len(baseline_masses)}")

    baseline_fit = fit_mass_histogram(
        baseline_masses,
        bins=80,
        mass_range=(70, 110)
    )

    print("\nBaseline fit:")
    print(f"Signal fraction = {baseline_fit['signal_fraction']:.4f}")
    print(f"Uncertainty      = {baseline_fit['signal_fraction_uncertainty']:.4f}")

    plot_fit_result(
        baseline_fit,
        "Invariant Mass Fit Before Isolation Cut"
    )

    # ---------- scan isolation cut ----------
    print("\nScanning isolation cuts...")

    results = scan_isolation_cut(
        processed_events,
        iso_min=0.1,
        iso_max=5.0,
        iso_step=0.1,
        bins=80,
        mass_range=(70, 110)
    )

    if not results:
        print("No successful fits found.")
        return

    best = find_best_isolation(results)

    print("\nBest isolation cut:")
    print(f"Iso cut           = {best['iso_cut']:.2f}")
    print(f"Number of events  = {best['n_events']}")
    print(f"Signal fraction   = {best['signal_fraction']:.4f}")
    print(f"Uncertainty       = {best['uncertainty']:.4f}")
    print(f"Relative error    = {best['relative_uncertainty']:.4f}")

    plot_isolation_scan(results)

    plot_fit_result(
        best["fit_result"],
        f"Invariant Mass Fit at Best Isolation Cut = {best['iso_cut']:.2f}"
    )


if __name__ == "__main__":
    main()