import numpy as np

from functions.selection import apply_isolation_cut, get_masses
from functions.fit_model import fit_mass_histogram


def scan_isolation_cut(
    processed_events,
    iso_min=0.1,
    iso_max=5.0,
    iso_step=0.1,
    bins=80,
    mass_range=(70, 110)
):
    """
    Scan isolation cuts and fit invariant mass histogram.
    """

    results = []

    iso_values = np.arange(
        iso_min,
        iso_max + iso_step,
        iso_step
    )

    for iso_cut in iso_values:

        selected = apply_isolation_cut(
            processed_events,
            iso_cut
        )

        masses = get_masses(
            selected,
            opposite_sign_only=True
        )

        if len(masses) < 20:
            continue

        try:

            fit_result = fit_mass_histogram(
                masses,
                bins=bins,
                mass_range=mass_range
            )

            signal_fraction = fit_result["signal_fraction"]

            n_events = len(masses)
            signal_count = signal_fraction * n_events
            background_count = (1 - signal_fraction) * n_events
            if signal_count > 0:
                uncertainty = np.sqrt(n_events) / signal_count
            else:
                uncertainty = np.inf
            relative_uncertainty = uncertainty
            #uncertainty = fit_result[
            #    "signal_fraction_uncertainty"
            #]

            #relative_uncertainty = (
           #     uncertainty / signal_fraction
           #     if signal_fraction > 0
           #     else np.inf
           # )

            results.append({
                "iso_cut": iso_cut,
                "n_events": len(masses),
                "signal_fraction": signal_fraction,
                "signal_count": signal_count,
                "background_count": background_count,
                "uncertainty": uncertainty,
                "relative_uncertainty": relative_uncertainty,
                "fit_result": fit_result,
            })

        except RuntimeError:
            continue

    return results


def find_best_isolation(results):
    """
    Choose best isolation cut by minimizing uncertainty.
    """

    if not results:
        return None

    return min(
        results,
        key=lambda r: r["uncertainty"]
    )