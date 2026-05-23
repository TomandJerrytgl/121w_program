import numpy as np
from scipy.optimize import curve_fit
from scipy.special import wofz


def voigt(x, mean, alpha, gamma):
    """
    Voigt line shape for Z signal.
    alpha: Gaussian HWHM-like width
    gamma: Lorentzian HWHM
    """

    sigma = alpha / np.sqrt(2 * np.log(2))
    z = ((x - mean) + 1j * gamma) / (sigma * np.sqrt(2))

    return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))


def exponential_background(x, tau):
    """
    Falling exponential background shape.
    """

    return np.exp(-x / tau)


def fit_function(x, A, s, mean, alpha, gamma, tau):
    """
    Full model:
        A * [(1-s) * background + s * signal]
    """

    signal = voigt(x, mean, alpha, gamma)
    background = exponential_background(x, tau)

    return A * ((1 - s) * background + s * signal)


def background_only_function(x, A, tau):
    """
    Background-only model used for plotting.
    """

    return A * exponential_background(x, tau)


def fit_mass_histogram(masses, bins=80, mass_range=(70, 110)):
    """
    Histogram mass data and fit signal + background model.

    Returns:
        fit result dictionary
    """

    counts, edges = np.histogram(masses, bins=bins, range=mass_range)
    centers = 0.5 * (edges[:-1] + edges[1:])

    uncertainties = np.sqrt(counts)
    uncertainties[uncertainties == 0] = 1.0

    initial_guess = [
        max(counts),  # A
        0.8,          # signal fraction s
        91.0,         # Z mass mean
        2.0,          # alpha
        2.5,          # gamma
        40.0,         # tau
    ]

    bounds = (
        [0, 0, 80, 0.1, 0.1, 1],
        [np.inf, 1, 100, 20, 20, 500]
    )

    popt, pcov = curve_fit(
        fit_function,
        centers,
        counts,
        p0=initial_guess,
        sigma=uncertainties,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000
    )

    perr = np.sqrt(np.diag(pcov))

    return {
        "counts": counts,
        "edges": edges,
        "centers": centers,
        "uncertainties": uncertainties,
        "params": popt,
        "errors": perr,
        "signal_fraction": popt[1],
        "signal_fraction_uncertainty": perr[1],
    }