import numpy as np

MUON_MASS = 0.105658  # GeV


def momentum_components(muon):
    """
    Convert pt, eta, phi into px, py, pz.
    """

    pt = muon["pt"]
    eta = muon["eta"]
    phi = muon["phi"]

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    return px, py, pz


def energy(muon):
    """
    Compute relativistic energy.
    """

    px, py, pz = momentum_components(muon)
    p2 = px**2 + py**2 + pz**2

    return np.sqrt(p2 + MUON_MASS**2)


def invariant_mass(muon1, muon2):
    """
    Compute invariant mass of two muons.
    """

    e1 = energy(muon1)
    e2 = energy(muon2)

    px1, py1, pz1 = momentum_components(muon1)
    px2, py2, pz2 = momentum_components(muon2)

    total_E = e1 + e2
    total_px = px1 + px2
    total_py = py1 + py2
    total_pz = pz1 + pz2

    mass2 = total_E**2 - total_px**2 - total_py**2 - total_pz**2

    if mass2 < 0:
        return 0.0

    return np.sqrt(mass2)