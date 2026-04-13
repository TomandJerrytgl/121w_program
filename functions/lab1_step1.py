import math


def read_events(filename):
    """
    Read collision events from a text file.

    Each event is stored as:
        NumElectrons: N
        Electron 1 Pt ... Eta ... Phi ... Charge ...
        Electron 2 Pt ... Eta ... Phi ... Charge ...
        ...

    Returns:
        events: list of dict
            Example:
            {
                "num_electrons": 2,
                "electrons": [
                    {"pt": 519.673, "eta": 0.424, "phi": 2.739, "charge": 1},
                    {"pt": 469.388, "eta": 0.798, "phi": -0.511, "charge": -1}
                ]
            }
    """
    events = []

    with open(filename, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Start of a new event
        if line.startswith("NumElectrons:"):
            num_electrons = int(line.split(":")[1].strip())
            event = {
                "num_electrons": num_electrons,
                "electrons": []
            }

            # Read the following electron lines
            for _ in range(num_electrons):
                i += 1
                electron_line = lines[i]
                parts = electron_line.split()

                # Expected format:
                # Electron 1 Pt 519.673 Eta 0.424 Phi 2.739 Charge 1
                electron = {
                    "pt": float(parts[3]),
                    "eta": float(parts[5]),
                    "phi": float(parts[7]),
                    "charge": int(parts[9])
                }
                event["electrons"].append(electron)

            events.append(event)

        i += 1

    return events


def is_good_event(event):
    """
    Check whether the event passes the lab selection:
    - exactly two electrons
    - opposite charge
    """
    if event["num_electrons"] != 2:
        return False

    e1, e2 = event["electrons"]
    return (e1["charge"] * e2["charge"] == -1)


def electron_four_vector(pt, eta, phi, mass=0.000511):
    """
    Convert (pt, eta, phi) into a four-vector (E, px, py, pz).

    Uses:
        px = pt * cos(phi)
        py = pt * sin(phi)
        pz = pt * sinh(eta)
        E  = sqrt(px^2 + py^2 + pz^2 + m^2)

    mass default is electron mass in GeV.
    """
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)
    energy = math.sqrt(px**2 + py**2 + pz**2 + mass**2)

    return energy, px, py, pz


def invariant_mass(electron1, electron2):
    """
    Compute invariant mass of two electrons from their four-vectors.

    m^2 = E^2 - px^2 - py^2 - pz^2
    """
    E1, px1, py1, pz1 = electron_four_vector(
        electron1["pt"], electron1["eta"], electron1["phi"]
    )
    E2, px2, py2, pz2 = electron_four_vector(
        electron2["pt"], electron2["eta"], electron2["phi"]
    )

    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2

    m2 = E**2 - px**2 - py**2 - pz**2

    # Protect against tiny negative values from floating-point roundoff
    if m2 < 0:
        m2 = 0.0

    return math.sqrt(m2)


def extract_masses(filename):
    """
    Read a file, select valid events, and compute invariant masses.

    Returns:
        masses: list of invariant masses
        total_events: total number of events read
        selected_events: number of selected good events
    """
    events = read_events(filename)

    masses = []
    selected_events = 0

    for event in events:
        if is_good_event(event):
            e1, e2 = event["electrons"]
            mass = invariant_mass(e1, e2)
            masses.append(mass)
            selected_events += 1

    return masses, len(events), selected_events


def save_masses_to_file(masses, output_filename):
    """
    Optional helper:
    Save invariant masses to a text file, one mass per line.
    """
    with open(output_filename, "w") as f:
        for m in masses:
            f.write(f"{m:.6f}\n")


def main():
    # Change this to your actual file name
    input_file = "zp_mzp750_electrons.txt"
    output_file = "masses_output.txt"

    masses, total_events, selected_events = extract_masses(input_file)

    print(f"Input file: {input_file}")
    print(f"Total events read: {total_events}")
    print(f"Selected good events: {selected_events}")
    print(f"Rejected events: {total_events - selected_events}")

    print("\nFirst 10 invariant masses:")
    for m in masses[:10]:
        print(f"{m:.3f} GeV")

    # Optional: save masses for later histogram work
    save_masses_to_file(masses, output_file)
    print(f"\nSaved masses to: {output_file}")

