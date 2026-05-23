import gzip


def read_muon_events(file_path):
    """
    Read CMS muon data from .txt or .txt.gz file.

    Returns:
        events: list of events
        each event is a list of muon dictionaries
    """

    events = []

    open_func = gzip.open if file_path.endswith(".gz") else open
    mode = "rt"

    with open_func(file_path, mode) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("NumMuons:"):
            num_muons = int(line.split(":")[1].strip())
            muons = []

            for _ in range(num_muons):
                i += 1
                parts = lines[i].strip().split()

                muon = {
                    "pt": float(parts[3]),
                    "eta": float(parts[5]),
                    "phi": float(parts[7]),
                    "charge": int(parts[9]),
                    "iso": float(parts[11]),
                }

                muons.append(muon)

            events.append(muons)

        i += 1

    return events