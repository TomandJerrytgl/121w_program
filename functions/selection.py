from functions.invariant_mass import invariant_mass


def select_two_muon_events(events):
    """
    Keep only events with exactly two muons.
    """

    return [event for event in events if len(event) == 2]


def process_events(events):
    """
    Process two-muon events and compute:
        invariant mass
        isolation
        charge relation
    """

    processed = []

    for event in select_two_muon_events(events):

        mu1, mu2 = event

        mass = invariant_mass(mu1, mu2)

        opposite_sign = (
            mu1["charge"] * mu2["charge"] == -1
        )

        processed.append({
            "mass": mass,
            "iso1": mu1["iso"],
            "iso2": mu2["iso"],
            "charge1": mu1["charge"],
            "charge2": mu2["charge"],
            "opposite_sign": opposite_sign,
        })

    return processed


def apply_isolation_cut(processed_events, iso_cut):
    """
    Keep events where both muons satisfy isolation cut.
    """

    return [
        event for event in processed_events
        if event["iso1"] <= iso_cut
        and event["iso2"] <= iso_cut
    ]


def get_masses(events, opposite_sign_only=True):
    """
    Extract invariant masses.
    """

    if opposite_sign_only:

        return [
            event["mass"]
            for event in events
            if event["opposite_sign"]
        ]

    return [event["mass"] for event in events]