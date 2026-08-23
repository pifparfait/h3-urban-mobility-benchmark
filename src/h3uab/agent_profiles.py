"""
Compact synthetic agent profiles and activity schedules.

These are controlled behavioral archetypes for benchmarking. They are NOT
claimed to reproduce Madrid demographics.

The goal is to give the Continuous and H3 simulations the SAME interpretable
activity logic so that OD, accessibility, occupancy and mobility metrics have
urban meaning.

Only one uint8 profile id is stored per agent. The schedule templates are tiny
global matrices shared by all agents, which remains suitable for large N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from h3uab.semantic_layers import code_to_id, id_to_code


# ---------------------------------------------------------------------
# Profile IDs
# ---------------------------------------------------------------------

PROFILE_WORKER = 0
PROFILE_STUDENT = 1
PROFILE_SENIOR = 2
PROFILE_CAREGIVER = 3

PROFILE_NAMES = (
    "worker",
    "student",
    "senior",
    "caregiver",
)

# Controlled synthetic proportions. They are benchmark parameters, not
# demographic estimates.
PROFILE_PROBABILITIES = np.asarray(
    [0.50, 0.20, 0.15, 0.15],
    dtype=float,
)


# ---------------------------------------------------------------------
# Activity-purpose IDs
# ---------------------------------------------------------------------

PURPOSE_HOME = 0
PURPOSE_WORK = 1
PURPOSE_EDUCATION = 2
PURPOSE_LUNCH = 3
PURPOSE_SHOPPING = 4
PURPOSE_HEALTH = 5
PURPOSE_LEISURE = 6
PURPOSE_CAFE = 7
PURPOSE_DAYCARE = 8

PURPOSE_NAMES = {
    PURPOSE_HOME: "home",
    PURPOSE_WORK: "work",
    PURPOSE_EDUCATION: "education",
    PURPOSE_LUNCH: "lunch",
    PURPOSE_SHOPPING: "shopping",
    PURPOSE_HEALTH: "health",
    PURPOSE_LEISURE: "leisure",
    PURPOSE_CAFE: "cafe",
    PURPOSE_DAYCARE: "daycare",
}


# ---------------------------------------------------------------------
# Purpose -> ordered D-layer candidates.
#
# The first available layer is used. This makes the schedule compatible with
# the current 5-category legacy OSM cache without inventing missing categories.
#
# A future richer OSM extraction automatically uses the more specific layer.
# ---------------------------------------------------------------------

PURPOSE_LAYER_CANDIDATES = {
    PURPOSE_HOME: (),
    PURPOSE_WORK: ("D4", "D20", "D14", "D3"),
    PURPOSE_EDUCATION: ("D5", "D18", "D3"),
    PURPOSE_LUNCH: ("D21", "D22", "D3"),
    PURPOSE_SHOPPING: ("D15", "D3"),
    PURPOSE_HEALTH: ("D8", "D17"),
    PURPOSE_LEISURE: ("D25", "D16", "D21"),
    PURPOSE_CAFE: ("D22", "D21"),
    PURPOSE_DAYCARE: ("D19", "D5"),
}


# 8-step daily activity templates.
PROFILE_PURPOSE_TEMPLATES = np.asarray(
    [
        # worker
        [
            PURPOSE_HOME,
            PURPOSE_WORK,
            PURPOSE_LUNCH,
            PURPOSE_WORK,
            PURPOSE_SHOPPING,
            PURPOSE_LEISURE,
            PURPOSE_HOME,
            PURPOSE_HOME,
        ],

        # student
        [
            PURPOSE_HOME,
            PURPOSE_EDUCATION,
            PURPOSE_LUNCH,
            PURPOSE_EDUCATION,
            PURPOSE_CAFE,
            PURPOSE_LEISURE,
            PURPOSE_HOME,
            PURPOSE_HOME,
        ],

        # senior
        [
            PURPOSE_HOME,
            PURPOSE_HEALTH,
            PURPOSE_LEISURE,
            PURPOSE_LUNCH,
            PURPOSE_SHOPPING,
            PURPOSE_LEISURE,
            PURPOSE_HOME,
            PURPOSE_HOME,
        ],

        # caregiver
        [
            PURPOSE_HOME,
            PURPOSE_DAYCARE,
            PURPOSE_WORK,
            PURPOSE_LUNCH,
            PURPOSE_SHOPPING,
            PURPOSE_DAYCARE,
            PURPOSE_HOME,
            PURPOSE_HOME,
        ],
    ],
    dtype=np.uint8,
)


def purpose_name(purpose_id: int) -> str:
    return PURPOSE_NAMES.get(
        int(purpose_id),
        f"purpose_{int(purpose_id)}",
    )


def profile_name(profile_id: int) -> str:
    profile_id = int(profile_id)

    if 0 <= profile_id < len(PROFILE_NAMES):
        return PROFILE_NAMES[profile_id]

    return f"profile_{profile_id}"


def resolve_purpose_layer(
    purpose_id: int,
    active_layer_codes,
) -> int:
    """
    Resolve an activity purpose to the most specific semantic layer available
    in the current OSM cache.

    Returns a compact layer id. 0 means no spatial query / stay home.
    """
    purpose_id = int(purpose_id)

    if purpose_id == PURPOSE_HOME:
        return 0

    available = set(
        str(x)
        for x in active_layer_codes
    )

    for code in PURPOSE_LAYER_CANDIDATES.get(
        purpose_id,
        (),
    ):
        if code in available:
            return code_to_id(code)

    # If no valid urban layer exists, the activity becomes a stay-in-place
    # event rather than inventing data.
    return 0


def build_schedule_tables(
    active_layer_codes,
    n_steps: int,
):
    """
    Build tiny shared profile schedule matrices.

    Returns:
        purpose_table: shape (n_profiles, n_steps), uint8
        layer_table:   shape (n_profiles, n_steps), uint8
        fallback_metadata: readable mapping used in this city dataset
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    n_profiles = len(PROFILE_NAMES)

    purpose_table = np.empty(
        (n_profiles, n_steps),
        dtype=np.uint8,
    )

    base_steps = (
        PROFILE_PURPOSE_TEMPLATES.shape[1]
    )

    for p in range(n_profiles):
        for step in range(n_steps):
            purpose_table[p, step] = (
                PROFILE_PURPOSE_TEMPLATES[
                    p,
                    step % base_steps,
                ]
            )

    layer_table = np.zeros_like(
        purpose_table,
        dtype=np.uint8,
    )

    for p in range(n_profiles):
        for step in range(n_steps):
            layer_table[p, step] = (
                resolve_purpose_layer(
                    int(purpose_table[p, step]),
                    active_layer_codes,
                )
            )

    fallback_metadata = {}

    for purpose_id, purpose_label in PURPOSE_NAMES.items():
        resolved = resolve_purpose_layer(
            purpose_id,
            active_layer_codes,
        )

        fallback_metadata[
            purpose_label
        ] = (
            id_to_code(resolved)
            if resolved != 0
            else "HOME/NO_QUERY"
        )

    return (
        purpose_table,
        layer_table,
        fallback_metadata,
    )


def sample_profiles(
    n_agents: int,
    rng: np.random.Generator,
):
    """
    One compact uint8 profile id per agent.
    """
    return rng.choice(
        len(PROFILE_NAMES),
        size=n_agents,
        replace=True,
        p=PROFILE_PROBABILITIES,
    ).astype(np.uint8)
