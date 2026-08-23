import numpy as np

from h3uab.agent_profiles import (
    build_schedule_tables,
    sample_profiles,
)
from h3uab.agent_sampling import sample_agents
from h3uab.models import World
from h3uab.osm_data import load_osm_pois


def build_world(cfg):
    rng = np.random.default_rng(
        cfg.seed
    )

    pois = load_osm_pois(cfg)

    if pois.empty:
        raise RuntimeError(
            "No OSM POIs remain after semantic normalization."
        )

    active_layer_codes = tuple(
        sorted(
            pois[
                "layer_code"
            ]
            .astype(str)
            .unique()
        )
    )

    agent_lat, agent_lng, population_metadata = (
        sample_agents(
            cfg,
            rng,
        )
    )

    # Compact synthetic behavioral profiles.
    agent_profile_type = sample_profiles(
        cfg.n_agents,
        rng,
    )

    (
        schedule_purpose_ids,
        schedule_layer_ids,
        schedule_fallbacks,
    ) = build_schedule_tables(
        active_layer_codes,
        cfg.n_steps,
    )

    world = World(
        agent_lat=agent_lat,
        agent_lng=agent_lng,

        agent_profile_type=agent_profile_type,
        profile_schedule_purpose_ids=schedule_purpose_ids,
        profile_schedule_layer_ids=schedule_layer_ids,

        amenity_lat=pois[
            "lat"
        ].to_numpy(float),
        amenity_lng=pois[
            "lng"
        ].to_numpy(float),
        amenity_layer_code=pois[
            "layer_code"
        ].astype(str).to_numpy(),
        amenity_attractiveness=pois[
            "attractiveness"
        ].to_numpy(float),
        amenity_osm_id=pois[
            "osm_id"
        ].astype(str).to_numpy(),
        amenity_name=pois[
            "name"
        ].fillna("").astype(str).to_numpy(),

        active_layer_codes=active_layer_codes,
    )

    unique_profiles, profile_counts = np.unique(
        agent_profile_type,
        return_counts=True,
    )

    metadata = {
        **population_metadata,
        "n_real_osm_pois":
            int(len(pois)),
        "active_layer_codes":
            list(active_layer_codes),

        "behavior_model":
            "controlled_synthetic_profiles_and_schedules",

        "profile_counts":
            {
                int(k): int(v)
                for k, v in zip(
                    unique_profiles,
                    profile_counts,
                )
            },

        "schedule_layer_resolution":
            schedule_fallbacks,
    }

    return world, metadata
