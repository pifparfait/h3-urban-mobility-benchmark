from __future__ import annotations

from collections import Counter

import h3


class DynamicAgentState:
    """
    V9.8 dynamic H3 observation state.

    Critical rule:
      H3 does NOT define the physical trajectory.

    The agent has continuous physical coordinates in engine.py.
    We derive H12/H11 from those coordinates for:
      - fast spatial state lookup,
      - visualization,
      - occupancy,
      - local aggregation,
      - regional OD metrics.

    H12:
      fine agent location cell.

    H11:
      local occupancy / mobility visualization-aggregation layer.

    OSM:
      physical mobility/routing.
    """

    def __init__(
        self,
        world,
        h3_config,
    ):
        self.hcfg = (
            h3_config
        )

        self.agent_h12 = [
            h3.latlng_to_cell(
                float(
                    lat
                ),
                float(
                    lng
                ),
                self.hcfg.position_res,
            )
            for lat, lng in zip(
                world.agent_lat,
                world.agent_lng,
            )
        ]

        self.occupancy_h11 = Counter(
            h3.cell_to_parent(
                cell,
                self.hcfg.interaction_res,
            )
            for cell in self.agent_h12
        )

        self.mobility_flows_h8 = Counter()

        self.population_h9 = Counter(
            h3.cell_to_parent(
                cell,
                self.hcfg.accessibility_res,
            )
            for cell in self.agent_h12
        )

    def position(
        self,
        agent_idx,
    ):
        return self.agent_h12[
            agent_idx
        ]

    def move_to_coords(
        self,
        agent_idx,
        new_lat,
        new_lng,
    ):
        """
        Update H3 observation state FROM the new continuous coordinates.
        """
        old_h12 = self.agent_h12[
            agent_idx
        ]

        new_h12 = h3.latlng_to_cell(
            float(
                new_lat
            ),
            float(
                new_lng
            ),
            self.hcfg.position_res,
        )

        if old_h12 == new_h12:
            return (
                old_h12,
                new_h12,
            )

        old_h11 = h3.cell_to_parent(
            old_h12,
            self.hcfg.interaction_res,
        )

        new_h11 = h3.cell_to_parent(
            new_h12,
            self.hcfg.interaction_res,
        )

        if old_h11 != new_h11:
            self.occupancy_h11[
                old_h11
            ] -= 1

            if self.occupancy_h11[
                old_h11
            ] <= 0:
                del self.occupancy_h11[
                    old_h11
                ]

            self.occupancy_h11[
                new_h11
            ] += 1

        old_h8 = h3.cell_to_parent(
            old_h12,
            self.hcfg.amenity_functional_res,
        )

        new_h8 = h3.cell_to_parent(
            new_h12,
            self.hcfg.amenity_functional_res,
        )

        self.mobility_flows_h8[
            (
                old_h8,
                new_h8,
            )
        ] += 1

        old_h9 = h3.cell_to_parent(
            old_h12,
            self.hcfg.accessibility_res,
        )

        new_h9 = h3.cell_to_parent(
            new_h12,
            self.hcfg.accessibility_res,
        )

        if old_h9 != new_h9:
            self.population_h9[
                old_h9
            ] -= 1

            if self.population_h9[
                old_h9
            ] <= 0:
                del self.population_h9[
                    old_h9
                ]

            self.population_h9[
                new_h9
            ] += 1

        self.agent_h12[
            agent_idx
        ] = new_h12

        return (
            old_h12,
            new_h12,
        )
