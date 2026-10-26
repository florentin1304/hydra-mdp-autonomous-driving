"""Packed HDF5 field names and default caps."""

from __future__ import annotations

from dataclasses import dataclass

from womd_il import constants as C


@dataclass
class PackConfig:
    num_agents: int = C.NUM_AGENTS
    num_map_polylines: int = C.NUM_MAP_POLYLINES
    polyline_points: int = C.POLYLINE_POINTS
    num_traffic_lights: int = C.NUM_TRAFFIC_LIGHTS
    num_route_paths: int = C.NUM_ROUTE_PATHS
    route_points: int = C.ROUTE_POINTS
    history_steps: int = C.HISTORY_STEPS
    future_steps: int = C.FUTURE_STEPS
    current_step: int = C.CURRENT_STEP
    dt: float = C.DT

    @property
    def total_steps(self) -> int:
        return self.history_steps + self.future_steps

    @classmethod
    def from_cfg(cls, cfg) -> "PackConfig":
        return cls(
            num_agents=int(cfg.num_agents),
            num_map_polylines=int(cfg.num_map_polylines),
            polyline_points=int(cfg.polyline_points),
            num_traffic_lights=int(cfg.num_traffic_lights),
            num_route_paths=int(cfg.num_route_paths),
            route_points=int(cfg.route_points),
            history_steps=int(cfg.history_steps),
            future_steps=int(cfg.future_steps),
            current_step=int(cfg.current_step),
            dt=float(cfg.dt),
        )


DATASETS = (
    "xyz",
    "yaw",
    "vel",
    "valid",
    "length",
    "width",
    "height",
    "type",
    "is_sdc",
    "tracks_to_predict",
    "objects_of_interest",
    "map_xyz",
    "map_valid",
    "map_type",
    "map_on_route",
    "tl_xyz",
    "tl_state",
    "tl_valid",
    "route_xyz",
    "route_valid",
    "route_on_route",
    "sdc_index",
)
