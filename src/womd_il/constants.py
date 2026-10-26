"""Shared WOMD / Wayformer constants."""

HISTORY_STEPS = 11  # 1 s @ 10 Hz including the current timestep
FUTURE_STEPS = 80  # 8 s @ 10 Hz
TOTAL_STEPS = HISTORY_STEPS + FUTURE_STEPS
CURRENT_STEP = HISTORY_STEPS - 1  # 10
DT = 0.1

NUM_AGENTS = 64
NUM_MAP_POLYLINES = 512
POLYLINE_POINTS = 20
NUM_TRAFFIC_LIGHTS = 32
NUM_ROUTE_PATHS = 16
ROUTE_POINTS = 128

# Object types (WOMD ObjectType)
TYPE_UNSET = 0
TYPE_VEHICLE = 1
TYPE_PEDESTRIAN = 2
TYPE_CYCLIST = 3
TYPE_OTHER = 4
NUM_OBJECT_TYPES = 5

# Map feature types used in packed arrays
MAP_UNKNOWN = 0
MAP_LANE_FREEWAY = 1
MAP_LANE_SURFACE = 2
MAP_LANE_BIKE = 3
MAP_ROAD_LINE = 4
MAP_ROAD_EDGE = 5
MAP_STOP_SIGN = 6
MAP_CROSSWALK = 7
MAP_SPEED_BUMP = 8
MAP_DRIVEWAY = 9
NUM_MAP_TYPES = 10

# Traffic-light lane states (WOMD TrafficSignalLaneState)
TL_UNKNOWN = 0
TL_ARROW_STOP = 1
TL_ARROW_CAUTION = 2
TL_ARROW_GO = 3
TL_STOP = 4
TL_CAUTION = 5
TL_GO = 6
TL_FLASHING_STOP = 7
TL_FLASHING_CAUTION = 8
NUM_TL_STATES = 9

# Raw feature dims (must match configs/model/wayformer.yaml)
# hist: xy2 + vel2 + acc2 + yaw_sin_cos2 + yaw_rate1 + size3 + type5 + valid1
HIST_IN_DIM = 18
# map: start2 + end2 + mid2 + q1_2 + q3_2 + dir2 + length1 + bow1 + type10 + dist1 + on_route1
MAP_IN_DIM = 26
# Lane-center polylines within this distance of the preferred SDC path are on-route.
# Lateral half-width of the on-route corridor (~1.5 lane widths).
# Matches nuPlan roadblock semantics: parallel same-direction lanes stay on-route.
ROUTE_LANE_THRESH_M = 6.0
# Min |cos| between lane tangent and local route tangent (reject cross-traffic).
ROUTE_LANE_HEADING_COS = 0.5

# tl: xy2 + state9 (per history step)
TL_IN_DIM = 11
# route: xy2 + progress1
ROUTE_IN_DIM = 3
# ego: speed1 + acc2 + yaw_sin_cos2 + size3 + type1 + valid1 + t0 + dt
EGO_IN_DIM = 12

ACCEL_CLIP = 6.0
KAPPA_CLIP = 0.3
MAX_SPEED = 50.0
