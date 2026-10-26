from __future__ import annotations

import struct

from womd_il.data.parse_scenario import parse_tfrecord_bytes
from womd_il.data.tfrecord import iter_tfrecord_file
from womd_il.protos import scenario_pb2


def _wrap_tfrecord(payload: bytes) -> bytes:
    return struct.pack("<Q", len(payload)) + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00"


def test_tfrecord_roundtrip(tmp_path):
    proto = scenario_pb2.Scenario()
    proto.scenario_id = "unit_test"
    proto.sdc_track_index = 0
    proto.current_time_index = 10
    track = proto.tracks.add()
    track.id = 7
    track.object_type = 1
    for i in range(91):
        st = track.states.add()
        st.center_x = float(i)
        st.center_y = 1.0
        st.heading = 0.1
        st.velocity_x = 10.0
        st.valid = True
        st.length = 4.0
        st.width = 2.0
        st.height = 1.5
    lane = proto.map_features.add()
    lane.id = 1
    for x in range(5):
        pt = lane.lane.polyline.add()
        pt.x = float(x)
        pt.y = 0.0
    path = tmp_path / "one.tfrecord"
    path.write_bytes(_wrap_tfrecord(proto.SerializeToString()))
    raw = next(iter_tfrecord_file(path))
    sample = parse_tfrecord_bytes(raw)
    assert sample["scenario_id"] == "unit_test"
    assert sample["valid"][0].all()
    assert sample["xyz"][0, 10, 0] == 10.0
    assert sample["map_valid"][0].any()
