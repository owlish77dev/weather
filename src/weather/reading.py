from pathlib import Path
from typing import Sequence

from rosbags.highlevel import AnyReader
import json
import pandas as pd


snow_path = Path(r"/media/andrea/Andrea's FESTPLATTE/Wetter_FZD/Snow")


def read_snow_data_definition():
    defs = []
    for dir_p, dir_name, files in snow_path.walk():
        # for file_name in files:
        #     fp = snow_path.joinpath(file_name)
        bag_files = [dir_p.joinpath(f) for f in files]
        with AnyReader([bag_files[0]]) as ar:
            for c in ar.connections:
                d = {
                    "topic": c.topic,
                    "msg_type": c.msgtype,
                    "msg_def": c.msgdef.data,
                    "id": c.id
                }
                defs.append(d)
    with open(Path.cwd().joinpath("topics.json"), "w+") as fh:
        json.dump(defs, fh, indent=4)
    # return defs


def read_radar_detections(bag_files: Sequence[Path]):
    topic = "/as_tx/radar_detections"
    detections = []
    # for dir_p, dir_name, files in snow_path.walk():
    #     bag_files = [dir_p.joinpath(f) for f in files]
    with AnyReader(bag_files) as ar:
        conn = [c for c in ar.connections if c.topic == topic]
        for c, ts, raw_data in ar.messages(conn):
            msg = ar.deserialize(raw_data, c.msgtype)
            for det in msg.detections:
                detections.append(
                    (
                        msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                        det.detection_id,
                        det.amplitude,
                        det.position.x,
                        det.position.y,
                        det.position.z,
                        det.velocity.x,
                        det.velocity.y,
                        det.velocity.z,
                    )
                )
    cols = ["ts", "idx", "rcs", "dist_x", "dist_y", "dist_z", "vel_x", "vel_y", "vel_z"]
    det_df = pd.DataFrame.from_records(data=detections, columns=cols)
    det_df.set_index(keys=["ts", "idx"], inplace=True)
    det_df = det_df.loc[det_df.rcs.abs() >= 1e-3]
    # amplitude (rcs) seems to have 0 as default value. Old detections which still linger on the interface
    # seem to be indicated by amplitude = 0.
    yield det_df


def read_data():
    for dir_p, dir_name, files in snow_path.walk():
        bag_files = [dir_p.joinpath(f) for f in files if dir_p.joinpath(f).suffix == ".bag"]
        bag_files = [bag_files[0]]  # can be uncommented if you have time
        det_df = read_radar_detections(bag_files)
        pass


if __name__ == "__main__":
    read_data()
