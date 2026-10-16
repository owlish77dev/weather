"""Reading weather data from FZD. Scenarios are static only."""

from pathlib import Path
from typing import Sequence

from rosbags.highlevel import AnyReader
import json
import pandas as pd


snow_path = Path(r"/media/andrea/Andrea's FESTPLATTE/Wetter_FZD/Snow")


def read_snow_data_definition():
    """Read data definitions from rosbags."""
    defs = []
    for dir_p, dir_name, files in snow_path.walk():
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


def read_radar(bag_files: Sequence[Path]):
    """Read radar detections and objects."""
    topics = ["/as_tx/radar_detections", "/as_tx/objects"]
    detections = []
    objects = []
    with AnyReader(bag_files) as ar:
        conn = [c for c in ar.connections if c.topic in topics]
        for c, ts, raw_data in ar.messages(conn):
            msg = ar.deserialize(raw_data, c.msgtype)
            if hasattr(msg, "detections"):
                for det in msg.detections:
                    detections.append(
                        (
                            msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                            det.detection_id,
                            det.amplitude,
                            det.position.x,
                            det.position.y,
                        )
                    )
            elif hasattr(msg, "objects"):
                for i, obj in enumerate(msg.objects):
                    objects.append(
                        (
                            msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                            i,
                            obj.pose.pose.position.x,
                            obj.pose.pose.position.y
                        )
                    )
    det_cols = ["ts", "idx", "amplitude", "dist_x", "dist_y"]
    obj_cols = ["ts", "idx", "dist_x", "dist_y"]
    det_df = pd.DataFrame.from_records(data=detections, columns=det_cols)
    det_df.set_index(keys=["ts", "idx"], inplace=True)
    det_df = det_df.loc[det_df.amplitude.abs() >= 1e-3]
    # amplitude seems to have 0 as default value. Old detections which still linger on the interface
    # seem to be indicated by amplitude = 0.
    obj_df = pd.DataFrame.from_records(data=objects, columns=obj_cols)
    obj_df.set_index(keys=["ts", "idx"], inplace=True)
    yield det_df, obj_df


def read_transformations(bag_files: Sequence[Path]):
    """Read the static position and rotations of the three lidar sensors and mapScala."""
    topic = "/tf_static"
    tf = []
    if len(bag_files) == 0:
        return
    elif len(bag_files) > 1:
        bag_files = [bag_files[0]]
    with AnyReader(bag_files) as ar:
        conn = [c for c in ar.connections if c.topic == topic]
        for c, ts, raw_data in ar.messages(conn):
            msg = ar.deserialize(rawdata=raw_data, typ=c.msgtype)
            for t in msg.transforms:
                tf.append(
                    (
                        t.child_frame_id,
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w,
                    )
                )
    tf_cols = ["frame_id", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "rot_w"]
    transform_df = pd.DataFrame.from_records(data=tf, columns=tf_cols)
    transform_df.set_index(keys="frame_id", inplace=True)
    return transform_df


def read_weather(bag_files: Sequence[Path]):
    """Read data from weather sensor."""
    topic = "/weather_sensor_data"
    wd = []
    with AnyReader(bag_files) as ar:
        conn = [c for c in ar.connections if c.topic == topic]
        for c, ts, raw_data in ar.messages(connections=conn):
            msg = ar.deserialize(raw_data, c.msgtype)
            wd.append(
                (
                    msg.stamp.sec * 1e9 + msg.stamp.nanosec,
                    msg.ambient_temperature,
                    msg.brightness,
                    msg.pressure,
                    msg.rain_daily_sum,
                    msg.rain_event,
                    msg.rain_intensity,
                    msg.relative_humidity,
                    msg.wind_direction,
                    msg.wind_speed,
                    msg.synop
                )
            )
    cols = ["ts", "ambient_temperature", "brightness", "pressure", "rain_daily_sum",
            "rain_event", "rain_intensity", "relative_humidity", "wind_direction", "wind_speed", "synop"]
    weather_df = pd.DataFrame.from_records(data=wd, columns=cols)
    weather_df.set_index(keys="ts", inplace=True)
    yield weather_df


def read_data():
    for dir_p, dir_name, files in snow_path.walk():
        bag_files = [dir_p.joinpath(f) for f in files if dir_p.joinpath(f).suffix == ".bag"]
        bag_files = [bag_files[0]]  # can be commented out if you have time
        radar_gen = read_radar(bag_files)
        transform_df = read_transformations(bag_files)
        weather_gen = read_weather(bag_files)
        # for det_df, obj_df in radar_gen:
        #     print(f"Length of detection df: {det_df.shape[0]}")
        #     det_ts = det_df.index.unique(level=0).values
        #     obj_ts = obj_df.index.unique(level=0).values
        #     print(f"Timestamps are equal: {set(det_ts) == set(obj_ts)}")  # False
        #     print(f"Timestamps are disjoint: {set(det_ts).isdisjoint(set(obj_ts))}")  # True
        pass


if __name__ == "__main__":
    read_data()
