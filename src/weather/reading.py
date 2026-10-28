"""Reading weather data from FZD. Scenarios are static only."""

from pathlib import Path
from typing import Sequence
from abc import ABC, abstractmethod

import numpy as np
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


class IWeatherReader(ABC):

    @abstractmethod
    def read_all_data(self, files: Sequence[Path] | list[Path] | Path):
        pass


class FzdDataReader(IWeatherReader):

    def __init__(self, files: Sequence[Path] | list[Path] | Path):
        super().__init__()
        if isinstance(files, Path) and files.is_dir():
            self._files = []
            for dir_p, dir_name, files in files.walk():
                bag_files = [dir_p.joinpath(f) for f in files if dir_p.joinpath(f).suffix == ".bag"]
                bag_files = [bag_files[0]]  # can be commented out if you have time
                self._files.extend(bag_files)
        else:
            self._files = files

    def _read_radar(self):
        """Read radar detections and objects."""
        topics = ["/as_tx/radar_detections", "/as_tx/objects"]
        detections = []
        objects = []
        with AnyReader(self._files) as ar:
            conn = [c for c in ar.connections if c.topic in topics]
            for c, ts, raw_data in ar.messages(conn):
                msg = ar.deserialize(raw_data, c.msgtype)
                if hasattr(msg, "detections"):
                    for det in msg.detections:
                        detections.append(
                            (
                                ts,
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
                                ts,
                                msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                                i,
                                obj.pose.pose.position.x,
                                obj.pose.pose.position.y
                            )
                        )
        det_cols = ["bag_ts", "ts", "idx", "amplitude", "dist_x", "dist_y"]
        obj_cols = ["bag_ts", "ts", "idx", "dist_x", "dist_y"]
        det_df = pd.DataFrame.from_records(data=detections, columns=det_cols)
        det_df.set_index(keys=["ts", "idx"], inplace=True)
        det_df = det_df.loc[det_df.amplitude.abs() >= 1e-3]
        # amplitude seems to have 0 as default value. Old detections which still linger on the interface
        # seem to be indicated by amplitude = 0.
        obj_df = pd.DataFrame.from_records(data=objects, columns=obj_cols)
        obj_df.set_index(keys=["ts", "idx"], inplace=True)
        yield det_df, obj_df


    def _read_transformations(self) -> pd.DataFrame:
        """Read the static position and rotations of the three lidar sensors and mapScala."""
        topic = "/tf_static"
        tf = []
        if len(self._files) == 0:
            return pd.DataFrame()
        with AnyReader(self._files) as ar:
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


    def _read_delphi_lidar(self):
        """Read detections and objects from Delohi lidar."""
        topics = ["/delphi_esr_object_visu", "/delphi_esr_detection_visu"]
        points = []
        objects = []
        with AnyReader(paths=self._files) as ar:
            conn = [c for c in ar.connections if c.topic in topics]
            for c, ts, raw_data in ar.messages(connections=conn):
                msg = ar.deserialize(raw_data, c.msgtype)
                if c.msgtype == "sensor_msgs/msg/PointCloud2":
                    # h = msg.height is always equal to 1
                    enc = "big" if msg.is_bigendian else "little"
                    nof_pts = msg.width
                    d = msg.data
                    nof_bytes_per_point = msg.point_step  # size of point
                    len_data_arr = msg.row_step
                    if nof_pts == 0:
                        points.append(
                            (
                                ts,
                                msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                                -1,
                                np.nan, np.nan, np.nan, np.nan  # show missing data
                            )
                        )
                    else:
                        for p in range(nof_pts):
                            start_byte = p * nof_bytes_per_point
                            x = FzdDataReader._decode_uint8_to_float32(
                                int_arr=d[start_byte: start_byte + 4], encoding=enc
                            )
                            y = FzdDataReader._decode_uint8_to_float32(
                                int_arr=d[start_byte + 4: start_byte + 8], encoding=enc
                            )
                            z = FzdDataReader._decode_uint8_to_float32(
                                int_arr=d[start_byte + 8: start_byte + 12], encoding=enc
                            )
                            intensity = FzdDataReader._decode_uint8_to_float32(
                                int_arr=d[start_byte + 12: start_byte + 16], encoding=enc
                            )
                            points.append(
                                (
                                    ts,
                                    msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec,
                                    p, x, y, z, intensity
                                )
                            )
                elif c.msgtype == "visualization_msgs/msg/MarkerArray":
                    for i in range(len(msg.markers)):
                        mm = msg.markers[i]
                        objects.append(
                            (
                                ts,
                                mm.header.stamp.sec * 1e9 + mm.header.stamp.nanosec,
                                mm.id,
                                mm.lifetime.sec * 1e9 + mm.lifetime.nanosec,
                                mm.pose.position.x,
                                mm.pose.position.y,
                                mm.pose.position.z,
                                mm.pose.orientation.w,
                                mm.pose.orientation.x,
                                mm.pose.orientation.y,
                                mm.pose.orientation.z
                            )
                        )
                else:
                    raise ValueError(f"Unexpected message type '{c.msgtype}' for topic '{c.topic}'.")
        pcl_col = ["bag_ts", "ts", "id", "x", "y", "z", "intensity"]
        obj_col = ["bag_ts", "ts", "id", "lifetime", "pos_x", "pos_y", "pos_z", "w", "x", "y", "z"]
        pcl_df = pd.DataFrame.from_records(data=points, columns=pcl_col)
        obj_df = pd.DataFrame.from_records(data=objects, columns=obj_col)
        yield pcl_df, obj_df

    @staticmethod
    def _decode_uint8_to_float32(int_arr: np.ndarray, encoding: str) -> float:
        if len(int_arr) != 4:
            raise ValueError("Expected array of length 4.")
        b_arr = np.unpackbits(int_arr, bitorder=encoding).astype(np.int32)
        s = -1. if b_arr[0] else 1.
        exp = (128 * b_arr[1] + 64 * b_arr[2] + 32 * b_arr[3]
               + 16 * b_arr[4] + 8 * b_arr[5] + 4 * b_arr[6] +
               2 * b_arr[7] + b_arr[8]) - 127
        frac: np.float32 = 1.
        for i in range(1, 24, 1):
            frac += np.float_power(2., -i) * b_arr[i + 8]
        f_val = s * np.float_power(2., exp) * frac
        return f_val


    def _read_weather(self):
        """Read data from weather sensor."""
        topic = "/weather_sensor_data"
        wd = []
        with AnyReader(self._files) as ar:
            conn = [c for c in ar.connections if c.topic == topic]
            for c, ts, raw_data in ar.messages(connections=conn):
                msg = ar.deserialize(raw_data, c.msgtype)
                wd.append(
                    (
                        ts,
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
        cols = ["bag_ts", "ts", "ambient_temperature", "brightness", "pressure", "rain_daily_sum",
                "rain_event", "rain_intensity", "relative_humidity", "wind_direction", "wind_speed", "synop"]
        weather_df = pd.DataFrame.from_records(data=wd, columns=cols)
        weather_df.set_index(keys="ts", inplace=True)
        yield weather_df


    def read_all_data(self):
        # for dir_p, dir_name, files in snow_path.walk():
        #     bag_files = [dir_p.joinpath(f) for f in files if dir_p.joinpath(f).suffix == ".bag"]
        #     bag_files = [bag_files[0]]  # can be commented out if you have time
        radar_gen = self._read_radar()
        transform_df = self._read_transformations()
        print(f"n_rows: {transform_df.shape[0]}")
        weather_gen = self._read_weather()
        lidar_gen = self._read_delphi_lidar()
        for det_df, obj_df in lidar_gen:
            print(f"Length of detection df: {det_df.shape[0]}")
        #     det_ts = det_df.index.unique(level=0).values
        #     obj_ts = obj_df.index.unique(level=0).values
        #     print(f"Timestamps are equal: {set(det_ts) == set(obj_ts)}")  # False
        #     print(f"Timestamps are disjoint: {set(det_ts).isdisjoint(set(obj_ts))}")  # True
        pass


if __name__ == "__main__":
    pass
