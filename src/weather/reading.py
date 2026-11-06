"""Reading weather data from FZD. Scenarios are static only."""

from pathlib import Path
from typing import Sequence, Collection
from abc import ABC, abstractmethod

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.interfaces import Connection
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
    def read_all_data(self):
        pass


class FzdDataReader(IWeatherReader):

    def __init__(self, files: Sequence[Path] | list[Path] | Path):
        super().__init__()
        self._data = {}
        if isinstance(files, Path) and files.is_dir():
            self._files = []
            for dir_p, dir_name, files in files.walk():
                bag_files = [dir_p.joinpath(f) for f in files if dir_p.joinpath(f).suffix == ".bag"]
                # bag_files = [bag_files[0]]  # can be commented out if you have time
                self._files.extend(bag_files)
        elif isinstance(files, Path) and files.is_file():
            self._files = [files]
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

    @staticmethod
    def read_point_cloud(ar: AnyReader, connections: Collection[Connection]):
        points = {
            "bag_ts": [],
            "ts": [],
            "id": [],
            "x": [],
            "y": [],
            "z": [],
            "intensity": []
        }
        pcl_conn = [c for c in connections if c.msgtype == "sensor_msgs/msg/PointCloud2"]
        i = 0
        for c, ts, raw_data in ar.messages(connections=pcl_conn):
            msg = ar.deserialize(raw_data, c.msgtype)
            n_fields = len(msg.fields)
            # if i == 0:
            #     for field in msg.fields:
            #         print(
            #             field.name, field.offset, field.datatype, field.count
            #         )
            #     i += 1
            enc = "big" if msg.is_bigendian else "little"
            nof_pts = msg.width
            d = msg.data
            nof_bytes_per_point = msg.point_step  # size of point
            # len_data_arr = msg.row_step
            if nof_pts == 0:
                points["bag_ts"].append(ts)
                points["ts"].append(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
                points["id"].append(-1)
                points["x"].append(np.nan)
                points["y"].append(np.nan)
                points["z"].append(np.nan)
                points["intensity"].append(np.nan)
            else:
                lidar_points = FzdDataReader.decode_lidar_pointcloud(
                    data=d, big_end=msg.is_bigendian, n_fields=n_fields)
                points["x"].extend(lidar_points[:, 0])
                points["y"].extend(lidar_points[:, 1])
                points["z"].extend(lidar_points[:, 2])
                points["intensity"].extend(lidar_points[:, 3])
                points["bag_ts"].extend([ts] * nof_pts)
                points["ts"].extend(
                    [msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec] * nof_pts
                )
                points["id"].extend([i for i in range(nof_pts)])
        pcl_df = pd.DataFrame(data=points)
        pcl_df.set_index(keys=["ts", "id"], inplace=True)
        return pcl_df
    # BF lidar has additional field 'return_id' of type uint32

    @staticmethod
    def read_marker_array(ar: AnyReader, connections: Collection[Connection]):
        objects = []
        obj_conn = [c for c in connections if c.msgtype == "visualization_msgs/msg/MarkerArray"]
        for c, ts, raw_data in ar.messages(connections=obj_conn):
            msg = ar.deserialize(raw_data, c.msgtype)
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
        obj_col = ["bag_ts", "ts", "id", "lifetime", "pos_x", "pos_y", "pos_z", "w", "x", "y", "z"]
        obj_df = pd.DataFrame.from_records(data=objects, columns=obj_col)
        obj_df.set_index(keys=["ts", "id"], inplace=True)
        return obj_df

    def _read_bf_lidar(self):
        """Read BF lidar point cloud."""
        topic = "/bf_lidar/points_raw"  # apparently only point cloud, no objects
        with AnyReader(paths=self._files) as ar:
            conn = [c for c in ar.connections if c.topic == topic]
            bf_lidar_pcl = FzdDataReader.read_point_cloud(ar=ar, connections=conn)
        yield bf_lidar_pcl

    def _read_delphi_lidar(self):
        """Read detections and objects from Delphi lidar."""
        topics = ["/delphi_esr_object_visu", "/delphi_esr_detection_visu"]
        with AnyReader(paths=self._files) as ar:
            conn = [c for c in ar.connections if c.topic in topics]
            pcl_df = FzdDataReader.read_point_cloud(ar, conn)
            obj_df = FzdDataReader.read_marker_array(ar, conn)
        yield pcl_df, obj_df

    @staticmethod
    def decode_lidar_pointcloud(data: np.ndarray, big_end: bool, n_fields: int) -> np.ndarray:
        endian = ">" if big_end else "<"
        dt = np.dtype(endian + "f4")
        out_data = np.frombuffer(data, dtype=dt)
        points = out_data.reshape(-1, n_fields)
        return points

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
        """Read all data from sensors."""
        radar_gen = self._read_radar()
        # loops will just be processed once
        for det_df, obj_df in radar_gen:
            self._data["det_radar"] = det_df
            self._data["obj_radar"] = obj_df
        self._data["transformation"] = self._read_transformations()
        weather_gen = self._read_weather()
        for weather_df in weather_gen:
            self._data["weather"] = weather_df
        lidar_gen = self._read_delphi_lidar()
        bf_lidar = self._read_bf_lidar()
        for pcl_df, obj_df in lidar_gen:
            self._data["pcl_delphi_lidar"] = pcl_df
            self._data["obj_delphi_lidar"] = obj_df
        for pcl_df in bf_lidar:
            self._data["pcl_bf_lidar"] = pcl_df
        pass

    @property
    def weather(self) -> pd.DataFrame:
        if "weather" in self._data.keys():
            return self._data["weather"]
        else:
            return pd.DataFrame()

    @property
    def transformation(self) -> pd.DataFrame:
        return self._data["transformation"]

    @property
    def radar_detections(self) -> pd.DataFrame:
        if "det_radar" in self._data.keys():
            return self._data["det_radar"]
        else:
            return pd.DataFrame()

    @property
    def radar_objects(self) -> pd.DataFrame:
        if "obj_radar" in self._data.keys():
            return self._data["obj_radar"]
        else:
            return pd.DataFrame()

    @property
    def point_cloud_bf_lidar(self) -> pd.DataFrame:
        if "pcl_bf_lidar" in self._data.keys():
            return self._data["pcl_bf_lidar"]
        else:
            return pd.DataFrame()

    @property
    def point_cloud_delphi_lidar(self) -> pd.DataFrame:
        if "pcl_delphi_lidar" in self._data.keys():
            return self._data["pcl_delphi_lidar"]
        else:
            return pd.DataFrame()

    @property
    def objects_delphi_lidar(self) -> pd.DataFrame:
        if "obj_delphi_lidar" in self._data.keys():
            return self._data["obj_delphi_lidar"]
        else:
            return pd.DataFrame()


if __name__ == "__main__":
    pass
