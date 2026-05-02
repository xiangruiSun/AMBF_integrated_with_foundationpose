import os
import csv
import json
import time
import argparse

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

from ambf_msgs.msg import CameraState, RigidBodyState


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pose_msg_to_matrix(pose):
    x = pose.position.x
    y = pose.position.y
    z = pose.position.z

    qx = pose.orientation.x
    qy = pose.orientation.y
    qz = pose.orientation.z
    qw = pose.orientation.w

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    qx, qy, qz, qw = q

    R = np.array([
        [1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw,       2.0*qx*qz + 2.0*qy*qw],
        [2.0*qx*qy + 2.0*qz*qw,       1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw],
        [2.0*qx*qz - 2.0*qy*qw,       2.0*qy*qz + 2.0*qx*qw,       1.0 - 2.0*qx*qx - 2.0*qy*qy],
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def pose_msg_to_dict(pose):
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation_xyzw": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def pointcloud2_to_depth(msg, height=480, width=640, positive_mode="auto"):
    points = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=False,
    )

    depth_flat = np.asarray(points["z"], dtype=np.float32)

    if msg.height == 1 and msg.width == height * width:
        H, W = height, width
    else:
        H, W = int(msg.height), int(msg.width)

    if depth_flat.size != H * W:
        raise RuntimeError(
            f"Unexpected PointCloud2 size: {depth_flat.size}, expected {H * W}. "
            f"msg.height={msg.height}, msg.width={msg.width}"
        )

    depth = depth_flat.reshape(H, W).astype(np.float32)

    if positive_mode == "negative_z":
        depth = -depth
    elif positive_mode == "abs":
        depth = np.abs(depth)
    elif positive_mode == "auto":
        valid = np.isfinite(depth)
        if valid.any() and np.nanmedian(depth[valid]) < 0:
            depth = -depth

    depth[~np.isfinite(depth)] = 0.0
    depth[depth < 0.0] = 0.0
    return depth


def save_depth_png(depth, path):
    valid = np.isfinite(depth) & (depth > 0)

    if valid.any():
        d_min = np.min(depth[valid])
        d_max = np.max(depth[valid])
        depth_vis = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_vis = np.clip(depth_vis * 255.0, 0, 255).astype(np.uint8)
    else:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)

    cv2.imwrite(path, depth_vis)


def make_needle_mask_from_rgb(img_bgr, s_max=100, v_min=140, min_area=10):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, v_min], dtype=np.uint8)
    upper = np.array([179, s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == label] = 255

    return clean


class StaticAMBFDataCollector(Node):
    def __init__(self, args):
        super().__init__("static_ambf_data_100_collector")

        self.args = args
        self.bridge = CvBridge()

        self.base_dir = os.path.expanduser(args.out_dir)
        self.rgb_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.mask_dir = os.path.join(self.base_dir, "masks")
        self.needle_dir = os.path.join(self.base_dir, "needle")
        self.camera_dir = os.path.join(self.base_dir, "cameraR")

        for d in [self.rgb_dir, self.depth_dir, self.mask_dir, self.needle_dir, self.camera_dir]:
            os.makedirs(d, exist_ok=True)

        self.latest_rgb = None
        self.latest_depth = None
        self.latest_needle = None
        self.latest_camera = None

        self.last_saved_rgb_stamp = None
        self.count = 0

        self.rgb_sub = self.create_subscription(
            Image,
            "/ambf/env/cameras/cameraR/ImageData",
            self.rgb_cb,
            10,
        )

        self.depth_sub = self.create_subscription(
            PointCloud2,
            "/ambf/env/cameras/cameraR/DepthData",
            self.depth_cb,
            10,
        )

        self.needle_sub = self.create_subscription(
            RigidBodyState,
            "/ambf/env/phantom/Needle/State",
            self.needle_cb,
            10,
        )

        self.camera_sub = self.create_subscription(
            CameraState,
            "/ambf/env/cameras/cameraR/State",
            self.camera_cb,
            10,
        )

        self.metadata_path = os.path.join(self.base_dir, "metadata.csv")
        self.metadata_file = open(self.metadata_path, "w", newline="")
        self.metadata_writer = csv.writer(self.metadata_file)

        self.metadata_writer.writerow([
            "frame_id",
            "rgb_stamp_ns",
            "depth_stamp_ns",
            "needle_stamp_ns",
            "camera_stamp_ns",
            "rgb_file",
            "depth_npy",
            "depth_png",
            "mask_file",
            "needle_pose_txt",
            "camera_pose_txt",
        ])

        self.get_logger().info(f"Static camera collection. Saving to: {self.base_dir}")

    def rgb_cb(self, msg):
        self.latest_rgb = msg

    def depth_cb(self, msg):
        self.latest_depth = msg

    def needle_cb(self, msg):
        self.latest_needle = msg

    def camera_cb(self, msg):
        self.latest_camera = msg

    def ready(self):
        return (
            self.latest_rgb is not None and
            self.latest_depth is not None and
            self.latest_needle is not None and
            self.latest_camera is not None
        )

    def wait_until_ready(self):
        self.get_logger().info("Waiting for RGB, depth, needle state, and camera state...")

        while rclpy.ok() and not self.ready():
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("All required topics received. Camera will NOT be moved.")

    def save_current_sample(self):
        rgb_msg = self.latest_rgb
        depth_msg = self.latest_depth
        needle_msg = self.latest_needle
        camera_msg = self.latest_camera

        rgb_stamp = stamp_to_ns(rgb_msg.header.stamp)

        if self.last_saved_rgb_stamp == rgb_stamp:
            return False

        frame_id = f"frame_{self.count:06d}"

        img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")

        if img.ndim == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

        depth = pointcloud2_to_depth(
            depth_msg,
            height=self.args.height,
            width=self.args.width,
            positive_mode=self.args.depth_positive,
        )

        rgb_file = os.path.join(self.rgb_dir, f"{frame_id}.png")
        depth_npy = os.path.join(self.depth_dir, f"{frame_id}.npy")
        depth_png = os.path.join(self.depth_dir, f"{frame_id}.png")
        mask_file = os.path.join(self.mask_dir, f"{frame_id}.png")
        needle_txt = os.path.join(self.needle_dir, f"{frame_id}.txt")
        needle_json = os.path.join(self.needle_dir, f"{frame_id}.json")
        camera_txt = os.path.join(self.camera_dir, f"{frame_id}.txt")
        camera_json = os.path.join(self.camera_dir, f"{frame_id}.json")

        cv2.imwrite(rgb_file, img_bgr)

        np.save(depth_npy, depth)
        save_depth_png(depth, depth_png)

        mask = make_needle_mask_from_rgb(
            img_bgr,
            s_max=self.args.mask_s_max,
            v_min=self.args.mask_v_min,
            min_area=self.args.mask_min_area,
        )
        cv2.imwrite(mask_file, mask)

        T_needle = pose_msg_to_matrix(needle_msg.pose)
        T_camera = pose_msg_to_matrix(camera_msg.pose)

        np.savetxt(needle_txt, T_needle)
        np.savetxt(camera_txt, T_camera)

        with open(needle_json, "w") as f:
            json.dump({
                "frame_id": frame_id,
                "stamp_ns": stamp_to_ns(needle_msg.header.stamp) if hasattr(needle_msg, "header") else None,
                "pose": pose_msg_to_dict(needle_msg.pose),
            }, f, indent=2)

        with open(camera_json, "w") as f:
            json.dump({
                "frame_id": frame_id,
                "stamp_ns": stamp_to_ns(camera_msg.header.stamp) if hasattr(camera_msg, "header") else None,
                "pose": pose_msg_to_dict(camera_msg.pose),
            }, f, indent=2)

        self.metadata_writer.writerow([
            frame_id,
            rgb_stamp,
            stamp_to_ns(depth_msg.header.stamp),
            stamp_to_ns(needle_msg.header.stamp) if hasattr(needle_msg, "header") else "",
            stamp_to_ns(camera_msg.header.stamp) if hasattr(camera_msg, "header") else "",
            rgb_file,
            depth_npy,
            depth_png,
            mask_file,
            needle_txt,
            camera_txt,
        ])
        self.metadata_file.flush()

        self.last_saved_rgb_stamp = rgb_stamp
        self.count += 1

        self.get_logger().info(f"Saved {frame_id} ({self.count}/{self.args.num_frames})")
        return True

    def collect(self):
        self.wait_until_ready()

        while rclpy.ok() and self.count < self.args.num_frames:
            rclpy.spin_once(self, timeout_sec=0.1)

            if not self.ready():
                continue

            self.save_current_sample()

            if self.args.sleep_sec > 0:
                time.sleep(self.args.sleep_sec)

        self.get_logger().info("Finished static-camera recording.")
        self.metadata_file.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="~/FoundationPose/AMBF_data_100")
    parser.add_argument("--num_frames", type=int, default=100)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)

    parser.add_argument(
        "--depth_positive",
        type=str,
        default="auto",
        choices=["auto", "negative_z", "abs", "none"],
    )

    parser.add_argument("--sleep_sec", type=float, default=0.0)

    parser.add_argument("--mask_s_max", type=int, default=100)
    parser.add_argument("--mask_v_min", type=int, default=140)
    parser.add_argument("--mask_min_area", type=int, default=10)

    args = parser.parse_args()

    rclpy.init()
    node = StaticAMBFDataCollector(args)

    try:
        node.collect()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
