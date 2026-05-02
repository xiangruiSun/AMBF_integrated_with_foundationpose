import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class CameraRPointCloudDepthSaver(Node):
    def __init__(self):
        super().__init__('cameraR_pointcloud_depth_saver')

        self.save_dir = os.path.expanduser('~/FoundationPose/AMBF_data/depth')
        os.makedirs(self.save_dir, exist_ok=True)

        self.count = 0

        self.sub = self.create_subscription(
            PointCloud2,
            '/ambf/env/cameras/cameraR/DepthData',
            self.callback,
            10
        )

        self.get_logger().info(f'Saving cameraR PointCloud2 depth to: {self.save_dir}')

    def callback(self, msg):
        try:
            # In ROS 2 Jazzy, read_points returns a structured NumPy array
            # with named fields such as x, y, z.
            points = point_cloud2.read_points(
                msg,
                field_names=('x', 'y', 'z'),
                skip_nans=False
            )

            # Extract the z field directly as depth.
            depth_flat = np.asarray(points['z'], dtype=np.float32)

            # AMBF gives height=1, width=307200, which is 640*480.
            if msg.height == 1 and msg.width == 307200:
                H, W = 480, 640
            else:
                H, W = msg.height, msg.width

            if depth_flat.size != H * W:
                self.get_logger().error(
                    f'Unexpected depth size: {depth_flat.size}, expected {H * W}. '
                    f'msg.height={msg.height}, msg.width={msg.width}'
                )
                return

            depth = depth_flat.reshape(H, W)

            timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

            # Save raw depth as .npy. This is the useful file for later processing.
            npy_path = os.path.join(
                self.save_dir,
                f'cameraR_depth_{self.count:06d}_{timestamp}.npy'
            )
            np.save(npy_path, depth)

            # Save a visualized PNG only for checking.
            depth_vis = depth.copy()
            valid = np.isfinite(depth_vis)

            if valid.any():
                d_min = np.min(depth_vis[valid])
                d_max = np.max(depth_vis[valid])
                depth_vis = (depth_vis - d_min) / (d_max - d_min + 1e-8)
                depth_vis = (depth_vis * 255).astype(np.uint8)
            else:
                depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)

            png_path = os.path.join(
                self.save_dir,
                f'cameraR_depth_vis_{self.count:06d}_{timestamp}.png'
            )
            cv2.imwrite(png_path, depth_vis)

            if self.count % 10 == 0:
                self.get_logger().info(
                    f'Saved raw depth: {npy_path}, vis: {png_path}, shape: {depth.shape}, '
                    f'min={np.nanmin(depth):.6f}, max={np.nanmax(depth):.6f}'
                )

            self.count += 1

        except Exception as e:
            self.get_logger().error(f'Failed to convert PointCloud2 to depth: {e}')


def main():
    rclpy.init()
    node = CameraRPointCloudDepthSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
