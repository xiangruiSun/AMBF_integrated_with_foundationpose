import os
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraRDepthSaver(Node):
    def __init__(self):
        super().__init__('cameraR_depth_saver')

        self.bridge = CvBridge()
        self.save_dir = os.path.expanduser('~/FoundationPose/AMBF_data/depth')
        os.makedirs(self.save_dir, exist_ok=True)

        self.count = 0

        self.sub = self.create_subscription(
            Image,
            '/ambf/env/cameras/cameraR/DepthData',
            self.depth_callback,
            10
        )

        self.get_logger().info(f'Saving cameraR depth images to: {self.save_dir}')

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

            # Save raw depth as .npy to preserve exact floating-point values
            npy_path = os.path.join(
                self.save_dir,
                f'cameraR_depth_{self.count:06d}_{timestamp}.npy'
            )
            np.save(npy_path, depth)

            # Also save a visualized PNG for quick checking
            depth_vis = depth.copy()

            if np.issubdtype(depth_vis.dtype, np.floating):
                valid = np.isfinite(depth_vis)
                if valid.any():
                    d_min = np.min(depth_vis[valid])
                    d_max = np.max(depth_vis[valid])
                    depth_vis = (depth_vis - d_min) / (d_max - d_min + 1e-8)
                    depth_vis = (depth_vis * 255).astype(np.uint8)
                else:
                    depth_vis = np.zeros_like(depth_vis, dtype=np.uint8)
            else:
                depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                depth_vis = depth_vis.astype(np.uint8)

            png_path = os.path.join(
                self.save_dir,
                f'cameraR_depth_{self.count:06d}_{timestamp}.png'
            )
            cv2.imwrite(png_path, depth_vis)

            if self.count % 30 == 0:
                self.get_logger().info(f'Saved: {npy_path} and {png_path}')

            self.count += 1

        except Exception as e:
            self.get_logger().error(f'Failed to save depth image: {e}')


def main():
    rclpy.init()
    node = CameraRDepthSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
