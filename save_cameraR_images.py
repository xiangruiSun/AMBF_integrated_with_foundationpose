import os
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraRImageSaver(Node):
    def __init__(self):
        super().__init__('cameraR_image_saver')

        self.bridge = CvBridge()
        self.save_dir = os.path.expanduser('~/FoundationPose/AMBF_data/cameraR_rgb')
        os.makedirs(self.save_dir, exist_ok=True)

        self.count = 0

        self.sub = self.create_subscription(
            Image,
            '/ambf/env/cameras/cameraR/ImageData',
            self.image_callback,
            10
        )

        self.get_logger().info(
            f'Saving cameraR images to: {self.save_dir}'
        )

    def image_callback(self, msg):
        try:
            # AMBF ImageData is usually RGB/BGR-like sensor_msgs/Image.
            # passthrough keeps the original encoding.
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # If image is RGB, convert to BGR before saving with OpenCV.
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            timestamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            filename = os.path.join(self.save_dir, f'cameraR_{self.count:06d}_{timestamp}.png')

            cv2.imwrite(filename, img)

            if self.count % 30 == 0:
                self.get_logger().info(f'Saved: {filename}')

            self.count += 1

        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')


def main():
    rclpy.init()
    node = CameraRImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
