import time
import math
import rclpy
from rclpy.node import Node

from ambf_msgs.msg import CameraCmd, CameraState


class SlowCameraRMover(Node):
    def __init__(self):
        super().__init__('slow_cameraR_mover')

        self.state_topic = '/ambf/env/cameras/cameraR/State'
        self.cmd_topic = '/ambf/env/cameras/cameraR/Command'

        self.current_state = None

        self.sub = self.create_subscription(
            CameraState,
            self.state_topic,
            self.state_cb,
            10
        )

        self.pub = self.create_publisher(
            CameraCmd,
            self.cmd_topic,
            10
        )

        self.get_logger().info('Waiting for cameraR State...')

    def state_cb(self, msg):
        self.current_state = msg

    def wait_for_state(self):
        while rclpy.ok() and self.current_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Received cameraR State.')

    def publish_pose(self, x, y, z, qx, qy, qz, qw):
        cmd = CameraCmd()

        # Important: enable position control so AMBF follows the commanded pose.
        if hasattr(cmd, 'enable_position_controller'):
            cmd.enable_position_controller = True

        cmd.pose.position.x = float(x)
        cmd.pose.position.y = float(y)
        cmd.pose.position.z = float(z)

        cmd.pose.orientation.x = float(qx)
        cmd.pose.orientation.y = float(qy)
        cmd.pose.orientation.z = float(qz)
        cmd.pose.orientation.w = float(qw)

        self.pub.publish(cmd)

    def move_slowly(self):
        self.wait_for_state()

        start_pose = self.current_state.pose

        x0 = start_pose.position.x
        y0 = start_pose.position.y
        z0 = start_pose.position.z

        qx = start_pose.orientation.x
        qy = start_pose.orientation.y
        qz = start_pose.orientation.z
        qw = start_pose.orientation.w

        self.get_logger().info(
            f'Start pose: x={x0:.4f}, y={y0:.4f}, z={z0:.4f}, '
            f'q=({qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f})'
        )

        # Move amount in meters.
        # Very small and slow motion for safety.
        dx = -0.02      # move 2 cm in x
        dy = 0.00
        dz = 0.00

        duration = 5.0       # seconds
        rate_hz = 30.0
        steps = int(duration * rate_hz)

        self.get_logger().info(
            f'Moving cameraR slowly by dx={dx}, dy={dy}, dz={dz} over {duration} seconds.'
        )

        for i in range(steps + 1):
            alpha = i / steps

            x = x0 + alpha * dx
            y = y0 + alpha * dy
            z = z0 + alpha * dz

            self.publish_pose(x, y, z, qx, qy, qz, qw)

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(1.0 / rate_hz)

        self.get_logger().info('Motion complete. Holding final pose.')

        # Hold final pose briefly.
        for _ in range(60):
            self.publish_pose(x0 + dx, y0 + dy, z0 + dz, qx, qy, qz, qw)
            time.sleep(1.0 / rate_hz)


def main():
    rclpy.init()
    node = SlowCameraRMover()

    try:
        node.move_slowly()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
