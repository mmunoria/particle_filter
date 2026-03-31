import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry, Path

import numpy as np
from math import cos, sin, pi


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_quaternion(angle: float):
    qz = np.sin(angle * 0.5)
    qw = np.cos(angle * 0.5)
    return (0.0, 0.0, qz, qw)


class ParticleFilter(Node):
    def __init__(self):
        super().__init__("particle_filter")

        # Robot parameters
        self.r = 0.033
        self.b = 0.160

        # Measurement noise covariance for z = [v_enc, w_enc, w_imu]
        self.R = np.diag([
            0.03**2,
            0.15**2,
            0.01**2
        ]).astype(float)

        # Particle filter settings
        self.M = 1000
        self.is_initialized = False

        # Timing
        self.last_time = None
        self.last_wheel_pos = None
        self.last_wheel_t = None

        # First-order lag constants
        self.alpha_v = 8.0
        self.alpha_w = 10.0

        # Measurements
        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0
        self.z = None

        # Control input u = [v_cmd, w_cmd]
        self.u = np.zeros(2, dtype=float)

        # Estimated state x = [px, py, theta, v, w]
        self.x_est = np.zeros(5, dtype=float)

        # Particle set: shape (M, 5)
        self.particles = None
        self.weights = None

        # Pose covariance estimate
        self.P = np.eye(5, dtype=float) * 1e-3

        # Path message
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        # Subscriptions
        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel", self.cmd_callback, 10)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/pf/odom", 10)
        self.path_pub = self.create_publisher(Path, "/pf/path", 10)

        # Timer
        self.create_timer(0.02, self.particle_filter)  # 50 Hz

    def joint_callback(self, msg: JointState):
        names = ["wheel_left_joint", "wheel_right_joint"]
        try:
            wl_i = msg.name.index(names[0])
            wr_i = msg.name.index(names[1])
        except ValueError:
            return

        t_now = self.get_clock().now().nanoseconds * 1e-9
        wl_pos = float(msg.position[wl_i])
        wr_pos = float(msg.position[wr_i])

        if self.last_wheel_pos is None:
            self.last_wheel_pos = (wl_pos, wr_pos)
            self.last_wheel_t = t_now
            return

        dt = t_now - self.last_wheel_t
        if dt <= 1e-6:
            return

        wl_prev, wr_prev = self.last_wheel_pos
        wl = (wl_pos - wl_prev) / dt
        wr = (wr_pos - wr_prev) / dt

        self.last_wheel_pos = (wl_pos, wr_pos)
        self.last_wheel_t = t_now

        self.v_enc = (self.r / 2.0) * (wl + wr)
        self.w_enc = (self.r / self.b) * (wr - wl)

        # Keep your sign convention if needed
        self.w_enc = -self.w_enc

        self.update_z_vector()

    def imu_callback(self, msg: Imu):
        self.w_imu = float(msg.angular_velocity.z)
        self.update_z_vector()

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([v_cmd, w_cmd], dtype=float)

    def update_z_vector(self):
        self.z = np.array([self.v_enc, self.w_enc, self.w_imu], dtype=float)

    def f_x(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """
        State x = [px, py, theta, v, w]
        Control u = [v_cmd, w_cmd]
        Returns 1D array shape (5,)
        """
        px, py, theta, v, w = x
        v_cmd, w_cmd = u

        px_new = px + v * dt * cos(theta)
        py_new = py + v * dt * sin(theta)
        theta_new = wrap_angle(theta + w * dt)
        v_new = v + self.alpha_v * dt * (v_cmd - v)
        w_new = w + self.alpha_w * dt * (w_cmd - w)

        return np.array([px_new, py_new, theta_new, v_new, w_new], dtype=float)

    def h(self, x: np.ndarray) -> np.ndarray:
        """
        Measurement model:
        z = [v_enc, w_enc, w_imu]
        h(x) = [v, w, w]
        """
        v = x[3]
        w = x[4]
        return np.array([v, w, w], dtype=float)

    def gaussian_likelihood(self, x: np.ndarray, z: np.ndarray) -> float:
        """
        p(z | x) for multivariate Gaussian noise.
        """
        y = z - self.h(x)
        y = y.reshape(-1, 1)

        R_inv = np.linalg.inv(self.R)
        det_R = np.linalg.det(self.R)
        n = z.shape[0]

        exponent = float(-0.5 * (y.T @ R_inv @ y))
        norm = 1.0 / np.sqrt(((2.0 * np.pi) ** n) * det_R)

        return float(norm * np.exp(exponent))

    def initialize_particles(self):
        """
        Local initialization around current estimate.
        If you truly do not know pose, replace px/py/theta sampling
        with broad map bounds.
        """
        px, py, theta, v, w = self.x_est

        particles = np.zeros((self.M, 5), dtype=float)
        particles[:, 0] = np.random.uniform(px - 1.0, px + 1.0, self.M)
        particles[:, 1] = np.random.uniform(py - 1.0, py + 1.0, self.M)
        particles[:, 2] = np.random.uniform(-pi, pi, self.M)
        particles[:, 3] = np.random.uniform(v - 0.5, v + 0.5, self.M)
        particles[:, 4] = np.random.uniform(w - 0.5, w + 0.5, self.M)

        weights = np.ones(self.M, dtype=float) / self.M
        return particles, weights

    def resample(self, particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
        weights = weights / np.sum(weights)
        indices = np.random.choice(np.arange(len(particles)), size=len(particles), p=weights)
        return particles[indices]

    def estimate_state(self):
        """
        Weighted mean estimate.
        """
        self.x_est = np.average(self.particles, axis=0, weights=self.weights)

        # Handle wrapped angle better than plain average
        sin_mean = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        cos_mean = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        self.x_est[2] = np.arctan2(sin_mean, cos_mean)

        diff = self.particles - self.x_est
        diff[:, 2] = np.array([wrap_angle(a) for a in diff[:, 2]])

        self.P = np.zeros((5, 5), dtype=float)
        for i in range(self.M):
            d = diff[i].reshape(5, 1)
            self.P += self.weights[i] * (d @ d.T)

    def particle_filter_step(self, dt: float):
        if self.z is None:
            return

        if not self.is_initialized:
            # Seed initial velocity states from first measurement
            self.x_est[3] = self.v_enc
            self.x_est[4] = self.w_imu

            self.particles, self.weights = self.initialize_particles()
            self.is_initialized = True

        # Predict
        new_particles = np.zeros_like(self.particles)
        for i in range(self.M):
            x_pred = self.f_x(self.particles[i], self.u, dt)

            # Add process noise
            process_noise = np.array([
                np.random.normal(0.0, 0.01),
                np.random.normal(0.0, 0.01),
                np.random.normal(0.0, 0.01),
                np.random.normal(0.0, 0.05),
                np.random.normal(0.0, 0.05),
            ], dtype=float)

            x_pred = x_pred + process_noise
            x_pred[2] = wrap_angle(x_pred[2])

            new_particles[i] = x_pred

        # Update weights
        new_weights = np.zeros(self.M, dtype=float)
        for i in range(self.M):
            new_weights[i] = self.gaussian_likelihood(new_particles[i], self.z)

        weight_sum = np.sum(new_weights)
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            new_weights = np.ones(self.M, dtype=float) / self.M
        else:
            new_weights /= weight_sum

        self.particles = new_particles
        self.weights = new_weights

        # Estimate before resampling
        self.estimate_state()

        # Resample
        self.particles = self.resample(self.particles, self.weights)
        self.weights = np.ones(self.M, dtype=float) / self.M

    def publish_odometry(self, t_now):
        px, py, theta, v_f, w_f = self.x_est
        qx, qy, qz, qw = to_quaternion(theta)

        cov6 = np.zeros((6, 6), dtype=float)
        cov6[0, 0] = self.P[0, 0]
        cov6[0, 1] = self.P[0, 1]
        cov6[1, 0] = self.P[1, 0]
        cov6[1, 1] = self.P[1, 1]
        cov6[5, 5] = self.P[2, 2]

        odom = Odometry()
        odom.header.stamp = t_now.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = float(px)
        odom.pose.pose.position.y = float(py)
        odom.pose.pose.position.z = 0.0

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance = cov6.reshape(-1).tolist()

        odom.twist.twist.linear.x = float(v_f)
        odom.twist.twist.angular.z = float(w_f)

        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose

        self.path_msg.header.stamp = odom.header.stamp
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def particle_filter(self):
        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9

        if self.last_time is None:
            self.last_time = t
            return

        dt = t - self.last_time
        self.last_time = t

        if dt <= 0.0:
            return

        self.particle_filter_step(dt)
        if self.is_initialized:
            self.publish_odometry(t_now)


def main():
    rclpy.init()
    node = ParticleFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()