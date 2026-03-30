import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
from math import cos, sin, atan2,pi
import random

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.r = 0.033   
        self.b = 0.160

        self.R = np.diag([
            0.03**2,   
            0.15**2,   
            0.01**2 
        ]).astype(float)

        self.v_enc = 0.0
        self.w_enc = 0.0
        self.w_imu = 0.0
        self.z = None
        self.M = 1000
        self.runtime = 0

        # Timing
        self.last_time = None

        # First order lag constants. Because real motors don't hit max speed instantly. it gradually increase
        self.alpha_v = 8.0
        self.alpha_w = 10.0

        self.last_wheel_pos = None
        self.last_wheel_t = None


        self.pf_path = Path()
        self.pf_path.header.frame_id = "odom"

        self.create_subscription(JointState, "/joint_states", self.joint_callback, 10)
        self.create_subscription(Imu, "/imu", self.imu_callback, 10)
        self.create_subscription(TwistStamped, "/cmd_vel",self.cmd_callback, 10)

        self.odom_pub = self.create_publisher(Odometry, "/pf/Odom", 10)
        self.pf_path = self.create_publisher(Path, "/pf/path", 10)

    
    def joint_callback(self, msg: JointState):

        # Find which index is which wheel
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
        if dt < 0.10:
            return

        wl_prev, wr_prev = self.last_wheel_pos
        wl = (wl_pos - wl_prev) / dt   
        wr = (wr_pos - wr_prev) / dt 

        self.last_wheel_pos = (wl_pos, wr_pos)
        self.last_wheel_t = t_now

        # Differential drive forward kinematics
        self.v_enc = (self.r / 2.0) * (wl + wr)
        self.w_enc = (self.r / self.b) * (wr - wl)

        self.w_enc = - self.w_enc

        self.update_z_vector()


    def imu_callback(self, msg: Imu):
        
        self.w_imu = float(msg.angular_velocity.z)
       
        if self.z is not None:
             self.z[2, 0] = self.w_imu

        self.update_z_vector()

    def update_z_vector(self):
        self.z = np.array([
            [self.v_enc],
            [self.w_enc],
            [self.w_imu]
        ], dtype=float)

    def cmd_callback(self, msg: TwistStamped):
        v_cmd = float(msg.twist.linear.x)
        w_cmd = float(msg.twist.angular.z)
        self.u = np.array([[v_cmd], [w_cmd]], dtype=float)

    
    def f_x(self, x, u , dt) -> np.ndarray:

        px, py, theta, v, w = x.flatten()

        v_cmd, w_cmd = u.flatten()

        px_new = px + v_cmd * dt * cos(theta)
        py_new = py + v_cmd * dt * sin(theta)
        theta_new = theta + w * dt
        v_new = v + self.alpha_v * dt * (v_cmd - v)
        w_new = w + self.alpha_w * dt * (w_cmd - w)

        return np.array([ [px_new], [py_new], [theta_new], [v_new], [w_new] ])

    def h(self, x):
        v = float(x[3, 0])
        w = float(x[4, 0])
        return np.array([[v], [w], [w]], dtype=float)

    def generate_particles(self, x_t):

        particles = []

        px, py, theta, v, w = x_t.flatten()

        for _ in range(self.M):
            particle = [
                random.uniform(px - 1.0, px + 1.0),      
                random.uniform(py - 1.0, py + 1.0),      
                random.uniform(-pi, pi),
                random.uniform(v - 0.5, v + 0.5),        
                random.uniform(w - 0.5, w + 0.5)         
            ]

            particles.append(particle)

        particles = np.array(particles)

        weights = np.ones(self.M) / self.M

        return particles, weights

    
    def particle_filter_step(self):
         
        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t
         
        if self.runtime == 0:
            x_t = self.f_x(self.x, self.u,dt)
            particles, weights = self.generate_particles(x_t)

         
        else:
            for i in range(self.M):
                pass
            
            self.runtime += 1
    




def main():
    
    rclpy.init()
    node = ParticleFilter()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
