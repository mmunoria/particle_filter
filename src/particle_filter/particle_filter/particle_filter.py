import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
from math import cos, sin, atan2,pi, sqrt, exp
import random


def wrap_angle(a: float) -> float:
    # Because angles larger than pi or smaller than -pi are just showing off. 
    # Keep it in the [-pi, pi].
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def to_quaternion(agnle: float):
    # Since we are essentially in 2D (yaw only), x and y are strictly zero.
    qz = np.sin(agnle * 0.5)
    qw = np.cos(agnle * 0.5)
    return (0.0, 0.0, qz, qw)

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
        self.particle = np.empty()
        self.weights = np.empty()


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
    
    def resample(particles, weights):
        M = len(particles)
        weights = weights / np.sum(weights)

        indices = np.random.choice(
            range(M),
            size=M,
            p=weights
        )

        new_particles = particles[indices]

        return new_particles
    
    def p_z(self, x_t, z):

        a = 1 / ( sqrt( (2*pi)**3 * self.R ) )
        expo = exp( -0.5 * (z - self.h(x_t)).T * np.linalg.inv(self.R) * (z - self.h(x_t)) )

        return a * expo

    
    def particle_filter_step(self, dt):
        
        particles = np.empty()
        weights =  np.empty()
         
        if self.runtime == 0:
            x_t = self.f_x(self.x, self.u,dt)
            particles, weights = self.generate_particles(x_t)

        else:
            for i in range(self.particles.size):
                np.append(particles  ,self.f_x(self.particles[i], self.u,dt) )
                np.append(weights, self.p_z(x_t, self.z))
                
            self.runtime += 1
        
        self.particles = self.resample(particles ,weights)

        return particles

    def particle_filter(self):

        t_now = self.get_clock().now()
        t = t_now.nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = t
            return
        dt = t - self.last_time
        self.last_time = t

        self.particle_filter_step(dt)

        px = float(self.x[0, 0])
        py = float(self.x[1, 0])
        theta = float(self.x[2, 0])
        v_f = float(self.x[3, 0])
        w_f = float(self.x[4, 0])

        qx, qy, qz, qw = to_quaternion(theta)

        current_time_msg = t_now.to_msg()

        # Map internal 5x5 covariance matrix to standard ROS 6x6 PoseWithCovariance structure
        cov6 = np.zeros((6, 6), dtype=float)
        cov6[0, 0] = self.P[0, 0]   
        cov6[0, 1] = self.P[0, 1] 
        cov6[1, 0] = self.P[1, 0]   
        cov6[1, 1] = self.P[1, 1]   
        cov6[5, 5] = self.P[2, 2] # Map 2D orientation covariance to 3D Z-axis rotation

        odom = Odometry()
        odom.header.stamp = current_time_msg
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"

        odom.pose.pose.position.x = px
        odom.pose.pose.position.y = py
        odom.pose.pose.position.z = 0.0
        
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.pose.covariance = cov6.reshape(-1).tolist()

        odom.twist.twist.linear.x = v_f
        odom.twist.twist.angular.z = w_f
        self.odom_pub.publish(odom)





    




def main():
    
    rclpy.init()
    node = ParticleFilter()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
