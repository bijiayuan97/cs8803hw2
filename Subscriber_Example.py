#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from nav_msgs.msg import Odometry

import math

import numpy as np

import gtsam

import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot

def vector3(x, y, z):
    """
    create 3d double numpy array
    """
    return np.array([x, y, z], dtype=np.float)

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z

all_prev_poses = [(0.0, 0.0, 0.0)]

prev_pose = Odometry()
PRIOR_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.1, 0.1, 0.1))
graph = gtsam.NonlinearFactorGraph()
ODOMETRY_NOISE = gtsam.noiseModel_Diagonal.Sigmas(vector3(0.1, 0.1, 0.1))
graph.add(gtsam.PriorFactorPose2(1, gtsam.Pose2(0.0, 0.0, 0.0), PRIOR_NOISE))
initial_estimate = gtsam.Values()
initial_estimate.insert(1, gtsam.Pose2(0.0, 0.0, 0.0))
prev_i = 1
max_i = 1

j = 0

def callback(msg):
    global j
    global all_prev_poses
    global prev_i
    global max_i
    global prev_pose
    global graph
    global initial_estimate
    global ODOMETRY_NOISE
    global PRIOR_NOISE
    is_one_meter = (prev_pose.pose.pose.position.x - msg.pose.pose.position.x)**2 + \
                   (prev_pose.pose.pose.position.y - msg.pose.pose.position.y)**2 >= 1.0 # move at least one meter

    if is_one_meter:
        print(msg.pose.pose.position)
        print(euler_from_quaternion(msg.pose.pose.orientation.x,
                                    msg.pose.pose.orientation.y,
                                    msg.pose.pose.orientation.z,
                                    msg.pose.pose.orientation.w))
        prev_pose = msg
        is_same_pose = min([(prev_pose.pose.pose.position.x - i[0])**2 + (prev_pose.pose.pose.position.y - i[1])**2 for i in all_prev_poses]) < 0.1
        graph.add(gtsam.BetweenFactorPose2(
            prev_i, prev_i + 1, gtsam.Pose2(msg.pose.pose.position.x, msg.pose.pose.position.y,
                                            euler_from_quaternion(msg.pose.pose.orientation.x,
                                                                  msg.pose.pose.orientation.y,
                                                                  msg.pose.pose.orientation.z,
                                                                  msg.pose.pose.orientation.w)[2]),
            ODOMETRY_NOISE
        ))
        initial_estimate.insert(prev_i+1,
                                gtsam.Pose2(msg.pose.pose.position.x,
                                            msg.pose.pose.position.y,
                                            euler_from_quaternion(msg.pose.pose.orientation.x,
                                                                  msg.pose.pose.orientation.y,
                                                                  msg.pose.pose.orientation.z,
                                                                  msg.pose.pose.orientation.w)[2]))
        prev_i = prev_i + 1
        all_prev_poses.append(vector3(msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z))
        if prev_i + 1 > max_i:
            max_i = prev_i + 1
        parameters = gtsam.GaussNewtonParams()
        parameters.setRelativeErrorTol(1e-5)
        parameters.setMaxIterations(100)
        optimizer = gtsam.GaussNewtonOptimizer(graph, initial_estimate, parameters)

        result = optimizer.optimize()
        marginals = gtsam.Marginals(graph, result)
        fig = plt.figure(0)
        for i in range(1, max_i - 1):
            print("X{} covariance:\n{}\n".format(i, marginals.marginalCovariance(i)))
            gtsam_plot.plot_pose2(0, result.atPose2(i), 0.5, marginals.marginalCovariance(i))

        plt.axis("equal")
        plt.show()
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('Pose2SLAM', anonymous=True)

    rospy.Subscriber('odom', Odometry, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
