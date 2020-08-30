#!/usr/bin/env python

import rospy
import math
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
from std_msgs.msg import Float64

yaw=0
theta=0
e=0
x=0
desired_angle=0


def angle_callback(msg):

    global desired_angle
    desired_angle=msg.data



    

def callback(data):
	global x
	if data.position[0]!=0:
		x=data.position[0]
	#print(x)



sub = rospy.Subscriber("/catapult/joint_states", JointState, callback)
sub2=rospy.Subscriber("/angle",Float64 ,angle_callback)
pub = rospy.Publisher("/catapult/base_rotation_controller/command", Float64, queue_size=1)
goal_status=rospy.Publisher("/goal_status",Float64,queue_size=1)

speed=Float64()


rospy.init_node("test2")



while not rospy.is_shutdown():

    error_angle=desired_angle-x
    print(desired_angle,error_angle)
    if abs(error_angle)>0.01:

        speed.data=0.3*error_angle
        pub.publish(speed)
        

    if abs(error_angle)<0.01:
        
        goal_status.publish(1.0)
        speed.data=0
        pub.publish(speed)



    



