#!/usr/bin/env python
#!/usr/bin/python
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from pylab import *
import numpy as np
from matplotlib import pyplot as plt
import cv2
from cv_bridge import CvBridge
import rospy
import tensorflow as tf
from sensor_msgs.msg import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, ZeroPadding2D, BatchNormalization, Add
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MAE
from tensorflow.keras.optimizers import Adam
#from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import glorot_uniform
import rospy
from std_msgs.msg import Float64
from tf.transformations import euler_from_quaternion
from math import atan2
import math
import time
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
x=0.0
yaw=0
e=0

angle_publisher = rospy.Publisher("/angle",Float64,queue_size=1)
goal_angle=Float64()

pixel =64
#global alphabets 
alphabets = ['P', 'S', 'L', 'U', 'B', 'O', 'G', 'N', 'C', 'Y', 'K', 'C', 'M', 'F', 'I', 'H', 'A', 'T']

predict =0


global out
global base_rotation
global throwing

rospy.init_node('umic_bot',anonymous=True)

pub=rospy.Publisher('/mybot/mobile_base_controller/cmd_vel',Twist,queue_size=10)
pub2=rospy.Publisher('/mybot/gripper_extension_controller/command',Float64,queue_size=10)


pub3=rospy.Publisher('/catapult/base_rotation_controller/command',Float64,queue_size=10)
pub4=rospy.Publisher('/catapult/throwing_controller/command',Float64,queue_size=10)

base_rotation=Float64()
throwing=Float64()


out=Twist()

global PI
PI = 3.1415926535897




x = 0.0
y = 0.0
#z = 0.0
theta = 0.0
global kp
kp = 0.5
t=0
v=0

def newOdom(msg):
    global x
    global y
    #global z
    global theta
    global v         # v = angular speed (x)
    global t         # t = angular speed(z)

    x = msg.pose.pose.position.x  # pose.position.x
    y = msg.pose.pose.position.y  # pose.position.y
    rot_q = msg.pose.pose.orientation
    (roll, pitch, theta) = euler_from_quaternion(
        [rot_q.x, rot_q.y, rot_q.z, rot_q.w])


    v = msg .twist.twist.linear.x
    t = msg .twist.twist.angular.z


'''
def callback(data):
	global x
	if data.position[0]!=0:
		x=data.position[0]
	print(x)

'''
def callback(msg):

    global yaw
    global e

    e = msg.pose[16].orientation

    (roll, pitch, yaw) = euler_from_quaternion(
        [e.x, e.y, e.z, e.w])
    #print(yaw)



def create_model():
    def res_identity(x, filters, gamma=0.0001):
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(
            x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        # second block # bottleneck (but size kept same with padding)
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(gamma))(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        # third block activation used after adding the input
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(
            x)
        x = BatchNormalization()(x)
        # x = Activation('tanh')(x)

        # add the input
        x = Add()([x, x_skip])
        x = Activation('tanh')(x)

        return x

    def res_conv(x, s, filters, gamma=0.0001):
        
        #here the input size changes
        x_skip = x
        f1, f2 = filters

        # first block
        x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.l2(gamma))(
            x)
        # when s = 2 then it is like downsizing the feature map
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        # second block
        x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(gamma))(x)
        x = BatchNormalization()(x)
        x = Activation('tanh')(x)

        # third block
        x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(
            x)
        x = BatchNormalization()(x)

        # shortcut
        x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        kernel_regularizer=regularizers.l2(gamma))(x_skip)
        x_skip = BatchNormalization()(x_skip)

        # add
        x = Add()([x, x_skip])
        x = Activation('tanh')(x)

        return x

    def resnet50(input_shape=(pixel, pixel, 1), classes=len(alphabets)):
        input_im = Input(input_shape)
        x = ZeroPadding2D(padding=(3, 3))(input_im)

        # 1st stage
        # here we perform maxpooling, see the figure above

        x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # 2nd stage
        # frm here on only conv block and identity block, no pooling

        x = res_conv(x, s=1, filters=(64, 256))
        x = res_identity(x, filters=(64, 256))
        x = res_identity(x, filters=(64, 256))

        # 3rd stage

        x = res_conv(x, s=2, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))
        x = res_identity(x, filters=(128, 512))

        # 4th stage

        x = res_conv(x, s=2, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))
        x = res_identity(x, filters=(256, 1024))

        # 5th stage

        x = res_conv(x, s=2, filters=(512, 2048))
        x = res_identity(x, filters=(512, 2048))
        x = res_identity(x, filters=(512, 2048))

        # ends with average pooling and dense connection

        x = AveragePooling2D((2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dense(len(alphabets), activation='softmax', kernel_initializer='he_normal')(x)

        # define the model

        model = Model(inputs=input_im, outputs=x, name='Resnet50')

        return model


    model = resnet50()
    opt1 = tf.keras.optimizers.Adam(learning_rate=0.00003)
    model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


model = create_model()

def detect(data):
     
    global predict

    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(data, "bgr8")



    weights_path = "/home/panyam/mybot_ws/src/hitter_pole/updated_model_1.h5"

    model.load_weights(weights_path)
    #img_path ='/home/panyam/hail_UMICaana/dataset_gazeboworld/letter_gazebo/MIXED1/Screenshot from 2020-08-13 22-10-40.png'

    
    img = image                       #cv2.imread(img_path)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    alpha = 1.4
    beta = 50
    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            new_img[y, x] = np.clip(alpha * img[y, x] + beta, 0, 255)

    im2 = new_img.copy()
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=5)
    _,contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        while h>0.6*w and y>30:
            h = h - 10
        while ((h - y) / float(w - x)) > 0.46 and h > 110:
            y = y + 10
            h = h - 10
        if h < 100 or w < 100 or w*h <28000:
            continue
        rect = cv2.rectangle(im2, (x, y), (x+w, y+h),(255,0,0),2 )

        cropped = im2[y:y + h, x:x + w]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # cv2_imshow(cropped)
        cropped = cv2.resize(cropped, (128, 64))
        imgs = np.array(cropped).reshape(64, 128)
        img1 = imgs[:, :64]
        img2 = imgs[:, 64:]

        img1 = img1.reshape(1, 64, 64, 1)
        img2 = img2.reshape(1, 64, 64, 1)

        # cv2_imshow(img1.reshape(64, 64))
        # cv2_imshow(img2.reshape(64, 64))

        pred1 = model.predict(img1)
        pred2 = model.predict(img2)

        l1= alphabets[np.argmax(pred1[0])]
        l2= alphabets[np.argmax(pred2[0])]

    letters = l1 + l2
    country1 = "PUKISTAN"   #1
    country2 = "NUNNIGA"    #2
    country3 = "BULLICHYA"  #3

    
    if country1.find(letters) != -1:
        
        predict=1
        print("TARGET - PUKISTAN")
        print(predict)

    elif country2.find(letters) != -1:

        predict=2
        print("TARGET - NUNNIGA")
        print(predict)

    elif country3.find(letters) != -1:

        predict=3
        print("TARGET - BULLICHYA")
        print(predict)

    else:
        print("SOMETHING IS WRONG")
        

    print(letters)

    #cv2.imshow("Image",image)

    k = cv2.waitKey(5) & 0xFF

h= angular_velocity

def catapult(s,f,h):

    base_rotation.data=0
    throwing.data=0


    t0=0
    t1=0


    while not rospy.is_shutdown():

        
        '''
        base_rotation.data=1*f

        while t1-t0<=s:                                 #(PI: /180)
            pub3.publish(base_rotation)
            #print('base_rotation')
            t1=time.time()
            #rospy.sleep()
        base_rotation.data=0.0
        pub3.publish(base_rotation)
        
        while t1-t0<=s:                                 #(PI: /180)
            pub3.publish(base_rotation)
            print('base_rotation')
            t1=time.time()
            #rospy.sleep()
        '''
        t0=time.time()
        t1=time.time()

        throwing.data=h

        while t1-t0<=5:                               #(PI/180):
            pub4.publish(throwing)
            print('throwing')
            t1=time.time()
            #rospy.sleep() 


        break



def distance_1(x1,y1,x2,y2):
	d=math.sqrt(((x2-x1)**2)+((y2-y1)**2))

	return d


def reverse_path_plan(px, py):

    
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
    

    speed = Twist()

    r = rospy.Rate(10)
    
    goal = Point()
    goal.x = px
    goal.y = py

    while not rospy.is_shutdown():

        inc_x = goal.x - x
        inc_y = goal.y - y

        #print("goal_x={}  goal_y:{}".format(goal.x,goal.y))
        #print("current_x={}  current_y:{}".format(x,y))
        #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))

        

        angle_to_goal = atan2(-inc_y, -inc_x)
        #print("Target_angle={}  Current_theta:{}".format(angle_to_goal,theta))
        #print(abs(angle_to_goal - theta))

        if abs(angle_to_goal - theta) > 0.1:      #0.01
            
            #print("1")
            speed.linear.x = 0.0
            #speed.angular.z = 0.3
            speed.angular.z =((angle_to_goal)- theta)*0.3
            #print("linearspeed={}  angularspeed:{}".format(v ,t))
            '''
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                print("2")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break

            '''
            if abs(inc_x )<0.05 and abs(inc_y) <0.05 :
                #print("2")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed1={}  angularspeed1:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed2={}  angularspeed2:{}".format(speed.linear.x ,speed.angular.z))
                r.sleep()
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed3={}  angularspeed3:{}".format(speed.linear.x ,speed.angular.z))
                break
            
              
        else:
            
            
            
            '''
            if inc_x >0.05 and inc_y >0.05 :

                print("3")
                speed.linear.x = 0.2
                speed.angular.z = 0.0
                pub.publish(speed)
                r.sleep()
                

            else :
                print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                pub.publish(speed)
                r.sleep()
                break
            
            '''

            #print("3")
            speed.linear.x = -0.3
            speed.angular.z = 0.0
            pub.publish(speed)


            #print("v={}  w:{}".format(v ,t))
            #print("linearspeed4={}  angularspeed4:{}".format(speed.linear.x ,speed.angular.z))
            '''
            if inc_x <0.01 and inc_y <0.01 :
                print("4")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break
             '''   
            
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                #print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 

                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed5={}  angularspeed5:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                r.sleep()
                break
                

        pub.publish(speed)
        
        #print("confirm")
        r.sleep()
        #print("not stopping") 


def rotate_90(d):


    #rospy.init_node("speed_controller")
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    #sub = rospy.Subscriber("/mybot/mobile_base_controller/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)

    speed = Twist()
    r = rospy.Rate(10)
    goal = Point()
    ##r = rospy.Rate(1000)
    
    #print(goal.x)
    #print(goal.y)
    while not rospy.is_shutdown():

        
        #print(x,y)
        #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))
        
        
        angle_to_goal = ((90*math.pi)/180)*d
        #angle_to_goal_rad = angle_to_goal * (math.pi/180)
        #print(angle_to_goal)
        #print(theta)
        #print(abs(angle_to_goal - theta))
        if abs(angle_to_goal - theta) > 0.05:
            

            #speed.angular.z =0.9
            speed.angular.z =((angle_to_goal)- theta)*1.5        #0.3
            speed.linear.x = 0.0
            
        
        else:
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            pub.publish(speed)
            r.sleep()
            break
            
                
        pub.publish(speed)

        r.sleep()
  


def path_plan(px, py):

    
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
    

    speed = Twist()

    r = rospy.Rate(10)
    
    goal = Point()
    goal.x = px
    goal.y = py

    while not rospy.is_shutdown():

        inc_x = goal.x - x
        inc_y = goal.y - y

        #print("goal_x={}  goal_y:{}".format(goal.x,goal.y))
        #print("current_x={}  current_y:{}".format(x,y))
        #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))

        

        angle_to_goal = atan2(inc_y, inc_x)
        #print("Target_angle={}  Current_theta:{}".format(angle_to_goal,theta))
        #print(abs(angle_to_goal - theta))

        if abs(angle_to_goal - theta) > 0.1:      #0.01
            
            #print("1")
            speed.linear.x = 0.0
            #speed.angular.z = 0.3
            speed.angular.z =((angle_to_goal)- theta)*0.3
            #print("linearspeed={}  angularspeed:{}".format(v ,t))
            '''
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                print("2")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break

            '''
            if abs(inc_x )<0.05 and abs(inc_y) <0.05 :
                #print("2")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed1={}  angularspeed1:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed2={}  angularspeed2:{}".format(speed.linear.x ,speed.angular.z))
                r.sleep()
                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed3={}  angularspeed3:{}".format(speed.linear.x ,speed.angular.z))
                break
            
              
        else:
            
            
            
            '''
            if inc_x >0.05 and inc_y >0.05 :

                print("3")
                speed.linear.x = 0.2
                speed.angular.z = 0.0
                pub.publish(speed)
                r.sleep()
                

            else :
                print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                pub.publish(speed)
                r.sleep()
                break
            
            '''

            #print("3")
            speed.linear.x = 0.2
            speed.angular.z = 0.0


            #print("v={}  w:{}".format(v ,t))
            #print("linearspeed4={}  angularspeed4:{}".format(speed.linear.x ,speed.angular.z))
            '''
            if inc_x <0.01 and inc_y <0.01 :
                print("4")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break
             '''   
            
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                #print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 

                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed5={}  angularspeed5:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                r.sleep()
                break
                

        pub.publish(speed)
        
        #print("confirm")
        r.sleep()
        #print("not stopping") 


def distance(px,py,k):

    
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
    

    speed = Twist()

    r = rospy.Rate(10)
    
    goal = Point()
    goal.x = px
    goal.y = py

    while not rospy.is_shutdown():

        inc_x = goal.x - x
        inc_y = goal.y - y
        #print("1")
        #print("goal_x={}  goal_y:{}".format(goal.x,goal.y))
        #print("current_x={}  current_y:{}".format(x,y))
        #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))

        speed.linear.x = 0.2*k      #0.2
        speed.angular.z = 0.0

        #print("v={}  w:{}".format(v ,t))
        #print("linearspeed4={}  angularspeed4:{}".format(speed.linear.x ,speed.angular.z))

        if abs(inc_y) <0.1 :
                #print("4")
                #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))
                speed.linear.x = 0.0
                speed.angular.z = 0.0 

                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed5={}  angularspeed5:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                r.sleep()
                break
                

        pub.publish(speed)
        
        #print("confirm")
        r.sleep()
        #print("not stopping") 
        
        '''
        angle_to_goal = atan2(inc_y, inc_x)
        print("Target_angle={}  Current_theta:{}".format(angle_to_goal,theta))
        print(abs(angle_to_goal - theta))

        if abs(angle_to_goal - theta) > 0.1:      #0.01
            
            print("1")
            speed.linear.x = 0.0
            #speed.angular.z = 0.3
            speed.angular.z =((angle_to_goal)- theta)*0.3
            print("linearspeed={}  angularspeed:{}".format(v ,t))
            
            if abs(inc_x) <0.05 and abs(inc_y) <0.05 :
                print("2")
                print("linearspeed={}  angularspeed:{}".format(speed.linear.x ,speed.angular.z))
                break

            
            if abs(inc_x )<0.05 and abs(inc_y) <0.05 :
                print("2")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                print("v={}  w:{}".format(v ,t))
                print("linearspeed1={}  angularspeed1:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                print("v={}  w:{}".format(v ,t))
                print("linearspeed2={}  angularspeed2:{}".format(speed.linear.x ,speed.angular.z))
                r.sleep()
                print("v={}  w:{}".format(v ,t))
                print("linearspeed3={}  angularspeed3:{}".format(speed.linear.x ,speed.angular.z))
                break
            
              
        
            
            
            
        

            if inc_x >0.05 and inc_y >0.05 :

                print("3")
                speed.linear.x = 0.2
                speed.angular.z = 0.0
                pub.publish(speed)
                r.sleep()
                

            else :
                print("4")
                speed.linear.x = 0.0
                speed.angular.z = 0.0 
                pub.publish(speed)
                r.sleep()
                break
            '''
            
def distance_X(px,py,k):

    
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)
    

    speed = Twist()

    r = rospy.Rate(10)
    
    goal = Point()
    goal.x = px
    goal.y = py

    while not rospy.is_shutdown():

        inc_x = goal.x - x
        inc_y = goal.y - y
        #print("1")
        #print("goal_x={}  goal_y:{}".format(goal.x,goal.y))
        #print("current_x={}  current_y:{}".format(x,y))
        #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))

        speed.linear.x = 0.3*k
        speed.angular.z = 0.0

        #print("v={}  w:{}".format(v ,t))
        #print("linearspeed4={}  angularspeed4:{}".format(speed.linear.x ,speed.angular.z))

        if abs(inc_x) <0.1 :
                #print("4")
                #print("inc_x={}  inc_y:{}".format(inc_x,inc_y))
                speed.linear.x = 0.0
                speed.angular.z = 0.0 

              
                r.sleep()
                break
                

        pub.publish(speed)
        
        #print("confirm")
        r.sleep()
        #print("not stopping") 



def orient_along(px, py):


    #rospy.init_node("speed_controller")

    #sub = rospy.Subscriber("/mybot/mobile_base_controller/odom", Odometry, newOdom)
    sub = rospy.Subscriber("/mybot/odom", Odometry, newOdom)
    pub = rospy.Publisher("/mybot/mobile_base_controller/cmd_vel", Twist, queue_size=1)

    speed = Twist()

    goal = Point()
    ##r = rospy.Rate(1000)
    goal.x = px
    goal.y = py
    #print(goal.x)
    #print(goal.y)
    while True:

        inc_x = goal.x - x
        inc_y = goal.y - y
        #print(x,y)
        #print(inc_x,inc_y)
        
        
        angle_to_goal = atan2(inc_y, inc_x)
        #angle_to_goal_rad = angle_to_goal * (math.pi/180)
        #print(angle_to_goal)
        #print(theta)
        #print(abs(angle_to_goal - theta))
        if abs(angle_to_goal - theta) > 0.01:
            

            #speed.angular.z =0.9
            speed.angular.z =((angle_to_goal)- theta)*0.3
            
        
        else:
        #    speed.linear.x = 0.0
            speed.angular.z = 0.0
            #pub.publish(speed)
            #r.sleep()
            break
            
                
        pub.publish(speed)
        #print("Target={}  Current:{}".format(angle_to_goal,theta))
        r.sleep()
        #print("not stopping") 



def point_decide(bx,by,tx,ty):
	angle_to_goal = atan2(ty-by,tx-bx)
	xf=bx-0.28*cos(angle_to_goal)
	yf=by-0.28*sin(angle_to_goal)
	return (xf,yf)

def decide_piston_speed(bx,by,tx,ty):
	d=distance(bx,by,tx,ty)

	v= math.sqrt(55*d/4)

	return v


def launch(a):

    if a==1 :

        angle_publisher.publish(0)
        reached=rospy.wait_for_message("/goal_status",Float64)

        
        catapult(15.2,1,-9.7)
        catapult(1.9422 ,1,50)

        
        
        


    if a==2 :

        angle_publisher.publish(4.437)
        reached=rospy.wait_for_message("/goal_status",Float64)


        catapult(1.9422 ,-1,-8)
        catapult(1.9422 ,1,50)

        

        
    if a==3 :

        angle_publisher.publish(2.021)
        reached=rospy.wait_for_message("/goal_status",Float64)


        catapult(0.955 ,1,-15)
        catapult(0.955 ,-1 ,50)

        



def publisher():


    global r

    
    r = rospy.Rate(1000)

    

    base_rotation.data=10
    throwing.data=0.1
    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0
    t0=0
    t1=0


    while not rospy.is_shutdown():






        
        distance_X(0.4,0.5,1)

        
        
        rospy.sleep(2)
        sub1=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub1)
        #rospy.sleep()

        print("predict")

        print(predict)

        rotate_90(1)

        distance(0.41,1,1)

        rotate_90(0)

        distance_X(1.4,1,1)

        rotate_90(-1)

        distance(0.41,0,1)

        rotate_90(0)

        #catapult(0,1,50)
        #catapult(11.2,-1,50)
        
        angle_publisher.publish(3.14)
        reached=rospy.wait_for_message("/goal_status",Float64)

        distance_X(-1.2,1,-1) #1.6
        rospy.sleep(2) 
        



        distance_X(0,1,1)
        launch(predict)
        
#end of 2nd object

        path_plan(0.65,0.55)


        rotate_90(0)


        rospy.sleep(2)
        sub2=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub2)
        print(predict)
        rospy.sleep(2)


        rotate_90(1)

        distance(0.41,1,1)

        rotate_90(0)

        distance_X(2,1,1)



        reverse_path_plan(0,0)

        rotate_90(0)

        angle_publisher.publish(3.14)
        reached=rospy.wait_for_message("/goal_status",Float64)

        distance_X(-1.3,1,-1)
#3rd object

        

        distance_X(0,1,1)

        launch(predict)

        path_plan(0.65,-0.55)

        


        rotate_90(0)


        rospy.sleep(2)
        sub2=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub2)
        rospy.sleep(2)


        rotate_90(-1)

        distance(0.41,-1,1)

        rotate_90(0)

        distance_X(2,-1,1)



        reverse_path_plan(0,0)

        rotate_90(0)

        angle_publisher.publish(3.14)
        reached=rospy.wait_for_message("/goal_status",Float64)

        distance_X(-1.3,1,-1)

        distance_X(0,1,1)

        launch(predict)
        
        
        
        break



publisher()