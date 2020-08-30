#!/usr/bin/env python

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

pixel =64
#global alphabets 
alphabets = ['P', 'S', 'L', 'U', 'B', 'O', 'G', 'N', 'C', 'Y', 'K', 'C', 'M', 'F', 'I', 'H', 'A', 'T']

global c
global out

rospy.init_node('umic_bot',anonymous=True)
pub=rospy.Publisher('/mybot/mobile_base_controller/cmd_vel',Twist,queue_size=10)
pub2=rospy.Publisher('/mybot/gripper_extension_controller/command',Float64,queue_size=10)


c=Float64()
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
letters = ""

def detect(data):
     


    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(data, "bgr8")



    weights_path = "/home/panyam/mybot_ws/src/hitter_pole/updated_model_1.h5"

    model.load_weights(weights_path)
    #img_path ='/home/panyam/hail_UMICaana/dataset_gazeboworld/letter_gazebo/MIXED1/Screenshot from 2020-08-13 22-10-40.png'

    
    img = image
    img = cv2.resize(img, (512, 512))
    im2 = img.copy()
    #cv2.imread(img_path)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    alpha = 1.0
    beta = 50
    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            new_img[y, x] = np.clip(alpha * img[y, x] + beta, 0, 255)


    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, rect_kernel, iterations=5)
    _,contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #print("7")
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h < 100 or w < 100 or y > 300 or x > 70:
            continue
        while w - x > 480:
            x = x + 10
            w = w - 10
        while ((h - y) / float(w - x)) > 0.45 and h > 110:
            y = y + 10
            h = h - 20
        """if w*h < 30000:
            continue"""
        #rect = cv2.rectangle(im2, (x, y), (x+w, y+h),(255,0,0),2)

        cropped = im2[y:y + h, x:x + w]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # cv2_imshow(cropped)
        cropped = cv2.resize(cropped, (128, 64))
        imgs = np.array(cropped).reshape(64, 128)
        img1 = imgs[:, :64]
        img2 = imgs[:, 64:]
        '''
        img1 = img1.reshape(1, 64, 64, 1)
        img2 = img2.reshape(1, 64, 64, 1)

        # cv2_imshow(img1.reshape(64, 64))
        # cv2_imshow(img2.reshape(64, 64))

        pred1 = model.predict(img1)
        pred2 = model.predict(img2)
        '''
        alpha = 1.4
        beta = 50
        new_img_1 = np.zeros(img1.shape, img1.dtype)
        for y in range(img1.shape[0]):
            for x in range(img1.shape[1]):
                new_img_1[y,x] = np.clip(alpha*img1[y,x] + beta, 0, 255)

        new_img_2 = np.zeros(img2.shape, img2.dtype)
        for y in range(img2.shape[0]):
                for x in range(img2.shape[1]):
                    new_img_2[y,x] = np.clip(alpha*img2[y,x] + beta, 0, 255)
  
        new_img_1 = new_img_1.reshape(1, 64, 64, 1)
        new_img_2 = new_img_2.reshape(1, 64, 64, 1)

        pred1 = model.predict(new_img_1)
        pred2 = model.predict(new_img_2)

        l1= alphabets[np.argmax(pred1[0])]
        l2= alphabets[np.argmax(pred2[0])]

        letters = l1 + l2

    country1 = "PUKISTAN"   #1
    country2 = "NUNNIGA"    #2
    country3 = "BULLICHYA"  #3
    #print("8")
    
    if country1.find(letters) != -1:
        
        t0=time.time()
        t1=time.time()
        c.data = 5
        while t1-t0<=5:
            pub2.publish(c)

            t1=time.time()
        c.data=-5
        pub2.publish(c)

    elif country2.find(letters) != -1:

        t0=time.time()
        t1=time.time()
        c.data = 5.6
        while t1-t0<=5.6:
            pub2.publish(c)

            t1=time.time()
        c.data=-5
        pub2.publish(c)

    elif country3.find(letters) != -1:

        t0=time.time()
        t1=time.time()
        c.data =7
        while t1-t0<=7:
            pub2.publish(c)

            t1=time.time()
        c.data=-5
        pub2.publish(c)

    else:
        print("SOMETHING IS WRONG")
        

    print(letters)

    #cv2.imshow("Image",image)

    k = cv2.waitKey(5) & 0xFF

            
            


def publisher():


    c.data=0



    out.linear.x = 0
    out.linear.y = 0
    out.linear.z = 0
    out.angular.x = 0
    out.angular.y = 0
    out.angular.z = 0
    t0=0
    t1=0


    while not rospy.is_shutdown():


        '''
        t0=time.time()
        t1=time.time()
        out.linear.x = 0.3
        while (t1-t0)*0.3<=0.45:
            print((t1-t0)*0.3)
            pub2.publish(c)
            pub.publish(out)
            print('publishing1')
            t1=time.time()
        out.linear.x=0
        pub.publish(out)

        
        
        
        out.angular.z=-0.05     
        t0=time.time()
        t1=time.time()
        while (t1-t0)*0.05<= 1.78999:#1.59079634999:            #(math.pi)/2 :          # 1.59079632697       
            print((t1-t0)*0.3)
            pub2.publish(c)
            pub.publish(out)
            print('publishing2')
            t1=time.time()
        out.angular.z=0
        pub.publish(out)
        '''
        #path_plan(0.37,0)
        #rotate_90(-1)

        distance_X(0.32,0,1)
        out.linear.x = 0.0
        pub2.publish(c)

        rospy.sleep(2)
        rotate_90(-1)
                        #waits for sometime so that ml model can detect
        '''                  
        t0=time.time()
        t1=time.time()
        out.linear.x = 0.1
        while (t1-t0)*0.1<=0.125:
            print((t1-t0)*0.1)
            pub2.publish(c)
            pub.publish(out)
            print('publishing1')
            t1=time.time()
        out.linear.x=0
        pub.publish(out)
        '''
        
        distance(0.35,-0.034,1)
        rospy.sleep(2)
        sub1=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub1)
        rospy.sleep(8)

            
        distance(0.41,-0.538,1)
        '''
        print("go")
        t0=time.time()
        t1=time.time()
        out.linear.x = 0.3
        while (t1-t0)*0.3<=0.55:
            print((t1-t0)*0.3)
            pub2.publish(c)
            pub.publish(out)
            print('publishing1')
            t1=time.time()
        out.linear.x=0
        pub.publish(out)
        
        rospy.sleep(2)
        t0=time.time()
        t1=time.time()
        '''
        rospy.sleep(2)
        sub2=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub2)
        rospy.sleep(8)
        
        
        distance(0.41,0.5,-1)
        '''
        t0=time.time()
        t1=time.time()
        out.linear.x = -0.3
        while (t1-t0)*0.3<=1.21:
            print((t1-t0)*0.3)
            pub2.publish(c)
            pub.publish(out)
            print('publishing1')
            t1=time.time()
        out.linear.x=0
        pub.publish(out)

        rospy.sleep(2)
        
        
        t0=time.time()
        t1=time.time()
        '''
        
        rospy.sleep(2)
        sub3=rospy.wait_for_message('/mybot/mybot/camera1/image_raw',Image)
        detect(sub3)
        rospy.sleep(8)

      
        break



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
        if abs(angle_to_goal - theta) > 0.01:
            

            #speed.angular.z =0.9
            speed.angular.z =((angle_to_goal)- theta)*0.3
            speed.linear.x = 0.0
            
        
        else:
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            pub.publish(speed)
            r.sleep()
            break
            
                
        pub.publish(speed)
        #print("Target_angle={}  Current_theta:{}".format(angle_to_goal,theta))
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

        speed.linear.x = 0.2*k
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

                #print("v={}  w:{}".format(v ,t))
                #print("linearspeed5={}  angularspeed5:{}".format(speed.linear.x ,speed.angular.z))
                pub.publish(speed)
                r.sleep()
                break
                

        pub.publish(speed)
        
        #print("confirm")
        r.sleep()
        #print("not stopping") 

        
if __name__=='__main__':
	try:
		publisher()

		
	except rospy.ROSInterruptException:
		pass
