# This code was written and run in google colab
# It is recommended the user do the same
# Please use GPU as runtime while running the code


#import statements
import numpy as np
import cv2
import os
from google.colab import drive
from google.colab.patches import cv2_imshow
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, ZeroPadding2D, BatchNormalization, Add
from keras.models import Model
from tensorflow.keras.losses import MAE
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation
from keras import backend as K
from keras import regularizers
from keras.initializers import glorot_uniform
from sklearn.utils import shuffle
drive.mount('/content/drive')
%cd '/content/drive/My Drive'

#declaring global variables
pixel = 64
symbols = []
alphabets = []

#loading the dictionary and changing brightness of original images
img_path = '/content/drive/My Drive/UMIC/Letters'
for file in os.listdir('/content/drive/My Drive/UMIC/Letters'):
  img = cv2.imread(os.path.join(img_path, file))
  img = cv2.resize(img, (pixel, pixel))
  alpha = 1
  beta = 50
  for i in range(-3, 2):
    new_img = np.zeros(img.shape, img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_img[y,x,c] = np.clip(alpha*img[y,x,c] + beta*i, 0, 255)
    symbols.append(new_img)
  alphabets.append(file[0])


#One-hot encoding the labels and adding all the images to the list data
one_hot = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]

data = []
label = []

for j in range(len(symbols)):
    symbols[j] = cv2.cvtColor(symbols[j], cv2.COLOR_BGR2GRAY)
    data.append(symbols[j])

for j in range(len(alphabets)):
  for i in range(-3, 2):
    label.append(one_hot[j])

#Adding rotated and translated images
for j in range(len(symbols)):
    for i in range(-3, 4):
        rows, cols = data[j].shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 5 * i, 1)
        res = cv2.warpAffine(data[j], M, (rows, cols))
        label.append(label[j])
        data.append(res)

for i in range(len(data)):
    data[i] = cv2.resize(data[i], (pixel, pixel))

num = len(label)

for j in range(num):
    height, width = data[j].shape[:2]

    dis_y, dis_x = height // 10, width // 10
    T = np.float32([[1, 0, dis_x], [0, 1, dis_y]])

    img_trans = cv2.warpAffine(data[j], T, (width, height))
    data.append(img_trans)
    label.append(label[j])

    T = np.float32([[1, 0, -dis_x], [0, 1, dis_y]])

    img_trans = cv2.warpAffine(data[j], T, (width, height))
    data.append(img_trans)
    label.append(label[j])

    T = np.float32([[1, 0, dis_x], [0, 1, -dis_y]])

    img_trans = cv2.warpAffine(data[j], T, (width, height))
    data.append(img_trans)
    label.append(label[j])

    T = np.float32([[1, 0, -dis_x], [0, 1, -dis_y]])

    img_trans = cv2.warpAffine(data[j], T, (width, height))
    data.append(img_trans)
    label.append(label[j])

#Adding blurring and filters with appropriate kernels
for j in range(num):
  res_img = cv2.GaussianBlur(data[j], (3, 3), 0)
  data.append(res_img)
  label.append(label[j])


for j in range(num):
  res_img = cv2.medianBlur(data[j], 5)
  data.append(res_img)
  label.append(label[j])


for j in range(num):
  res_img = cv2.bilateralFilter(data[j], 11, 200, 200)
  data.append(res_img)
  label.append(label[j])

#Inverting each image and adding them to the dataset
l = len(data)
for j in range(l):
  res_img = cv2.bitwise_not(data[j])
  data.append(res_img)
  label.append(label[j])


#converting the images into numpy array to use in ml model
images = np.array(data).reshape(-1, pixel, pixel, 1)
y = np.array(label)


#Defining the ML model and its's functions
def res_identity(x, filters, gamma=0.0001):
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(gamma))(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(x)
    x = BatchNormalization()(x)
    # x = Activation('tanh')(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation('tanh')(x)

    return x

def res_conv(x, s, filters, gamma = 0.0001):
    x_skip = x
    f1, f2 = filters

  # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.l2(gamma))(x)
  # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

  # second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=regularizers.l2(gamma))(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

  #third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=regularizers.l2(gamma))(x)
    x = BatchNormalization()(x)

  # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=regularizers.l2(gamma))(x_skip)
    x_skip = BatchNormalization()(x_skip)

  # add
    x = Add()([x, x_skip])
    x = Activation('tanh')(x)

    return x

def resnet50(input_shape = (pixel, pixel, 1), classes = len(alphabets)):

    input_im = Input(input_shape)
    x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

    x = Conv2D(64, kernel_size=(5, 5), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage
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


#initiating and compiling the model
model = resnet50()
batch_size = 200

opt = tf.keras.optimizers.SGD(learning_rate=0.5)
opt1 = tf.keras.optimizers.Adam(learning_rate=0.00003)
model.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])

#making test and train sets
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2)

#fitting the model
history = model.fit(X_train, y_train, epochs= 200, batch_size=batch_size, validation_data=(X_test, y_test))

#Plotting the accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='bottom right')
plt.show()