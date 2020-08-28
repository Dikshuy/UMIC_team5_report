#This code was again run in colab for testing purposes
#the final code might be different to enable it to run with python 2.7

#import statements
import numpy as np
import cv2
from google.colab import drive
import tensorflow as tf

drive.mount('/content/drive')
%cd '/content/drive/My Drive'

#loading the model
model = tf.keras.models.load_model("/content/drive/My Drive/UMIC/updated_model_1.h5")

#declaring variables
alphabets = ['P', 'S', 'L', 'U', 'B', 'O', 'G', 'N', 'C', 'Y', 'K', 'C', 'M', 'F', 'I', 'H', 'A', 'T']

#path of test image
img_path = '/content/drive/My Drive/image_raw_screenshot_7726.08.2020.png'

#loading the image and pre-processing
img = cv2.imread(img_path)

#resizing the image to ensure consistency
img = cv2.resize(img, (512, 512))
im2 = img.copy()
img = cv2.GaussianBlur(img, (3, 3), 0)
alpha = 1
beta = 50
new_img = np.zeros(img.shape, img.dtype)
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        new_img[y, x] = np.clip(alpha * img[y, x] + beta, 0, 255)

gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

#thresholding and getting a rectangular element to enable easy contour detection
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilation = cv2.dilate(thresh, rect_kernel, iterations=5)

#detecting contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:

    #getting dimensions of the current contour
    x, y, w, h = cv2.boundingRect(cnt)

    #imposing size restrictions in perspective of current gazebo test images
    if h < 100 or w < 100:
        continue

    #another size constraint statement for length of rectangle
    while (w - x) > 480:
        x = x + 10
        w = w - 10

    #imposing ratio constraint since we know the images are two squares put together
    while ((h - y) / float(w - x)) > 0.45 and h > 110:
        y = y + 10
        h = h - 20


    #cropping the rectangle out from the earlier saved copy of the original image
    cropped = im2[y:y + h, x:x + w]
    cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    #resizing the image
    cropped = cv2.resize(cropped, (128, 64))

    #converting the image into a numpy array
    imgs = np.array(cropped).reshape(64, 128)

    #segmenting character 1 and 2 out of the cropped image
    img1 = imgs[:, :64]
    img2 = imgs[:, 64:]

    #performing some pre-processing to bring out high level features before passing to the ML model
    alpha = 1.4
    beta = 50
    new_img_1 = np.zeros(img1.shape, img1.dtype)
    for y in range(img1.shape[0]):
        for x in range(img1.shape[1]):
            new_img_1[y, x] = np.clip(alpha * img1[y, x] + beta, 0, 255)

    new_img_2 = np.zeros(img2.shape, img2.dtype)
    for y in range(img2.shape[0]):
        for x in range(img2.shape[1]):
            new_img_2[y, x] = np.clip(alpha * img2[y, x] + beta, 0, 255)

    #resizing before passing to model
    new_img_1 = new_img_1.reshape(1, 64, 64, 1)
    new_img_2 = new_img_2.reshape(1, 64, 64, 1)

    #predicting using the pre-trained model
    pred1 = model.predict(new_img_1)
    pred2 = model.predict(new_img_2)

    #assigning l1 and l2 which contain the letters
    l1 = alphabets[np.argmax(pred1[0])]
    l2 = alphabets[np.argmax(pred2[0])]
    letters = l1 + l2