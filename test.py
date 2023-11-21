# from model.model import CNN_Model
# import cv2
# from pathlib import Path
# from tensorflow.keras import optimizers
# import os
# # cnn = CNN_Model(weight_path="weight.h5")
# # model = cnn.build_model(rt=True)
# # model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])
# # print(model.summary())


# # # dataset_dir = './datasets/'

# # # for img_path in os.listdir(dataset_dir + "choice/"):
# # #     file_dir = dataset_dir + "choice/" + img_path
# # #     print(file_dir)
# # #     img = cv2.imread(str(file_dir), cv2.IMREAD_GRAYSCALE)
# # #     img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
# # #     img = img.reshape((1, 28, 28, 1))
# # #     print(img.shape)
# # #     predict = model.predict(img)
# # #     print(predict)
# # #     break
    

# # image_dir = "GiayThi-0.png"

# # img = cv2.imread(str(image_dir), cv2.IMREAD_GRAYSCALE)
# # # img = cv2.resize(img, (1200, 900))
# # print(img.shape)
# # cv2.imshow("123", img)
# # cv2.waitKey(0)


# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.datasets import mnist
# import cv2
# # Load the MNIST dataset
# (x_train, _), (_, _) = mnist.load_data()

# # Display an original image
# original_image = x_train[0]
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')
# plt.show()

# # # Define a function to add a random border to an image
# # def add_random_border(image, max_border_size=4):
# #     border_type = np.random.choice(['top', 'left', 'bottom', 'right'])

# #     if border_type == 'top':
# #         image = np.pad(image, ((0, max_border_size), (0, 0)), mode='constant', constant_values=0)
# #     elif border_type == 'left':
# #         image = np.pad(image, ((0, 0), (0, max_border_size)), mode='constant', constant_values=0)
# #     elif border_type == 'bottom':
# #         image = np.pad(image, ((max_border_size, 0), (0, 0)), mode='constant', constant_values=0)
# #     elif border_type == 'right':
# #         image = np.pad(image, ((0, 0), (max_border_size, 0)), mode='constant', constant_values=0)

# #     return image
# def add_border(image, border_size=2, border_color=0):
#     new_image = np.ones((28 + 2 * border_size, 28 + 2 * border_size)) * border_color
#     new_image[border_size:-border_size, border_size:-border_size] = image
#     return new_image

# # Apply random border augmentation to the first image
# augmented_image = add_border(original_image)

# # Display the augmented image
# cv2.imshow("123", augmented_image)
# cv2.waitKey(0)

import numpy as np
import cv2
import operator

def find_corners(polygon, limit_func, compare_func):
    """
    Input: Rectangle puzzle extract from contours
    Output: One of four cornet point depend on limit_func, compare_func
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function
    Note: (0,0) point is at the top-left

    top-left: (x+y) min
    top-right: (x-y) max
    bot-left: (x-y) min
    bot-right: (x+y) max
    """

    index, _ = limit_func(enumerate([compare_func(ptr[0][0], ptr[0][1]) for ptr in polygon]), key = operator.itemgetter(1))

    return polygon[index][0][0], polygon[index][0][1]


def find_contours(img):
    """
    contours: A tuple of all point creating contour lines, each contour is a np array of points (x,y).
    hierachy: [Next, Previous, First_Child, Parent]
    contour approximation: https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    """
    if (np.array(img).shape[2] != 1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (5,5), 0)
    edged = cv2.Canny(blured, 150, 200)
    # find contours on threshold image
    contours, hierachy =  cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort the largest contour to find the puzzle
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    polygon = contours[0]
    # find the largest rectangle-shape contour to make sure this is the puzzle
    # for con in contours:
    #     area = cv2.contourArea(con)
    #     perimeter = cv2.arcLength(con, closed = True)
    #     approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
    #     num_of_ptr = len(approx)
    #     if num_of_ptr == 4 and area > 10000:
    #         polygon = con   #finded puzzle
    #         break
    if polygon is not None:
        # find corner
        top_left = find_corners(polygon, limit_func= min, compare_func= np.add)
        top_right = find_corners(polygon, limit_func= max, compare_func= np.subtract)
        bot_left = find_corners(polygon,limit_func=min, compare_func= np.subtract)
        bot_right = find_corners(polygon,limit_func=max, compare_func=np.add)
        #Check polygon is square, if not return []
        #Set threshold rate for width and height to determine square bounding box
        # if not (0.5 < ((top_right[0]-top_left[0]) / (bot_right[1]-top_right[1]))<1.5):
        #     print("Exception 1 : Get another image to get square-shape puzzle")
        #     return [],[],[]
        # if bot_right[1] - top_right[1] == 0:
        #     print("Exception 2 : Get another image to get square-shape puzzle")
        #     return [],[],[]
        corner_list = [top_left, top_right, bot_right, bot_left]
        cv2.drawContours(img, [polygon], 0, (0,255,0), 3)

        return img, corner_list, 
        # draw_original: Img which drown contour and corner
        # corner_list: list of 4 corner points
        # original: Original imgs
    print("Can not detect puzzle")
    return [],[],[]
def preprocess(img):
    """
    Input: Original image
    Output: Gray-scale processed image
    """
    # convert RGB to gray-scale
    if (np.array(img).shape[2] != 1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (9,9), 0)
    #set a threshold
    thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    #invert so that the grid line and text are line, the rest is black
    inverted = cv2.bitwise_not(thresh, 0)
    morphy_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # Opening morphology to remove noise (while dot etc...)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, morphy_kernel)
    # dilate to increase border size
    result = cv2.dilate(morph, morphy_kernel, iterations=1)
    return result

dir = "./samples/5.jpg"

img = cv2.imread(dir)
copy = img.copy()
img = cv2.resize(img, (800, 600))
copy = cv2.resize(copy, (800, 600))
# if (np.array(img).shape[2] != 1):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #Gassian blur
# blured = cv2.GaussianBlur(gray_img, (3,3), 0)
# edged = cv2.Canny(blured, 200, 200)
# cv2.imshow("123", edged)
# cv2.waitKey(0)
img = preprocess(img)

contours, hierachy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #sort the largest contour to find the puzzle
contours = sorted(contours, key = cv2.contourArea, reverse = True)
cv2.drawContours(copy, contours, 1, (0, 255, 0), 2)
cv2.imshow("123", copy)
cv2.waitKey(0)


# for cont in contours:
#     if cv2.contourArea(cont) > 100:
#         x, y, w, h = cv2.boundingRect(cont)
#         mssv_img = gray_img[y: y+h, x: x+ w]
#         cv2.imshow("123", mssv_img)
#         cv2.waitKey(0)

        
# x, y, w, h = cv2.boundingRect(mssv)
# mssv_img = gray_img[y: y+h, x: x+ w]

# !!!!!!!!Note: Chup lai hinh thuc te sao cho xac dinh dung coutour ca to giay, firstly