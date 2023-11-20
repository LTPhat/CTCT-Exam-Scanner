import cv2
import numpy as np
from offset import Params
import math
from collections import defaultdict
from model.mnist_model import DigitModel


offset = Params()




def get_MSSV(model, header_img):
    if (np.array(header_img).shape[2] != 1):
        gray_img = cv2.cvtColor(header_img, cv2.COLOR_BGR2GRAY)

    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (5,5), 0)
    edged = cv2.Canny(blured, 150, 200)
    
    contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Get MSSV block
    mssv = contours[1]
    x, y, w, h = cv2.boundingRect(mssv)
    mssv_img = gray_img[y: y+h, x: x+ w]

    mssv_line_height = math.ceil(mssv_img.shape[0] // 11)

    mssv_line_img = mssv_img[2 : mssv_line_height + 2, :]

    # Extract each digit of MSSV
    digit_list = []
    offset = 34
    for i in range(7):
        digit_img = mssv_line_img[:,i * offset: (i+1) * offset]
        digit_img = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        digit_img = cv2.resize(digit_img, (16, 16))
        print(digit_img.shape)
        cv2.imshow("1", digit_img)
        cv2.waitKey(0)
        digit_list.append(digit_img)

    # Get prediction
    digit_list = np.array(digit_list)
    scores = model.predict(digit_list / 255.0)
    mssv = ""
    for score in scores:
        digit = np.argmax(score)
        mssv += str(digit)
    
    return mssv



if __name__ == "__main__":
    model = DigitModel(weight_path= "./model/mnist_weight.h5").build_model(rt=True)

    img_dir = "GiayThi-4.png"
    res = cv2.imread(img_dir)
    res = cv2.resize(res, offset.resize_shape)

    header = res[:offset.header_offset, :]
    # cv2.imshow("header", header)
    # cv2.waitKey(0)

    mssv = get_MSSV(model, header)
    print(mssv)
