import cv2
import numpy as np
from offset import Params
import math
from collections import defaultdict
from model.model import CNN_Model

offset = Params()




def get_MSSV_by_model(model, header_img):  # Deprecated########
    """
    Method 1: Using handwriting recognition (doesn't work well)
    """
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
        digit_list.append(digit_img)

    # Get prediction
    digit_list = np.array(digit_list)
    scores = model.predict(digit_list / 255.0)
    mssv = ""
    for score in scores:
        digit = np.argmax(score)
        mssv += str(digit)
    
    return mssv



def get_mssv_by_processing(model, mssv_img, n_lines = 11, n_cols = 7, height_err = 5, col_err = 3):
    """
    Method 2: Image processing (work well)
    """
    
    # Height and width of each bubble
    mssv_line_height = math.ceil(mssv_img.shape[0] // n_lines)
    mssv_line_width = math.ceil(mssv_img.shape[1] // n_cols)

    digit_list = []
    # Process cols
    for i in range(n_cols):
        # Get a column
        col_img = mssv_img[:, i * mssv_line_width +col_err: (i + 1) * mssv_line_width]
        # Get bubles each column
        for j in range(n_lines):
            # Ignore handwriting bubble
            if j == 0:
                continue
            digit_img = col_img[j * mssv_line_height + height_err: (j+1) * mssv_line_height, :]
            digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            digit_img = cv2.threshold(digit_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            digit_img = cv2.resize(digit_img, (20, 20))
            digit_list.append(digit_img)
    
    # Get prediction
    digit_list = np.array(digit_list)
    scores = model.predict(digit_list / 255.0)
    mssv = ""
    for i, score in enumerate(scores):
        idx = np.argmax(score)
        if idx == 1:
            mssv += str(i % 10)
    return mssv
    


if __name__ == "__main__":
    model = CNN_Model(weight_path= "./model/weight.h5").build_model(rt=True)
    img_dir = "header13.jpg"
    header = cv2.imread(img_dir)
    mssv = get_mssv_by_processing(model, header)
    print(mssv)
    
