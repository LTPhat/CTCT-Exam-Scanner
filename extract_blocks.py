import numpy as np
import cv2
import operator
from offset import Params

param = Params()

"""
This file code includes code to extract:
- Main_block: Largest block contain 4 answer columns
- MSSV_block: BLock contain MSSV
"""

def preprocess(img, gauss_filter_size = 19, thresh_block_size = 45):
    """
    Filter-Threshold-Morph --> Preprocessing
    Input: Original image
    Output: Gray-scale processed image
    """
    # convert RGB to gray-scale
    if (np.array(img).shape[2] != 1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (gauss_filter_size, gauss_filter_size), 0)
    #set a threshold
    thresh = cv2.adaptiveThreshold(blured, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, thresh_block_size, 2)
    # result = thresh
    #invert so that the grid line and text are line, the rest is black
    inverted = cv2.bitwise_not(thresh, 0)
    morphy_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    # Opening morphology to remove noise (while dot etc...)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, morphy_kernel)
    # dilate to increase border size
    result = cv2.dilate(morph, morphy_kernel, iterations=1)
    
    return result



def find_main_blocks(threshold_img):
    """
    Find largest main_blocks (containing for 4 answer columns)
    """
    contours, hierachy =  cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Store paper contour
    polygon = None
    # find the largest rectangle-shape contour to make sure this is the puzzle
    for con in contours:
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, closed = True)
        approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
        num_of_ptr = len(approx)
        if num_of_ptr == 4 and area > param.min_area:
            # Found paper
            polygon = con
            return polygon
        else:
            raise TypeError("Can not detect paper contour")
        
    return polygon


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


def get_corner(polygon, corner_bound_offset = 25):
    """
    Extract contour corner 
    Input: polygon -- list of contours
    """
    if isinstance(polygon, list):
        raise Exception("Empty paper contour!")

    # find corner
    top_left = find_corners(polygon, limit_func= min, compare_func= np.add)
    top_right = find_corners(polygon, limit_func= max, compare_func= np.subtract)
    bot_left = find_corners(polygon,limit_func=min, compare_func= np.subtract)
    bot_right = find_corners(polygon,limit_func=max, compare_func=np.add)


    corner_list = [(top_left[0] + corner_bound_offset, top_left[1] - corner_bound_offset), 
                   (top_right[0]-corner_bound_offset, top_right[1] -corner_bound_offset),
                   (bot_right[0] - corner_bound_offset, bot_right[1] +corner_bound_offset), 
                   (bot_left[0] +corner_bound_offset, bot_left[1] + corner_bound_offset)]

    return corner_list


def warp_image(corner_list, original):
    """
    Input: 4 corner points and threshold grayscale image
    Output: Perspective transformation matrix and transformed image
    Perspective transformation: https://theailearner.com/tag/cv2-warpperspective/
    """
    try:
        corners = np.array(corner_list, dtype= "float32")
        top_left, top_right, bot_left, bot_right = corners[0], corners[1], corners[2], corners[3]
        #Get the largest side to be the side of squared transfromed puzzle
        side = int(max([
            np.linalg.norm(top_right - bot_right),
            np.linalg.norm(top_left - bot_left),
            np.linalg.norm(bot_right - bot_left),
            np.linalg.norm(top_left - top_right)
        ]))
        out_ptr = np.array([[0,0],[side,0],[side,side], [0,side]],dtype="float32")
        transfrom_matrix = cv2.getPerspectiveTransform(corners, out_ptr)
        transformed_image = cv2.warpPerspective(original, transfrom_matrix, (side, side))
        return transformed_image, transfrom_matrix
    except IndexError:
        print("Can not detect corners")
    except:
        print("Something went wrong. Try another image")



def image_alignment(img, resize_shape=param.resize_shape):
    """
    Steps for each img alignment operation

    """
    threshold = preprocess(img)
    paper_contour = find_main_blocks(threshold_img=threshold)
    corner_list = get_corner(paper_contour)
    transformed_img, _ = warp_image(corner_list, img)
    res_img = cv2.resize(transformed_img, resize_shape)
    return res_img



def find_mssv_block(threshold_img):
    """
    Find MSSV block (the second largest block of header section)
    """
    contours, hierachy =  cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    # Store contours
    polygon = []
    # find the largest rectangle-shape contour to make sure this is the puzzle
    for con in contours:
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, closed = True)
        approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
        num_of_ptr = len(approx)
        if num_of_ptr == 4 and area > 10000:
            # Found paper
            polygon.append(con)
            if len(polygon) == 3:
                return polygon[1]
        else:
            raise ValueError("Can not rectangle contour")

    if len(polygon):
        raise TypeError("Can not detect paper contour")
        
    return polygon



if __name__ == "__main__":
    # One sample
    dir = "./samples/16.jpg"
    img = cv2.imread(dir)
    print(img.shape)
    img_show = cv2.resize(img, (800, 600))
    cv2.imshow("123", img_show)
    cv2.waitKey(0)


    original_img = img.copy()
    threshold = preprocess(img)
    # Visualize threshold img
    threshold_show = cv2.resize(threshold, (800, 600)) 
    cv2.imshow("123", threshold_show)
    cv2.waitKey(0)


    paper_contour  = find_main_blocks(threshold_img=threshold)
    print(paper_contour)
    corner_list = get_corner(paper_contour)
    tf_image, _ = warp_image(corner_list, img)

    tf_image_show = cv2.resize(tf_image, (800, 600))
    cv2.imshow("transformed image", tf_image_show)
    cv2.waitKey(0)

    
    # All-in-one
    first_res = image_alignment(img)
    print(first_res.shape)
    res_show = cv2.resize(first_res[param.header_offset:, :], (800, 600))
    cv2.imshow("first res", res_show)
    cv2.waitKey(0)

    main_block = first_res[param.header_offset: , :]

    threshold = preprocess(main_block, gauss_filter_size=9, thresh_block_size=25)
    threshold_show = cv2.resize(threshold, (800, 600))
    cv2.imshow("Threshold", threshold_show)
    cv2.waitKey(0)

    bound = find_main_blocks(threshold_img=threshold)
    cv2.drawContours(main_block, [bound], -1, (0, 255, 0), 2)

    main_block_show = cv2.resize(main_block, (800, 600))
    cv2.imshow("second", main_block_show)
    cv2.waitKey(0)


    corner_list = get_corner(bound, corner_bound_offset=0)
    transformed_img, _ = warp_image(corner_list, main_block)
    transformed_img_show = cv2.resize(transformed_img, (800, 600))
    
    cv2.imshow("second", transformed_img_show)
    cv2.waitKey(0)

    cv2.imwrite("main_block16.jpg", transformed_img)


    # header = first_res[:param.header_offset, :]
    # header_show = cv2.resize(header, (800, 600))
    # cv2.imshow("header", header_show)
    # cv2.waitKey(0)
    # threshold = preprocess(header, gauss_filter_size=3, thresh_block_size=9)
    # threshold_show = cv2.resize(threshold, (800, 600))
    # cv2.imshow("Threshold", threshold_show)
    # cv2.waitKey(0)
    # bound = find_mssv_block(threshold_img=threshold)
    # cv2.drawContours(header, [bound[1]], -1, (0, 255, 0), 2)

    # main_block_show = cv2.resize(header, (800, 600))
    # cv2.imshow("second", main_block_show)
    # cv2.waitKey(0)

    # corner_list = get_corner(bound[1], corner_bound_offset=0)
    # transformed_img, _ = warp_image(corner_list, header)
    # transformed_img_show = cv2.resize(transformed_img, (800, 600))
    
    # cv2.imshow("second", transformed_img_show)
    # cv2.waitKey(0)

    # cv2.imwrite("header16.jpg", transformed_img)