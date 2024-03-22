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


# def find_contours(img):
#     """
#     contours: A tuple of all point creating contour lines, each contour is a np array of points (x,y).
#     hierachy: [Next, Previous, First_Child, Parent]
#     contour approximation: https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
#     """
#     if (np.array(img).shape[2] != 1):
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     #Gassian blur
#     blured = cv2.GaussianBlur(gray_img, (5,5), 0)
#     edged = cv2.Canny(blured, 150, 200)
#     # find contours on threshold image
#     contours, hierachy =  cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     #sort the largest contour to find the puzzle
#     contours = sorted(contours, key = cv2.contourArea, reverse = True)
#     polygon = contours[0]
#     # find the largest rectangle-shape contour to make sure this is the puzzle
#     # for con in contours:
#     #     area = cv2.contourArea(con)
#     #     perimeter = cv2.arcLength(con, closed = True)
#     #     approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
#     #     num_of_ptr = len(approx)
#     #     if num_of_ptr == 4 and area > 10000:
#     #         polygon = con   #finded puzzle
#     #         break
#     if polygon is not None:
#         # find corner
#         top_left = find_corners(polygon, limit_func= min, compare_func= np.add)
#         top_right = find_corners(polygon, limit_func= max, compare_func= np.subtract)
#         bot_left = find_corners(polygon,limit_func=min, compare_func= np.subtract)
#         bot_right = find_corners(polygon,limit_func=max, compare_func=np.add)
#         #Check polygon is square, if not return []
#         #Set threshold rate for width and height to determine square bounding box
#         # if not (0.5 < ((top_right[0]-top_left[0]) / (bot_right[1]-top_right[1]))<1.5):
#         #     print("Exception 1 : Get another image to get square-shape puzzle")
#         #     return [],[],[]
#         # if bot_right[1] - top_right[1] == 0:
#         #     print("Exception 2 : Get another image to get square-shape puzzle")
#         #     return [],[],[]
#         corner_list = [top_left, top_right, bot_right, bot_left]
#         cv2.drawContours(img, [polygon], 0, (0,255,0), 3)

#         return img, corner_list, 
#         # draw_original: Img which drown contour and corner
#         # corner_list: list of 4 corner points
#         # original: Original imgs
#     print("Can not detect puzzle")
#     return [],[],[]

def draw_circle_at_corners(original, ptr):
    """
    Helper function to draw circle at corners
    """

    cv2.circle(original, ptr, 5, (0,255,0), cv2.FILLED)


def find_contours(img, original):
    """
    contours: A tuple of all point creating contour lines, each contour is a np array of points (x,y).
    hierachy: [Next, Previous, First_Child, Parent]
    contour approximation: https://pyimagesearch.com/2021/10/06/opencv-contour-approximation/
    """

    # find contours on threshold image
    contours, hierachy =  cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sort the largest contour to find the puzzle
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    
    polygon = None
    # find the largest rectangle-shape contour to make sure this is the puzzle
    for con in contours:
        area = cv2.contourArea(con)
        perimeter = cv2.arcLength(con, closed = True)
        approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
        num_of_ptr = len(approx)
        if num_of_ptr == 4 and area > 100000:
            polygon = con   #finded puzzle
            break
    if polygon is not None:
        # find corner
        top_left = find_corners(polygon, limit_func= min, compare_func= np.add)
        top_right = find_corners(polygon, limit_func= max, compare_func= np.subtract)
        bot_left = find_corners(polygon,limit_func=min, compare_func= np.subtract)
        bot_right = find_corners(polygon,limit_func=max, compare_func=np.add)
        #Check polygon is square, if not return []
        #Set threshold rate for width and height to determine square bounding box
        # if not (0.5 < ((top_right[0]-top_left[0]) / (bot_right[1]-top_right[1]))<3):
        #     print("Exception 1 : Get another image to get square-shape puzzle")
        #     return [],[],[]
        # if bot_right[1] - top_right[1] == 0:
        #     print("Exception 2 : Get another image to get square-shape puzzle")
        #     return [],[],[]
        corner_list = [top_left, top_right, bot_right, bot_left]
        draw_original = original.copy()
        cv2.drawContours(draw_original, [polygon], 0, (0,255,0), 3)
        #draw circle at each corner point
        for x in corner_list:
            draw_circle_at_corners(draw_original, x)

        return draw_original, corner_list, original
        # draw_original: Img which drown contour and corner
        # corner_list: list of 4 corner points
        # original: Original imgs
    print("Can not detect puzzle")
    return [],[],[]



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
        out_ptr = np.array([[0,0],[side-1,0],[side-1,side-1], [0,side-1]],dtype="float32")
        transfrom_matrix = cv2.getPerspectiveTransform(corners, out_ptr)
        transformed_image = cv2.warpPerspective(original, transfrom_matrix, (side, side))
        return transformed_image, transfrom_matrix
    except IndexError:
        print("Can not detect corners")
    except:
        print("Something went wrong. Try another image")


def preprocess(img, gauss_filter_size = 17, thresh_block_size = 45):
    """
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


if __name__ == "__main__":

    dir = "./samples/15.jpg"
    img = cv2.imread(dir)
    original_img = img.copy()
    threshold = preprocess(img)
    # Sharpen the image using the Laplacian operator 
    # threshold = cv2.Laplacian(threshold, cv2.CV_64F)
    threshold_show = cv2.resize(threshold, (800, 600)) 
    cv2.imshow("123", threshold_show)
    cv2.waitKey(0)


    contours, hierachy =  cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    polygons = []
    # find the largest rectangle-shape contour to make sure this is the puzzle
    for con in contours:
        area = cv2.contourArea(con)
        if area > 100000:
            print(area)
        
        perimeter = cv2.arcLength(con, closed = True)
        approx = cv2.approxPolyDP(con, epsilon=0.01 * perimeter, closed =  True)
        num_of_ptr = len(approx)
        if num_of_ptr == 4 and area > 10000:
            polygons.append(con)
            if len(polygons) == 7: 
                break
    print(len(polygons))
    for con in polygons:
        print(con)
        x, y, w, h = cv2.boundingRect(con)
        cv2.drawContours(original_img, [con], -1, (0, 255, 0), 2)
        original_img_show = cv2.resize(original_img, (800, 600))
        cv2.imshow('Contours', original_img_show) 
        cv2.waitKey(0) 


    # # !!!!!!!!Note: 
    # -- FIND LARGEST PAPER  to img align -- DONE
    # -- Next: Image alignment and extract ans block