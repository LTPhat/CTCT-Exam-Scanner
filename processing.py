import cv2
import numpy as np
from offset import Params
import math
from collections import defaultdict
offset = Params()


def do_rectangles_overlap(rect1, rect2):
    """
    Check two rectangle contours overlap
    rect: [x, y, w, h]
    """
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2



def get_contour_x(contour):
    """
    Get x-coordinate value of countours
    """
    x, _, _, _ = cv2.boundingRect(contour)
    return x


def map_answer(idx):
    if idx % 5 == 0:
        answer_circle = "A"
    elif idx % 5 == 1:
        answer_circle = "B"
    elif idx % 5 == 2:
        answer_circle = "C"
    elif idx % 5 == 3:
        answer_circle = "D"
    else:
        answer_circle = "E"
    return answer_circle



def get_blocks(img, test = False, resize_first = True):
    """
    Function to extract ans blocks
    Input: Original image
    Output: 
    - img_blocks: [img, [x, y, w, h]]
    - fitered_contours: coutours of blocks_img
    """
    
    if resize_first:
        img = cv2.resize(img, offset.resize_shape)

    #  convert RGB to gray-scale
    if (np.array(img).shape[2] != 1):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Gassian blur
    blured = cv2.GaussianBlur(gray_img, (5,5), 0)
    edged = cv2.Canny(blured, 150, 200)
    
    contours, hierarchy = cv2.findContours(edged,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    useful_contours = []
    if len(contours) > 0:
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # Choose useful contours
        for cont in sorted_contours:
            if cv2.contourArea(cont) > 100000:
                useful_contours.append(cont)
        
        bounding_rectangles = [cv2.boundingRect(contour) for contour in useful_contours]
        filtered_contours = []
        img_blocks = []

        # Check overlap contours
        for i, rect1 in enumerate(bounding_rectangles):
            overlap = False
            for j, rect2 in enumerate(bounding_rectangles):
                if i != j and do_rectangles_overlap(rect1, rect2):
                    overlap = True
                    raise Exception("Overlap answer block detected. Please update new image.")
            if not overlap:
                x, y, w, h = rect1
                filtered_contours.append(useful_contours[i])
                img_blocks.append([gray_img[y: y + h, x: x + w], [x, y, w, h]])

    # Sort filted contours based on x-coordinate
    final_contours = sorted(filtered_contours, key=get_contour_x)

    # Sort block_images base on x-coordinate
    img_blocks = sorted(img_blocks, key = lambda x: x[1][0])

    assert len(final_contours) == len(img_blocks)

    if test:
        # Show all contours on original image
        cv2.drawContours(img, final_contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', img) 
        cv2.waitKey(0) 

        # Show each extracted block
        for i, block in enumerate(img_blocks):
            cv2.imshow("Block {}".format(i), block[0])
            cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img_blocks, final_contours



def get_list_ans(img_blocks):
    """
    Function to extract each answer line in each ans_block
    Input: img_blocks - [img, [x, y, w, h]]
    Output: ans_lines - [q1-img, q2-img, ....]
    """
    ans_lines = []
    for ans_block_img, ans_block_coor in img_blocks:
        # Each ans_block_img has 5 boxes
        each_box_height = math.ceil(ans_block_img.shape[0] // offset.num_of_boxes)

        # Process each box
        for i in range(offset.num_of_boxes):
            box_img = np.array(ans_block_img[i * each_box_height : (i + 1) * each_box_height, :])
            box_height = box_img.shape[0]
            box_height_error = offset.box_height_error
            box_img = box_img[box_height_error: box_height - box_height_error + 2, :]
            
            # Extract answer lines
            each_line_height = math.ceil(box_img.shape[0] / offset.num_of_lines_each_box)
            for j in range(offset.num_of_lines_each_box):
                ans_lines.append(box_img[j * each_line_height : (j + 1)* each_line_height, :])

    assert len(ans_lines) == 100
    
    return ans_lines



def get_bubble_choice(ans_lines):
    """
    Get buble choice from ans_lines
    Input: ans_lines - list of answers
    Ouput: list_choice - list of bubble choices
    """
    start = offset.start_x
    bubble_width = offset.bubble_width
    list_choice = []
    for i, ans_line in enumerate(ans_lines):
        # The last box has bigger width
        if i > 76:
            start  = offset.start_x + 10
        # Remove question id
        ans_line = ans_line[:, start: ]

        for j in range(offset.num_of_choices):
            bubble = ans_line[:, j * bubble_width : (j + 1) * bubble_width]
            bubble = cv2.threshold(bubble, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            bubble = cv2.resize(bubble, (20, 20), cv2.INTER_AREA)
            list_choice.append(bubble)

    assert len(list_choice) == 100 * 5
    
    return list_choice
    

def get_answer(model, list_choice):
    """
    Get answer from list of bubble choices
    Input: 
    - model: predict a bubble is choosen or not
    - list_choice: list of bubble choices
    Output: 
    - ans_dict : {1: ["A"], 2: ["B"], ...}
    """
    ans_dict = defaultdict(list)
    list_choice = np.array(list_choice)
    scores = model.predict(list_choice / 255.0)
    for idx, score in enumerate(scores):
        question_idx = idx // offset.num_of_choices
        # Check choice 
        if score[1] > 0.9:
            ans_letter = map_answer(idx)
            ans_dict[question_idx + 1].append(ans_letter)
    
    return ans_dict



if __name__ == "__main__":
    from model.model import CNN_Model
    cnn = CNN_Model(weight_path= offset.weights_path).build_model(rt=True)
    img_dir = "./samples/GiayThi-0.png"
    res = cv2.imread(img_dir)
    res = cv2.resize(res, offset.resize_shape)
    img_blocks, contours  = get_blocks(res[offset.header_offset:, :], test = False, resize_first=False)
    ans_lines = get_list_ans(img_blocks)
    list_choice = get_bubble_choice(ans_lines)
    ans = get_answer(model=cnn, list_choice = list_choice)
    print(ans)


