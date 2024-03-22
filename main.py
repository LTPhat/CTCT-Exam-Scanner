import cv2
import numpy as np
from collections import defaultdict
from offset import Params
from model.model import CNN_Model
from extract_blocks import preprocess, find_largest_boundary, find_corners, get_corner, find_mssv_block, warp_image
from mssv_block_utils import get_mssv_by_processing
from main_block_utils import *
from model.model import CNN_Model
import sys
import argparse
cnn = CNN_Model(weight_path= offset.weights_path).build_model(rt=True)
param = Params()


def init_argparse():
    parser = argparse.ArgumentParser(
        description="--img_dir: Image directory \n"
                    
    )
    parser.add_argument("--img_dir", required=True, help='Your image directory')
    
    return parser


def process_main_block(paper_image, model = cnn):
    """
    End-to-end process for one image
    Input:
    img: img property (dir (str) or np.array)
    """
# ------------------- MAIN BLOCKS PROCESSING --------------------------------------
    
    # ---------- Extract main block----------------
    main_block = paper_image[param.header_offset - 100: , :]
    threshold = preprocess(main_block, gauss_filter_size=15, thresh_block_size=55)

    # threshold_show = cv2.resize(threshold, (800, 600))
    # cv2.imshow("Threshold", threshold_show)
    # cv2.waitKey(0)

    # ----------- Find contour----------------------------
    main_block_contour = find_largest_boundary(threshold_img=threshold)
    # cv2.drawContours(main_block, [main_block_contour], -1, (0, 255, 0), 2)
    # main_block_show = cv2.resize(main_block, (800, 600))
    # cv2.imshow("second", main_block_show)
    # cv2.waitKey(0)

    # ----------- Image alignment -----------
    corner_list = get_corner(main_block_contour, corner_bound_offset=0)
    main_block, _ = warp_image(corner_list, main_block)

    # transformed_img_show = cv2.resize(transformed_img, (800, 600))
    # cv2.imshow("warp image", transformed_img_show)
    # cv2.waitKey(0)
    # transformed_img = cv2.resize(transformed_img, param.main_block_shape)
    # print(transformed_img.shape)

    # -------------Extract 4 ans blocks----------
    ans_columns, _ = get_blocks(main_block, test=False)

    # -------------Extract all ans lines-----------
    ans_lines = get_list_ans(ans_columns)
    # for line in ans_lines:
    #     cv2.imshow("ans line", line)
    #     cv2.waitKey(0)

    # ------------Extract all bubble choices-------------------
    list_choices = get_bubble_choice(ans_lines)
    # for choice in list_choices:
    #     cv2.imshow("choice", choice)
    #     cv2.waitKey(0)

    # -------------Get answer-------------------
    ans = get_answer(model=model, list_choice = list_choices)
    # ---------------Get true ans-------------------
    true_ans = read_true_ans("./ans_keys/16.txt", mode="txt")
    # ----------- Get score ----------------------
    score = get_score(ans, true_ans)
    return score


def process_mssv_block(paper_image, model = cnn):
    """
    Extract MSSV from paper image
    """
# ------------------- MSSV BLOCK PROCESSING --------------------------------------

    # ---------------Extract header-------------
    header = paper_image[:param.header_offset, :]
    threshold = preprocess(header, gauss_filter_size=9, thresh_block_size=25)
    # threshold_show = cv2.resize(threshold, (800, 600))
    # cv2.imshow("Threshold", threshold_show)
    # cv2.waitKey(0)

    # ---------------- Find contour---------------------
    mssv_block_contour = find_mssv_block(threshold_img=threshold)
    # cv2.drawContours(header, [mssv_block_contour], -1, (0, 255, 0), 2)
    # main_block_show = cv2.resize(header, (800, 600))
    # cv2.imshow("mssv", main_block_show)
    # cv2.waitKey(0)

    # ----------------Image alignment -------------------------
    corner_list = get_corner(mssv_block_contour, corner_bound_offset=0)
    mssv_img, _ = warp_image(corner_list, header)
    # transformed_img_show = cv2.resize(mssv_img, (800, 600))
    # cv2.imshow("processed mssv", transformed_img_show)
    # cv2.waitKey(0)

    # ---------------- Get MSSV---------------------------
    mssv = get_mssv_by_processing(model=model, mssv_img=mssv_img)
    return mssv


if __name__ == "__main__":
    #--- input argument-----
    parser = init_argparse()
    args   = parser.parse_args()
    img = args.img_dir
    # img = "./samples/15.jpg"

    
    if isinstance(img, str):
        img = cv2.imread(img)

    # ----------Preprocess: Filter - Threshold---------------------- 
    threshold = preprocess(img,gauss_filter_size=19, thresh_block_size=45)
    # ----------First img alignment: Extract paper from img-----------
    paper_contour  = find_largest_boundary(threshold_img=threshold)
    corner_list = get_corner(paper_contour)
    paper_image, _ = warp_image(corner_list, img)

    # -----------Get score from main_block
    score = process_main_block(paper_image)
    mssv = process_mssv_block(paper_image)

    print(score)
    print(mssv)
