import numpy as np



class Params():
    def __init__(self):
        self.num_of_boxes = 5
        self.num_of_lines_each_box = 5
        self.num_of_choices = 5

        # Main-block standard shape
        self.resize_shape = (1410, 1000)
        self.header_offset = 1500
        self.min_area = 10000
        self.box_height_error = 3
        self.start_x = 50
        self.bubble_width = 48
        self.main_block_shape = (4200, 4200)
        # Model params
        self.weights_path = "./model/weight.h5"

        




