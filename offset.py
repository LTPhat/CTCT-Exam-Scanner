import numpy as np



class Params():
    def __init__(self):
        self.num_of_boxes = 5
        self.num_of_lines_each_box = 5
        self.num_of_choices = 5

        # Main-block standard shape
        self.resize_shape = (1410, 1000)
        self.header_offset = 340
        self.min_area = 10000
        self.box_height_error = 3
        self.start_x = 50
        self.bubble_width = 48

        # Model params
        self.weights_path = "./model/weight.h5"

        




