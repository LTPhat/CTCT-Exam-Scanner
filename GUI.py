import sys
import os
import csv
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QLineEdit, QFileDialog, QMessageBox, QVBoxLayout, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5 import QtCore
import pandas as pd
from main import process_main_block, process_mssv_block
from main_block_utils import *
from mssv_block_utils import *
from extract_blocks import *



class MultipleChoiceScanner(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multiple Choice Scanner")  # Set window title
        self.setWindowIcon(QIcon('ctct.jpg'))  # Set window icon

        self.app_name_label = QLabel("<h1>CTCT Multiple Choice Scanner</h1>")
        self.app_name_label.setAlignment(QtCore.Qt.AlignCenter)


        self.image_folder_label = QLabel("Image Folder:")
        self.image_folder_input = QLineEdit()
        self.image_folder_button = QPushButton("Browse")
        self.image_folder_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; border: none; border-radius: 4px;")

        self.dst_folder_label = QLabel("Destination Folder: ")
        self.dst_folder_input = QLineEdit()
        self.dst_folder_button = QPushButton("Browse")
        self.dst_folder_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; border: none; border-radius: 4px;")

        self.max_score_label = QLabel("Max Score:")
        self.max_score_input = QLineEdit()

        self.scan_button = QPushButton("Scan")
        self.scan_button.setStyleSheet("background-color: #008CBA; color: white; font-weight: bold; padding: 8px 16px; border: none; border-radius: 4px;")
        

        self.ans_file_label = QLabel("Answer key: (*.txt)")
        self.ans_file_input = QLineEdit()
        self.ans_file_button = QPushButton("Browse")
        self.ans_file_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; border: none; border-radius: 4px;")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        layout.addWidget(self.app_name_label)

        layout.addWidget(self.image_folder_label)
        layout.addWidget(self.image_folder_input)
        layout.addWidget(self.image_folder_button)

        layout.addWidget(self.dst_folder_label)
        layout.addWidget(self.dst_folder_input)
        layout.addWidget(self.dst_folder_button)

        
        layout.addWidget(self.ans_file_label)
        layout.addWidget(self.ans_file_input)
        layout.addWidget(self.ans_file_button)


        layout.addWidget(self.max_score_label)
        layout.addWidget(self.max_score_input)
        layout.addWidget(self.scan_button)

        self.setLayout(layout)

        self.image_folder_button.clicked.connect(self.browse_image_folder)
        self.dst_folder_button.clicked.connect(self.browse_dst_folder)
        self.ans_file_button.clicked.connect(self.browse_ans_file)
        self.scan_button.clicked.connect(self.scan_inputs)
    

    def browse_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.image_folder_input.setText(folder_path)

    def browse_dst_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Save Destination Folder")
        if folder_path:
            self.dst_folder_input.setText(folder_path)

    def browse_ans_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save ans File", "", "Txt Files (*.txt)")
        if file_path:
            self.ans_file_input.setText(file_path)

    def scan_inputs(self):

        image_folder = self.image_folder_input.text()
        dst_folder = self.dst_folder_input.text()
        ans_file = self.ans_file_input.text()
        max_score = self.max_score_input.text()

        # --------------Check validation-----------------
        if not (image_folder and dst_folder and max_score):
            QMessageBox.warning(self, "Warning", "Please fill in all fields.")
            return
        try:
            max_score = int(max_score)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Max Score must be an integer.")
            return

        # ------------Scan images and save results to CSV-----------------------
        try:
            columns = ["Filename", "MSSV", "Correct/Total", "Score"]
            # Create dataframe
            df = pd.DataFrame(columns=columns)
            # Scan images in the folder
            print("SSSSSSSSs")
            for filename in sorted(os.listdir(image_folder)):
                print(filename)
                # Process each image and calculate score
                print("Scaning {} .....".format(filename))
                img = image_folder +  "/" + filename
                if isinstance(img, str):
                    img = cv2.imread(img)

                # ----------Preprocess: Filter - Threshold---------------------- 
                threshold = preprocess(img,gauss_filter_size=19, thresh_block_size=45)
                # ----------First img alignment: Extract paper from img-----------
                paper_contour  = find_largest_boundary(threshold_img=threshold)
                corner_list = get_corner(paper_contour)
                paper_image, _ = warp_image(corner_list, img)

                # -----------Get score from main_block
                total_true, total_ans, score = process_main_block(paper_image, true_ans_dir=ans_file)
                mssv = process_mssv_block(paper_image)
                new_row = {"Filename": filename, "MSSV": str(mssv), 'Correct/Total': "{}/{}".format(total_true, total_ans), "Score": str(np.round(score, 2))}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(df)
                print(score)
                print(mssv)

            # Save csv
            df.to_csv(dst_folder + '/result.csv', index=False)

                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

        QMessageBox.information(self, "Success", "Scan completed and results saved to CSV.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MultipleChoiceScanner()
    window.setGeometry(100, 100, 400, 250)  # Set initial window size
    window.show()
    sys.exit(app.exec_())


# ! Note: CHECK INPUT AND OUTPUT