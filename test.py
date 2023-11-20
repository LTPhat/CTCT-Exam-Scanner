# from model.model import CNN_Model
# import cv2
# from pathlib import Path
# from tensorflow.keras import optimizers
# import os
# # cnn = CNN_Model(weight_path="weight.h5")
# # model = cnn.build_model(rt=True)
# # model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])
# # print(model.summary())


# # dataset_dir = './datasets/'

# # for img_path in os.listdir(dataset_dir + "choice/"):
# #     file_dir = dataset_dir + "choice/" + img_path
# #     print(file_dir)
# #     img = cv2.imread(str(file_dir), cv2.IMREAD_GRAYSCALE)
# #     img = cv2.resize(img, (28, 28), cv2.INTER_AREA)
# #     img = img.reshape((1, 28, 28, 1))
# #     print(img.shape)
# #     predict = model.predict(img)
# #     print(predict)
# #     break
    

# image_dir = "GiayThi-0.png"

# img = cv2.imread(str(image_dir), cv2.IMREAD_GRAYSCALE)
# # img = cv2.resize(img, (1200, 900))
# print(img.shape)
# cv2.imshow("123", img)
# cv2.waitKey(0)


import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import cv2
# Load the MNIST dataset
(x_train, _), (_, _) = mnist.load_data()

# Display an original image
original_image = x_train[0]
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.show()

# # Define a function to add a random border to an image
# def add_random_border(image, max_border_size=4):
#     border_type = np.random.choice(['top', 'left', 'bottom', 'right'])

#     if border_type == 'top':
#         image = np.pad(image, ((0, max_border_size), (0, 0)), mode='constant', constant_values=0)
#     elif border_type == 'left':
#         image = np.pad(image, ((0, 0), (0, max_border_size)), mode='constant', constant_values=0)
#     elif border_type == 'bottom':
#         image = np.pad(image, ((max_border_size, 0), (0, 0)), mode='constant', constant_values=0)
#     elif border_type == 'right':
#         image = np.pad(image, ((0, 0), (max_border_size, 0)), mode='constant', constant_values=0)

#     return image
def add_border(image, border_size=2, border_color=0):
    new_image = np.ones((28 + 2 * border_size, 28 + 2 * border_size)) * border_color
    new_image[border_size:-border_size, border_size:-border_size] = image
    return new_image

# Apply random border augmentation to the first image
augmented_image = add_border(original_image)

# Display the augmented image
cv2.imshow("123", augmented_image)
cv2.waitKey(0)