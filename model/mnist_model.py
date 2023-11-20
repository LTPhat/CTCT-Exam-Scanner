import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical



class DigitModel():
    def __init__(self, weight_path) -> None:
        self.weight_path = weight_path
        self.model = None
        self.batch_size = 64
    

    def load_data(self, augmented = True):
        # Load and preprocess the MNIST dataset
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
        test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

        # Resize images to (16, 16)
        train_images = tf.image.resize(train_images, (16, 16)).numpy()
        test_images = tf.image.resize(test_images, (16, 16)).numpy()
        train_labels = tf.keras.utils.to_categorical(train_labels)
        test_labels = tf.keras.utils.to_categorical(test_labels)

        datagen = None

        if augmented:
            datagen = ImageDataGenerator(
                rotation_range=10,
                zoom_range=0.1,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=False,
            )

        return train_images, train_labels, test_images, test_labels, datagen
    
    def build_model(self, rt = True):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(16, 16, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        if self.weight_path is not None:
            print("Load pretrained weight from {}".format(self.weight_path))
            self.model.load_weights(self.weight_path)
        if rt:
            return self.model
    
    def train(self):

        x_train, y_train, x_test, y_test, datagen = self.load_data(augmented=True)
        self.build_model(rt=False)
         # compile
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(1e-3), metrics=['acc'])

        # reduce learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=2, verbose=1, )

        # Model Checkpoint
        cpt_save = ModelCheckpoint('./model/mnist_weight.h5', save_best_only=True, monitor='val_acc', mode='max', verbose = 1)

        # Early Stopping
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1,mode='auto')

        self.model.fit(datagen.flow(x_train, y_train, batch_size=self.batch_size),
                    epochs=10,
                    validation_data=(x_test, y_test),
                    callbacks=[cpt_save, reduce_lr, early],
                    batch_size = self.batch_size)

if __name__ == "__main__":
    mnist_model = DigitModel(weight_path= None)
    mnist_model.train()