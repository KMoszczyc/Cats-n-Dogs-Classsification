from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random

plt.rc('font', size=20)

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


class DogsCatsClassificator:
    def load_data(self):
        filenames = os.listdir("train")
        categories = []
        for filename in filenames:
            category = filename.split('.')[0]
            if category == 'dog':
                categories.append(1)
            else:
                categories.append(0)

        df_train = pd.DataFrame({
            'filename': filenames,
            'category': categories
        })
        filenamesTest = os.listdir("test1")
        dfTest = pd.DataFrame({
            'filename': filenamesTest
        })
        test_df = dfTest.reset_index(drop=True)
        return df_train, test_df

    # print(df.head())
    # df['category'].value_counts().plot.bar()
    # plt.show()

    # sample = random.choice(filenames)
    # image = load_img("train/"+sample)
    # plt.imshow(image)
    # plt.show()

    # model based on pretrained imagenet for better results
    def create_model(self):
        model = Sequential()
        model.add(VGG16(include_top=False, input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

        for layer in model.layers:
            layer.trainable = False

        model.add(Flatten())
        model.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

        model.summary()

        return model

    def train(self, model, save_path):
        df, test_df = self.load_data()
        df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
        train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
        train_df = train_df.reset_index(drop=True)
        validate_df = validate_df.reset_index(drop="True")

        # train_df['category'].value_counts().plot.bar()
        # plt.show()
        # validate_df['category'].value_counts().plot.bar()
        # plt.show()

        total_train = train_df.shape[0]
        total_validate = validate_df.shape[0]
        print('train len: ', total_train, 'val len: ', total_validate)
        batch_size = 32

        train_datagen = ImageDataGenerator(
            # rescale=1./255,
            horizontal_flip=True,
            zoom_range=0.1,
            rotation_range=20,
            shear_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            featurewise_center=True
        )
        train_datagen.mean = [123.68, 116.779, 103.939]
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            "train",
            x_col='filename',
            y_col='category',
            target_size=IMAGE_SIZE,
            class_mode='categorical',
            batch_size=batch_size
        )
        validation_generator = train_datagen.flow_from_dataframe(
            validate_df,
            "train",
            x_col='filename',
            y_col='category',
            target_size=IMAGE_SIZE,
            class_mode='categorical',
            batch_size=batch_size
        )

        # show generated images

        x, y = train_generator.next()
        print(x.shape, y.shape)
        for i in range(len(x)):
            image = x[i]
            image = image + [123.68, 116.779, 103.939]
            plt.imshow(image.astype('uint8'))
            plt.show()

        epochs = 20

        checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=0, save_best_only=False,
                                        save_weights_only=False, mode='auto', period=1)
        model.fit_generator(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            steps_per_epoch=len(train_generator),
            callbacks=[checkpoint]
        )
        model.save_weights(save_path)

    def predict(self, model, img_path):
        img = load_img(img_path, target_size=IMAGE_SIZE)

        img_arr = img_to_array(img)
        img_arr = img_arr.reshape(1, 224, 224, 3)
        img_arr = img_arr.astype('float32')
        img_arr = img_arr - [123.68, 116.779, 103.939]
        prediction = model.predict(img_arr)
        print(prediction)
        text = ''
        if prediction[0, 0] > prediction[0, 1]:
            text = 'cat: ' + str(round(prediction[0, 0], 4))
        else:
            text = 'dog: ' + str(round(prediction[0, 1], 4))

        return prediction, text

    def manualTest(self, model):
        filenamesTest = os.listdir("test1")
        plt.axis('off')
        for i in range(32):
            fig = plt.figure(figsize=(8, 8))

            for j in range(1,5):
                sample_path = "test1/" + random.choice(filenamesTest)
                prediction, text = self.predict(model, sample_path)
                img_normal = load_img(sample_path)

                fig.add_subplot(2, 2, j)
                plt.text(0, -20, text)
                plt.axis('off')
                plt.imshow(img_normal)
            plt.show()

