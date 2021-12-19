"""
  The architecture that will be used as the classifier is MobileNetV2 using transfer learning technique,
  where only the deep convolution layer will from MobileNetV2 will be used. The top layer or FC layer of
  MobileNetV2 will be replace with the custom FC layer.
"""

# MobileNetV2 Archirecture
from tensorflow.keras.applications import MobileNetV2

# FC Layer Components
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

# Keras Model
from tensorflow.keras.models import Model

# Preprocessing and Data Augmentation Utility
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

# Reporting and Plotting Utility
from sklearn.metrics import classification_report

# Additional Library
import os
import numpy as np


DIRECTORY = os.getcwd() + './dataset'
CATEGORIES = ['with_mask', 'without_mask']


def load_dataset(directory, categories, label_binarizer):
    data = []
    labels = []
    for category in categories:
        directory_path = os.path.join(directory, category)
        images = os.listdir(directory_path)
        for image in images:
            image_path = os.path.join(directory_path, image)
            processed_image = load_img(image_path, target_size=(224, 224))
            processed_image = img_to_array(processed_image)
            processed_image = preprocess_input(processed_image)

            data.append(processed_image)
            labels.append(category)

    labels = label_binarizer.fit_transform(labels)
    labels = to_categorical(labels)

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def construct_model():
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    base_model_output = base_model.output
    fc_model = AveragePooling2D(pool_size=(7, 7))(base_model_output)
    fc_model = Flatten(name="flatten")(fc_model)
    fc_model = Dense(128, activation="relu")(fc_model)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(2, activation="softmax")(fc_model)

    for layer in base_model.layers:
        layer.trainable = False

    return Model(inputs=base_model.input, outputs=fc_model)


"""
  Training section
"""

LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 32

label_binarizer = LabelBinarizer()

data, labels = load_dataset(DIRECTORY, CATEGORIES, label_binarizer)

(train_x, test_x, train_y, test_y) = train_test_split(data, labels,
                                                  test_size=0.2, stratify=labels, random_state=42)
model = construct_model()
augmentation = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

optimizer = Adam(lr=LR, decay=LR/EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])

model.fit(augmentation.flow(train_x, train_y, batch_size=BATCH_SIZE),
          steps_per_epoch=len(train_x) // BATCH_SIZE,
          validation_data=(test_x, test_y),
          validation_steps=len(test_x) // BATCH_SIZE,
          epochs=EPOCHS)

model.save("./models/training_02/classifier_mobile_net_v2.h5")

"""
  Testing Classifier on Test dataset and provides Classification Report
"""

predicted_test = model.predict(test_x, batch_size=BATCH_SIZE)

predicted_test = np.argmax(predicted_test, axis=1)

print(classification_report(test_y.argmax(axis=1), predicted_test,
	target_names=label_binarizer.classes_))
