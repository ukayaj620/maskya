"""
  The architecture that will be used as the classifier is MobileNetV2 using transfer learning technique,
  where only the deep convolution layer will from MobileNetV2 will be used. The top layer or FC layer of
  MobileNetV2 will be replace with the custom FC layer.
"""

# Import Tensorflow Core
import tensorflow as tf

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
from tensorflow.keras.callbacks import LearningRateScheduler

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
from matplotlib import pyplot as plt

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

    for layer in base_model.layers:
        layer.trainable = False

    base_model_output = base_model.output
    fc_model = AveragePooling2D(pool_size=(5, 5))(base_model_output)
    fc_model = Flatten(name="flatten")(fc_model)
    fc_model = Dense(128, activation="relu")(fc_model)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(2, activation="softmax")(fc_model)

    return Model(inputs=base_model.input, outputs=fc_model)


"""
  Training section
"""

LR = 5e-3
EPOCHS = 10
BATCH_SIZE = 32

label_binarizer = LabelBinarizer()

data, labels = load_dataset(DIRECTORY, CATEGORIES, label_binarizer)

(train_x, test_x, train_y, test_y) = train_test_split(
    data, labels, test_size=0.2, stratify=labels)

model = construct_model()

augmentation = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

optimizer = Adam(lr=LR, decay=LR/EPOCHS)
model.compile(loss="binary_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])

STEP_PER_EPOCH = len(train_x) // BATCH_SIZE
VALIDATION_STEPS = len(test_x) // BATCH_SIZE

def classifier_scheduler(epoch, lr):
  return lr * tf.math.exp(-0.1)

ModelLearningRateScheduler = LearningRateScheduler(classifier_scheduler, verbose=1)

model_history = model.fit(augmentation.flow(train_x, train_y, batch_size=BATCH_SIZE),
          steps_per_epoch=STEP_PER_EPOCH,
          validation_data=(test_x, test_y),
          validation_steps=VALIDATION_STEPS,
          callbacks=[ModelLearningRateScheduler],
          epochs=EPOCHS)

model.save("./models/training_02/classifier_mobile_net_v2.h5")

"""
  Visualize Training Result
"""

plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(model_history.history['lr'])
plt.title('Learning rate')
plt.ylabel('learning rate')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')
plt.show()

"""
  Testing Classifier on Test dataset and provides Classification Report
"""

predicted_test = model.predict(test_x, batch_size=BATCH_SIZE)

predicted_test = np.argmax(predicted_test, axis=1)

print(classification_report(test_y.argmax(axis=1), predicted_test,
                            target_names=label_binarizer.classes_))
