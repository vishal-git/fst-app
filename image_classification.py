TRAIN_DIR = "./data/seg_train/seg_train"
VAL_DIR = "./data/seg_test/seg_test"

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(horizontal_flip=True)

train_generator = train_data_gen.flow_from_directory(
    TRAIN_DIR, target_size=(150, 150), color_mode="rgb", batch_size=32, shuffle=True
)

val_data_gen = ImageDataGenerator(horizontal_flip=True)

val_generator = val_data_gen.flow_from_directory(
    VAL_DIR, target_size=(150, 150), color_mode="rgb", batch_size=32, shuffle=True
)

labels = train_generator.class_indices
class_mapping = dict((v, k) for k, v in labels.items())

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Lambda,
    GlobalAveragePooling2D,
    Dropout,
    Dense,
)

before_mobilenet = Sequential([Input((150, 150, 3)), Lambda(preprocess_input)])

mobilenet = MobileNetV2(input_shape=(150, 150, 3), include_top=False)

after_mobilenet = Sequential(
    [GlobalAveragePooling2D(), Dropout(0.3), Dense(6, activation="softmax")]
)

model = Sequential([before_mobilenet, mobilenet, after_mobilenet])

from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=0.00001)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.build(((None, 150, 150, 3)))

from tensorflow.keras.callbacks import ModelCheckpoint

train_cb = ModelCheckpoint("./model/", save_best_only=True)

model.fit(
    train_generator, validation_data=val_generator, callbacks=[train_cb], epochs=4
)

from tensorflow import lite

conv = lite.TFLiteConverter.from_keras_model(model)
tflite_model = conv.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)
