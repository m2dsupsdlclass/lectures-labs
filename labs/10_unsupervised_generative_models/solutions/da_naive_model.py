from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow as tf


def get_network(input_shape=x_source_train.shape[1:]):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 5, padding='same', activation='relu', name='conv2d_1')(inputs)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_1')(x)
    x = Conv2D(48, 5, padding='same', activation='relu', name='conv2d_2')(x)
    x = MaxPool2D(pool_size=2, strides=2, name='max_pooling2d_2')(x)
    x = Flatten(name='flatten_1')(x)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dense(100, activation='relu', name='dense_2')(x)
    digits_classifier = Dense(10, activation="softmax", name="digits_classifier")(x)

    return Model(inputs=inputs, outputs=digits_classifier)

model = get_network()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True),
    metrics=['accuracy']
)

model.summary()