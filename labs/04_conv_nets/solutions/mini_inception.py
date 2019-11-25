from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.models import Model


def inception_layer(tensor, n_filters):
    branch1x1 = Conv2D(n_filters, kernel_size=(1, 1), activation="relu", padding="same")(tensor)
    branch5x5 = Conv2D(n_filters, kernel_size=(5, 5), activation="relu", padding="same")(tensor)
    branch3x3 = Conv2D(n_filters, kernel_size=(3, 3), activation="relu", padding="same")(tensor)

    branch_pool = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding="same")(tensor)

    output = Concatenate(axis=-1)(
        [branch1x1, branch5x5, branch3x3, branch_pool]
    )
    return output


input_tensor = Input(shape=input_shape)
x = Conv2D(16, kernel_size=(5, 5), padding="same")(input_tensor)
x = inception_layer(x, 32)
x = Flatten()(x)
output_tensor = Dense(10, activation="softmax")(x)

mini_inception = Model(inputs=input_tensor, outputs=output_tensor)

mini_inception.summary()
