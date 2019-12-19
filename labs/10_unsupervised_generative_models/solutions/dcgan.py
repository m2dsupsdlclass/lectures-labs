from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Flatten, Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model


def get_generator():
    input_noise = Input(shape=(1, 1, 100))

    x = Conv2DTranspose(1024, kernel_size=4, strides=1, padding="valid")(input_noise)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(128, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Our real images have pixels in the range [-1, +1], therefore we use the
    # tanh activation.
    # We use a single output channel as we aim to generate grayscale images, if
    # we wanted to generated RGB images 3 channels would have be needed.
    x = Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="tanh")(x)

    return Model(inputs=input_noise, outputs=x)


def get_discriminator():
    input_image = Input(shape=(64, 64, 1))

    x = Conv2D(128, kernel_size=4, strides=2, padding="same")(input_image)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(512, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1024, kernel_size=4, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # The last layer should have a sigmoid activation as we want to classify
    # between the "source" and "target" domains.
    # However we are not adding it, as the sigmoid activation will be included
    # in the Tensorflow loss providing better numerical stability.
    x = Conv2D(1, kernel_size=4, strides=1, padding="valid")(x)
    # The resulting shape of this conv is [batch_size, 1, 1, 1], these dimensions
    # can be squeezed with a Flatten into a shape of [batch_size, 1].
    x = Flatten()(x)

    return Model(inputs=input_image, outputs=x)

generator = get_generator()
discriminator = get_discriminator()

# Note that we are using LeakyReLU instead of ReLU:
#   ReLU(x) = max(0, x)
#   LeakyReLU(x, alpha=0.3) = max(alpha * x, x)
# ReLU risks to be "dead" if all inputs are negative, LeakyReLU not.