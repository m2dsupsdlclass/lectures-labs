
from tensorflow.keras.layers import BatchNormalization


def make_conv_encoder(img_rows, img_cols, img_chns,
                      latent_dim, intermediate_dim):
    x = Input(shape=(img_rows, img_cols, img_chns))
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu')(x)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu')(x_conv)
    x_conv = BatchNormalization()(x_conv)
    x_conv = Conv2D(filters,
                    kernel_size=kernel_size,
                    padding='same', activation='relu',
                    strides=(2, 2))(x_conv)
    flat = Flatten()(x_conv)
    hidden = Dense(intermediate_dim, activation='relu')(flat)
    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)
    return Model(inputs=x, outputs=[z_mean, z_log_var],
                 name='convolutional_encoder')


conv_encoder = make_conv_encoder(img_rows, img_cols, img_chns,
                                 latent_dim, intermediate_dim)
