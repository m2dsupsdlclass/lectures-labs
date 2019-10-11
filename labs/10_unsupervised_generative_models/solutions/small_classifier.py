from tensorflow.keras.layers import Dropout

# Use the VAE encoder to project the small training set into the latent space
small_x_train_encoded, _ = conv_encoder.predict(small_x_train, batch_size=100)

# Define a small MLP that takes the 2D vectors as input.
inp = x = Input(shape=(latent_dim,))
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(10, activation="softmax")(x)
mdl = Model(inp, x)
mdl.compile(loss="sparse_categorical_crossentropy",
            optimizer="adam", metrics=["acc"])

# Analysis:
#
# One can reach ~67% validation accuracy with this supervised model with so few
# data. Chance level is 10% (10 classes). State of the art is around 95% with all
# the labels and much deeper CNNs.
#
# Base on the previous plots, the 2D vectors make it possible to identify groups
# of images that approximately match classes but there is significant overlapping
# of the classes so we cannot expect to reach 90% level validation accuracy.
#
# The main limitation to our small model is probably the low dimensionality of the
# latent space of our VAE. We used latent_dim=2 to make it easy to visualize the
# manifold but this is probably detrimental from a predictive accuracy point of
# view. Setting latent_dim to 50 or more and retraining the VAE and re-encoding
# the small labeled training set is probably necessary to improve the quality of
# our supervised model.
#
# To improve further, one could also think of using the `z_log_var` output of the
# VAE to do latent space data augmentation of our small classification.
#
# One should also probably do image-space data augmentation as usual but that would
# require encoding each image-augmented minibatch. In keras it's possible to set
# conv_encoder.trainable = False and use it as the first layer of the supervised
# model.
#
# Finally the state of the art of semi-supervised training is not based on VAEs
# or GANs but a different semi-supervised learning scheme called "Mean Teachers":
#
# https://thecuriousaicompany.com/mean-teacher/
