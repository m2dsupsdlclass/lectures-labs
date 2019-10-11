from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.models import Model

input = base_model.layers[0].input

# Take the output of the last layer of the convnet
# layer:
x = base_model.layers[-1].output

# A 1x1 convolution, with 1000 output channels, one per class
x = Convolution2D(1000, (1, 1), name='conv1000')(x)

# Softmax on last axis of tensor to normalize the class
# predictions in each spatial area
output = SoftmaxMap(axis=-1)(x)

fully_conv_ResNet = Model(inputs=input, outputs=output)

# A 1x1 convolution applies a Dense to each spatial grid location
