from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

model = ResNet50(include_top=False)
input = model.layers[0].input

# Remove the average pooling layer
output = model.layers[-2].output
headless_conv = Model(inputs=input, outputs=output)
