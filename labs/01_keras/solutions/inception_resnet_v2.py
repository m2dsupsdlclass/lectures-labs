from time import time
import numpy as np
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.inception_resnet_v2 import decode_predictions
from skimage.io import imread
from skimage.transform import resize

tic = time()
model = InceptionResNetV2(weights='imagenet')
print("Loaded model in {:.3}s".format(time() - tic))

image = imread('laptop.jpeg')
image_resized = resize(image, (299, 299), preserve_range=True, mode='reflect')
image_resized_batch = np.expand_dims(image_resized, axis=0)

tic = time()
preds = model.predict(preprocess_input(image_resized_batch))
print("Computed predictions in {:.3}s".format(time() - tic))

print('Predicted image labels:')
class_names, confidences = [], []
for class_id, class_name, confidence in decode_predictions(preds, top=5)[0]:
    print("    {} (synset: {}): {:0.3f}".format(class_name, class_id, confidence))