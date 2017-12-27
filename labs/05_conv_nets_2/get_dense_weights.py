from keras.applications.resnet50 import ResNet50
import numpy as np
import h5py

# load a resnet WITH top
model = ResNet50(include_top=True)
dense = model.layers[-1]

# get the weights and change their shape
w, b = dense.get_weights()
w = w.reshape((1, 1, 2048, 1000))

# save the weights in h5 format
h5f = h5py.File('weights_dense.h5','w')
h5f.create_dataset('w', data=w)
h5f.create_dataset('b', data=b)
h5f.close()
